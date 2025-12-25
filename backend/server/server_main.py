"""
LinguaBridge Server Main
========================
FastAPI + Socket.IO server for real-time translation.

Endpoints:
    GET /api/ping - Health check
    GET /api/status - Engine status

Socket.IO Events:
    voice_chunk -> transcription_result
    translate_text -> translation_result
    synthesize_speech -> audio_result
    full_pipeline -> pipeline_result
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import socketio
from dotenv import load_dotenv

# Load environment
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# =============================================================================
# ENGINE IMPORTS (lazy-loaded singletons)
# =============================================================================
from .engine_stt import get_stt_engine
from .engine_nmt import get_nmt_engine
from .engine_tts import get_tts_engine
from .utils import wav_bytes_to_numpy, numpy_to_wav_bytes

# =============================================================================
# SOCKET.IO SETUP
# =============================================================================
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=os.getenv("CORS_ALLOWED_ORIGINS", "*").split(","),
    logger=True,
    engineio_logger=False,
)

# =============================================================================
# FASTAPI LIFESPAN
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize engines on startup, cleanup on shutdown."""
    logger.info("🚀 Starting LinguaBridge server...")
    
    # Pre-load engines (optional - can be lazy)
    try:
        logger.info("Loading STT engine...")
        stt = get_stt_engine()
        logger.info("Loading NMT engine...")
        nmt = get_nmt_engine()
        logger.info("Loading TTS engine...")
        tts = get_tts_engine()
        logger.info("✅ All engines ready!")
    except Exception as e:
        logger.warning(f"⚠️ Engine preload failed: {e}")
    
    yield
    
    logger.info("👋 Shutting down LinguaBridge server...")


# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="LinguaBridge API",
    description="Offline real-time translation server",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Socket.IO
asgi_app = socketio.ASGIApp(sio, other_asgi_app=app)

# =============================================================================
# REST ENDPOINTS
# =============================================================================
@app.get("/api/ping")
async def ping():
    """Health check endpoint."""
    return {"ok": True}


@app.get("/api/status")
async def status():
    """Get engine status and system info."""
    stt = get_stt_engine()
    nmt = get_nmt_engine()
    tts = get_tts_engine()
    
    return {
        "stt": {
            "available": True,
            "model": stt.get_model_info().get("model_name", "unknown"),
            "loaded": stt.get_model_info().get("loaded", False),
        },
        "nmt": {
            "available": nmt.get_model_info().get("available", False),
            "pairs": nmt.get_model_info().get("installed_pairs", []),
        },
        "tts": {
            "available": tts.get_model_info().get("available", False),
            "voices": [v["key"] for v in tts.get_model_info().get("voices", []) if v.get("installed")],
        },
        "server": {
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", 8000)),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
        }
    }


# =============================================================================
# SOCKET.IO EVENT HANDLERS
# =============================================================================
@sio.event
async def connect(sid, environ):
    """Handle client connection."""
    logger.info(f"Client connected: {sid}")


@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {sid}")


@sio.event
async def voice_chunk(sid, data):
    """
    Handle incoming voice chunk for STT.
    
    Input: binary WAV audio data
    Output: emits 'transcription_result' with {text, is_final}
    """
    try:
        stt = get_stt_engine()
        
        # Convert WAV bytes to numpy array
        audio = wav_bytes_to_numpy(data)
        
        # Transcribe
        text = stt.transcribe(audio)
        
        await sio.emit("transcription_result", {
            "text": text,
            "is_final": True,
        }, to=sid)
        
        logger.debug(f"STT result: {text[:50]}..." if len(text) > 50 else f"STT result: {text}")
        
    except Exception as e:
        logger.error(f"STT error: {e}")
        await sio.emit("error", {"type": "stt", "message": str(e)}, to=sid)


@sio.event
async def translate_text(sid, data):
    """
    Handle text translation request.
    
    Input: {text, source_lang, target_lang}
    Output: emits 'translation_result' with {original, translated, source, target}
    """
    try:
        nmt = get_nmt_engine()
        
        text = data.get("text", "")
        source = data.get("source_lang", "en")
        target = data.get("target_lang", "hi")
        
        translated = nmt.translate(text, source, target)
        
        await sio.emit("translation_result", {
            "original": text,
            "translated": translated,
            "source": source,
            "target": target,
        }, to=sid)
        
        logger.debug(f"NMT: {text[:30]} -> {translated[:30]}")
        
    except Exception as e:
        logger.error(f"NMT error: {e}")
        await sio.emit("error", {"type": "nmt", "message": str(e)}, to=sid)


@sio.event
async def synthesize_speech(sid, data):
    """
    Handle TTS request.
    
    Input: {text, voice} where voice is 'en_US' or 'hi_IN'
    Output: emits 'audio_result' with binary WAV data
    """
    try:
        tts = get_tts_engine()
        
        text = data.get("text", "")
        voice = data.get("voice", "en_US")
        
        # Synthesize
        audio_array = tts.synthesize(text, voice_key=voice)
        
        # Convert to WAV bytes
        wav_bytes = numpy_to_wav_bytes(audio_array, sample_rate=22050)
        
        await sio.emit("audio_result", wav_bytes, to=sid)
        
        logger.debug(f"TTS: {len(audio_array)} samples for '{text[:30]}...'")
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        await sio.emit("error", {"type": "tts", "message": str(e)}, to=sid)


@sio.event
async def full_pipeline(sid, data):
    """
    Full STT -> NMT -> TTS pipeline.
    
    Input: {audio: binary, source_lang, target_lang, voice}
    Output: emits 'pipeline_result' with {original, translated, audio: binary}
    """
    try:
        stt = get_stt_engine()
        nmt = get_nmt_engine()
        tts = get_tts_engine()
        
        audio_bytes = data.get("audio", b"")
        source = data.get("source_lang", "en")
        target = data.get("target_lang", "hi")
        voice = data.get("voice", "hi_IN" if target == "hi" else "en_US")
        
        # STT
        audio = wav_bytes_to_numpy(audio_bytes)
        original_text = stt.transcribe(audio)
        
        # NMT
        translated_text = nmt.translate(original_text, source, target)
        
        # TTS
        output_audio = tts.synthesize(translated_text, voice_key=voice)
        output_wav = numpy_to_wav_bytes(output_audio, sample_rate=22050)
        
        await sio.emit("pipeline_result", {
            "original": original_text,
            "translated": translated_text,
            "audio": output_wav,
        }, to=sid)
        
        logger.info(f"Pipeline: '{original_text[:30]}' -> '{translated_text[:30]}'")
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        await sio.emit("error", {"type": "pipeline", "message": str(e)}, to=sid)


# =============================================================================
# STATIC FILES (Frontend)
# =============================================================================
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    logger.info(f"Serving static files from: {STATIC_DIR}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Run the server directly."""
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port} (debug={debug})")
    
    uvicorn.run(
        asgi_app,
        host=host,
        port=port,
        reload=debug,
    )


if __name__ == "__main__":
    main()
