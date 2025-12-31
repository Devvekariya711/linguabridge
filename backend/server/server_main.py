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
    import numpy as np
    from .constants import LLM_ENABLED, MEMORY_ENABLED, PRELOAD_MODELS
    
    logger.info("Starting LinguaBridge server...")
    
    # Initialize translation memory database
    if MEMORY_ENABLED:
        try:
            from . import translation_memory
            translation_memory.init_db()
            logger.info("Translation memory initialized")
        except Exception as e:
            logger.warning(f"Memory init failed: {e}")
    
    if PRELOAD_MODELS:
        logger.info("Pre-loading models for fast first request...")
        
        try:
            # STT
            logger.info("[1/4] Loading STT engine (Whisper)...")
            stt = get_stt_engine()
            dummy_audio = np.zeros(1600, dtype=np.float32)
            stt.transcribe(dummy_audio)
            logger.info("      STT ready!")
            
            # NMT (Argos)
            logger.info("[2/4] Loading NMT engine (Argos)...")
            nmt = get_nmt_engine()
            nmt.translate("hello", "en", "hi")
            logger.info("      NMT ready!")
            
            # LLM (Ollama)
            if LLM_ENABLED:
                logger.info("[3/4] Checking LLM engine (Ollama)...")
                try:
                    from . import engine_llm
                    if engine_llm.is_llm_available():
                        logger.info(f"      LLM ready: {engine_llm.DEFAULT_MODEL}")
                    else:
                        logger.warning("      LLM not available (ollama not running)")
                except Exception as e:
                    logger.warning(f"      LLM check failed: {e}")
            else:
                logger.info("[3/4] LLM disabled in config")
            
            # TTS
            logger.info("[4/4] Loading TTS engine (Piper)...")
            tts = get_tts_engine()
            tts.load_voice("en_US")
            tts.load_voice("hi_IN")
            tts.synthesize("test", voice_key="en_US")
            logger.info("      TTS ready!")
            
            logger.info("=" * 50)
            logger.info("ALL ENGINES LOADED - Ready for fast requests!")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.warning(f"Engine preload failed: {e}")
            logger.warning("First request may be slow due to lazy loading")
    
    yield
    
    logger.info("Shutting down LinguaBridge server...")


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


@app.get("/api/memory/stats")
async def memory_stats():
    """Get translation memory statistics."""
    try:
        from . import translation_memory
        stats = translation_memory.get_stats()
        return {"ok": True, "stats": stats}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/memory/clear")
async def clear_memory():
    """
    Clear all translation memory and vector DB.
    Use with caution - this deletes all cached translations!
    """
    try:
        from . import translation_memory
        translation_memory.clear_memory()
        return {
            "ok": True,
            "message": "Translation memory cleared successfully"
        }
    except Exception as e:
        logger.error(f"Failed to clear memory: {e}")
        return {"ok": False, "error": str(e)}


# =============================================================================
# AUDIO DEVICE ENDPOINTS
# =============================================================================

@app.get("/api/audio/devices")
async def list_audio_devices():
    """List all available audio input/output devices."""
    try:
        from . import audio_devices
        devices = audio_devices.list_all_devices()
        summary = audio_devices.get_devices_summary()
        
        return {
            "ok": True,
            "devices": [d.to_dict() for d in devices],
            "summary": summary,
        }
    except Exception as e:
        logger.error(f"Failed to list audio devices: {e}")
        return {"ok": False, "error": str(e)}


@app.get("/api/audio/bluetooth")
async def list_bluetooth_devices():
    """List Bluetooth audio devices only."""
    try:
        from . import audio_devices
        devices = audio_devices.list_bluetooth_devices()
        
        return {
            "ok": True,
            "devices": [d.to_dict() for d in devices],
            "count": len(devices),
        }
    except Exception as e:
        logger.error(f"Failed to list Bluetooth devices: {e}")
        return {"ok": False, "error": str(e)}


from pydantic import BaseModel

class DeviceSelection(BaseModel):
    playback_device_id: int | None = None
    capture_device_id: int | None = None

@app.post("/api/audio/select")
async def select_audio_devices(selection: DeviceSelection):
    """Select preferred audio devices for playback and capture."""
    try:
        from . import audio_devices
        
        if selection.playback_device_id is not None:
            audio_devices.DevicePreference.set_playback(selection.playback_device_id)
        
        if selection.capture_device_id is not None:
            audio_devices.DevicePreference.set_capture(selection.capture_device_id)
        
        return {
            "ok": True,
            "playback_device": audio_devices.DevicePreference.get_playback(),
            "capture_device": audio_devices.DevicePreference.get_capture(),
        }
    except Exception as e:
        logger.error(f"Failed to select audio devices: {e}")
        return {"ok": False, "error": str(e)}


@app.post("/api/audio/test-playback")
async def test_playback_device(device_id: int | None = None):
    """Test playback device by playing a tone."""
    try:
        from . import audio_devices
        success = audio_devices.test_playback_device(device_id)
        return {"ok": success}
    except Exception as e:
        logger.error(f"Playback test failed: {e}")
        return {"ok": False, "error": str(e)}


@app.post("/api/audio/test-capture")
async def test_capture_device(device_id: int | None = None):
    """Test capture device by recording audio."""
    try:
        from . import audio_devices
        result = audio_devices.test_capture_device(device_id)
        return {"ok": result.get("success", False), **result}
    except Exception as e:
        logger.error(f"Capture test failed: {e}")
        return {"ok": False, "error": str(e)}


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
    
    # Cleanup streaming session
    try:
        from . import stt_streamer
        from . import pipeline
        stt_streamer.end_session(sid)
        pipeline.reset_session(sid)
    except Exception as e:
        logger.debug(f"Session cleanup error: {e}")


# =============================================================================
# STREAMING EVENTS (Live Translation)
# =============================================================================
@sio.event
async def start_stream(sid, data):
    """
    Start live streaming session.
    Client should call this before sending audio frames.
    
    Input: {source_lang, target_lang}
    """
    try:
        from . import stt_streamer
        
        source_lang = data.get("source_lang", "en")
        target_lang = data.get("target_lang", "hi")
        
        # Store language settings for session
        await sio.save_session(sid, {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "streaming": True,
        })
        
        # Start streaming STT
        await stt_streamer.start_session(sio, sid)
        
        logger.info(f"Started stream [{sid}]: {source_lang} → {target_lang}")
        
        await sio.emit("stream_started", {
            "source_lang": source_lang,
            "target_lang": target_lang,
        }, to=sid)
        
    except Exception as e:
        logger.error(f"Start stream error: {e}")
        await sio.emit("error", {"type": "stream", "message": str(e)}, to=sid)


@sio.event
async def stop_stream(sid, data=None):
    """
    Stop live streaming session.
    Triggers final LLM polish pass.
    """
    try:
        from . import stt_streamer
        from . import pipeline
        
        session = await sio.get_session(sid)
        source_lang = session.get("source_lang", "en")
        target_lang = session.get("target_lang", "hi")
        
        # Get final text for LLM polish
        stt_session = stt_streamer.SESSIONS.get(sid)
        if stt_session and stt_session.last_full_text:
            await pipeline.handle_final_pass(
                sio, sid,
                stt_session.last_full_text,
                source_lang, target_lang
            )
        
        # Cleanup
        stt_streamer.end_session(sid)
        pipeline.reset_session(sid)
        
        logger.info(f"Stopped stream [{sid}]")
        
        await sio.emit("stream_stopped", {}, to=sid)
        
    except Exception as e:
        logger.error(f"Stop stream error: {e}")


@sio.event
async def audio_frame(sid, data):
    """
    Handle incoming audio frame for live STT.
    
    Input: {data: PCM16 bytes, order: int}
    """
    try:
        from . import stt_streamer
        
        if isinstance(data, dict):
            pcm_bytes = data.get("data", b"")
            order = data.get("order", 0)
        else:
            pcm_bytes = data
            order = 0
        
        await stt_streamer.handle_audio_frame(sio, sid, pcm_bytes, order)
        
    except Exception as e:
        logger.error(f"Audio frame error: {e}")


@sio.event
async def stt_partial(sid, data):
    """
    Handle partial STT result and send through translation pipeline.
    This is called internally by stt_streamer.
    
    Input: {text, full_text}
    """
    try:
        from . import pipeline
        
        session = await sio.get_session(sid)
        source_lang = session.get("source_lang", "en")
        target_lang = session.get("target_lang", "hi")
        
        text = data.get("text", "")
        
        if text:
            await pipeline.handle_stt_partial(
                sio, sid, text,
                source_lang, target_lang
            )
        
    except Exception as e:
        logger.error(f"STT partial handler error: {e}")


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
        use_llm = data.get("use_llm", True)  # Client can disable LLM
        
        # Use smart_translate with memory -> LLM -> Argos fallback
        translated, meta = nmt.smart_translate(
            text, source, target, use_llm=use_llm
        )
        
        await sio.emit("translation_result", {
            "original": text,
            "translated": translated,
            "source": source,
            "target": target,
            "translation_source": meta.get("source", "unknown"),
            "latency": meta.get("latency", 0),
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
        
        # NMT with smart_translate (memory -> LLM -> Argos)
        use_llm = data.get("use_llm", True)
        translated_text, nmt_meta = nmt.smart_translate(
            original_text, source, target, use_llm=use_llm
        )
        
        # TTS
        output_audio = tts.synthesize(translated_text, voice_key=voice)
        output_wav = numpy_to_wav_bytes(output_audio, sample_rate=22050)
        
        await sio.emit("pipeline_result", {
            "original": original_text,
            "translated": translated_text,
            "audio": output_wav,
            "translation_source": nmt_meta.get("source", "unknown"),
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
