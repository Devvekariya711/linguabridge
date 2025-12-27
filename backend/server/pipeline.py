"""
LinguaBridge Streaming Pipeline
================================
Parallel NMT + TTS processing for low-latency live translation.
"""

import asyncio
import logging
import time
import base64
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Callable
import io
import wave
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
from .constants import get_env_int, get_env_bool, LLM_MIN_WORDS

NMT_WORKERS = get_env_int("NMT_WORKERS", 2)
TTS_WORKERS = get_env_int("TTS_WORKERS", 2)
LIVE_CHUNK_MAX_WORDS = get_env_int("LIVE_CHUNK_MAX_WORDS", 4)

# =============================================================================
# THREAD POOLS
# =============================================================================
_nmt_executor: Optional[ThreadPoolExecutor] = None
_tts_executor: Optional[ThreadPoolExecutor] = None


def get_nmt_executor() -> ThreadPoolExecutor:
    global _nmt_executor
    if _nmt_executor is None:
        _nmt_executor = ThreadPoolExecutor(max_workers=NMT_WORKERS, thread_name_prefix="nmt")
    return _nmt_executor


def get_tts_executor() -> ThreadPoolExecutor:
    global _tts_executor
    if _tts_executor is None:
        _tts_executor = ThreadPoolExecutor(max_workers=TTS_WORKERS, thread_name_prefix="tts")
    return _tts_executor


# =============================================================================
# TRANSLATION HEURISTICS
# =============================================================================
def has_named_entity(text: str) -> bool:
    """Simple named entity detection (capitalized words)."""
    words = text.split()
    for word in words:
        # Skip first word (sentence start) and short words
        if len(word) > 2 and word[0].isupper() and word[1:].islower():
            return True
    return False


def choose_translator(text: str, is_final: bool = False) -> str:
    """
    Choose translator based on heuristics.
    
    Returns: 'argos' or 'llm'
    """
    words = text.split()
    word_count = len(words)
    
    # Final pass always uses LLM for quality
    if is_final:
        return "llm"
    
    # Very short: always Argos (fast)
    if word_count <= 3:
        return "argos"
    
    # Short-medium: Argos for speed
    if word_count <= LIVE_CHUNK_MAX_WORDS:
        return "argos"
    
    # Named entities: use LLM
    if has_named_entity(text):
        return "llm"
    
    # Default: Argos for live (speed over quality)
    return "argos"


# =============================================================================
# CHUNK PROCESSING
# =============================================================================
def translate_chunk_sync(
    text: str,
    source_lang: str,
    target_lang: str,
    is_final: bool = False
) -> tuple:
    """
    Synchronous translation for thread pool.
    Returns: (translated_text, metadata)
    """
    from .engine_nmt import get_nmt_engine
    
    start = time.perf_counter()
    nmt = get_nmt_engine()
    
    translator = choose_translator(text, is_final)
    
    if translator == "llm":
        # Use smart_translate with LLM
        translated, meta = nmt.smart_translate(
            text, source_lang, target_lang,
            use_llm=True, use_memory=True
        )
    else:
        # Fast Argos path
        translated, meta = nmt.smart_translate(
            text, source_lang, target_lang,
            use_llm=False, use_memory=True
        )
    
    meta["latency_ms"] = int((time.perf_counter() - start) * 1000)
    meta["translator"] = translator
    
    return translated, meta


def synthesize_chunk_sync(
    text: str,
    voice_key: str = "hi_IN"
) -> tuple:
    """
    Synchronous TTS for thread pool.
    Returns: (audio_bytes, metadata)
    """
    from .engine_tts import get_tts_engine
    
    start = time.perf_counter()
    tts = get_tts_engine()
    
    audio_array = tts.synthesize(text, voice_key=voice_key)
    
    # Convert to WAV bytes
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(22050)
        audio_int16 = (audio_array * 32767).astype(np.int16)
        wav.writeframes(audio_int16.tobytes())
    
    wav_bytes = wav_buffer.getvalue()
    
    meta = {
        "voice": voice_key,
        "samples": len(audio_array),
        "latency_ms": int((time.perf_counter() - start) * 1000),
    }
    
    return wav_bytes, meta


# =============================================================================
# ASYNC PIPELINE
# =============================================================================
async def process_text_chunk(
    sio,
    sid: str,
    text: str,
    source_lang: str = "en",
    target_lang: str = "hi",
    chunk_order: int = 0,
    is_final: bool = False
):
    """
    Process a text chunk through NMT → TTS pipeline.
    Emits 'chunk_ready' event when audio is ready.
    """
    if not text or not text.strip():
        return
    
    text = text.strip()
    loop = asyncio.get_running_loop()
    
    try:
        # NMT
        translated, nmt_meta = await loop.run_in_executor(
            get_nmt_executor(),
            translate_chunk_sync,
            text, source_lang, target_lang, is_final
        )
        
        if not translated:
            return
        
        # Choose TTS voice based on target language
        voice_key = "hi_IN" if target_lang == "hi" else "en_US"
        
        # TTS
        audio_bytes, tts_meta = await loop.run_in_executor(
            get_tts_executor(),
            synthesize_chunk_sync,
            translated, voice_key
        )
        
        # Encode audio as base64
        audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
        
        # Emit result
        await sio.emit("chunk_ready", {
            "order": chunk_order,
            "original": text,
            "translated": translated,
            "audio_b64": audio_b64,
            "is_final": is_final,
            "metrics": {
                "nmt_ms": nmt_meta.get("latency_ms", 0),
                "tts_ms": tts_meta.get("latency_ms", 0),
                "source": nmt_meta.get("source", "unknown"),
                "translator": nmt_meta.get("translator", "unknown"),
            },
            "timestamp": time.time(),
        }, to=sid)
        
        logger.debug(
            f"Chunk [{chunk_order}]: '{text[:30]}' → '{translated[:30]}' "
            f"(NMT: {nmt_meta.get('latency_ms')}ms, TTS: {tts_meta.get('latency_ms')}ms)"
        )
        
    except Exception as e:
        logger.error(f"Pipeline error for chunk {chunk_order}: {e}")
        await sio.emit("error", {
            "type": "pipeline",
            "message": str(e),
            "chunk_order": chunk_order,
        }, to=sid)


# =============================================================================
# STT PARTIAL HANDLER
# =============================================================================
_chunk_counters: Dict[str, int] = {}


async def handle_stt_partial(
    sio,
    sid: str,
    partial_text: str,
    source_lang: str = "en",
    target_lang: str = "hi"
):
    """
    Handle partial STT text and send through pipeline.
    Called by stt_streamer when new text is detected.
    """
    if not partial_text or not partial_text.strip():
        return
    
    # Get chunk order for this session
    if sid not in _chunk_counters:
        _chunk_counters[sid] = 0
    
    chunk_order = _chunk_counters[sid]
    _chunk_counters[sid] += 1
    
    # Process asynchronously (don't block)
    asyncio.create_task(
        process_text_chunk(
            sio, sid, partial_text,
            source_lang, target_lang,
            chunk_order, is_final=False
        )
    )


async def handle_final_pass(
    sio,
    sid: str,
    full_text: str,
    source_lang: str = "en",
    target_lang: str = "hi"
):
    """
    Handle final LLM polish when speech ends.
    """
    if not full_text or not full_text.strip():
        return
    
    chunk_order = _chunk_counters.get(sid, 0)
    _chunk_counters[sid] = chunk_order + 1
    
    # Process with LLM for quality
    asyncio.create_task(
        process_text_chunk(
            sio, sid, full_text,
            source_lang, target_lang,
            chunk_order, is_final=True
        )
    )


def reset_session(sid: str):
    """Reset chunk counter for session."""
    _chunk_counters.pop(sid, None)


# =============================================================================
# STATS
# =============================================================================
def get_pipeline_stats() -> dict:
    return {
        "nmt_workers": NMT_WORKERS,
        "tts_workers": TTS_WORKERS,
        "live_chunk_max_words": LIVE_CHUNK_MAX_WORDS,
        "active_sessions": len(_chunk_counters),
    }
