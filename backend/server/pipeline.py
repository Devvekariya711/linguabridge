"""
LinguaBridge Streaming Pipeline v2
===================================
Parallel NMT + TTS with race-commit pattern, policy enforcement,
bounded queues, and circuit breakers.
"""

import asyncio
import logging
import time
import base64
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Any, Tuple
import io
import wave
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS
# =============================================================================
from .constants import get_env_int, get_env_bool
from .policy import (
    TranslationPolicy, SessionState, LIVE_POLICY, FINAL_POLICY,
    get_session, get_policy, enforce_policy
)
from .bounded_queue import BoundedQueue, get_queue, BackpressureSignal
from .circuit_breaker import (
    CircuitBreaker, CircuitOpenError, call_llm_safe, safe_call,
    is_llm_available, LLM_TIMEOUT_LIVE, LLM_TIMEOUT_FINAL
)
from .chunker import SemanticChunker, Chunk, get_chunker, remove_chunker

# =============================================================================
# CONFIGURATION
# =============================================================================
NMT_WORKERS = get_env_int("NMT_WORKERS", 2)
TTS_WORKERS = get_env_int("TTS_WORKERS", 2)
MAX_NMT_QUEUE = get_env_int("MAX_NMT_QUEUE", 8)
MAX_TTS_QUEUE = get_env_int("MAX_TTS_QUEUE", 8)

# Confidence thresholds
STT_CONF_THRESHOLD = 0.60
VECTOR_SIM_THRESHOLD = 0.85
VECTOR_SIM_SUGGESTION = 0.70

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
# TRANSLATION STRATEGIES
# =============================================================================

async def lookup_exact(text: str, src: str, tgt: str) -> Optional[Tuple[str, dict]]:
    """Fastest: exact match from SQLite."""
    try:
        from . import translation_memory
        result = translation_memory.find_exact(text, src, tgt)
        if result:
            return result, {"source": "exact", "priority": 1}
    except Exception as e:
        logger.debug(f"Exact lookup failed: {e}")
    return None


async def lookup_vector(
    text: str, src: str, tgt: str, threshold: float = VECTOR_SIM_THRESHOLD
) -> Optional[Tuple[str, dict]]:
    """Fast: vector similarity search."""
    try:
        from . import translation_memory
        result = translation_memory.find_similar(text, src, tgt, threshold)
        if result:
            return result, {"source": "vector", "priority": 2, "similarity": threshold}
    except Exception as e:
        logger.debug(f"Vector lookup failed: {e}")
    return None


async def translate_argos(text: str, src: str, tgt: str) -> Tuple[str, dict]:
    """Medium: Argos neural translation."""
    loop = asyncio.get_running_loop()
    
    def _translate():
        from .engine_nmt import get_nmt_engine
        nmt = get_nmt_engine()
        return nmt.translate(text, src, tgt)
    
    start = time.perf_counter()
    result = await loop.run_in_executor(get_nmt_executor(), _translate)
    latency = int((time.perf_counter() - start) * 1000)
    
    return result, {"source": "argos", "priority": 3, "latency_ms": latency}


async def translate_llm(text: str, src: str, tgt: str, timeout: float = LLM_TIMEOUT_LIVE) -> Tuple[str, dict]:
    """Slow but high quality: LLM translation."""
    from . import engine_llm
    
    async def _call():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            engine_llm.translate,
            text, src, tgt, None
        )
    
    start = time.perf_counter()
    
    try:
        result, meta = await call_llm_safe(_call(), timeout=timeout)
        meta["priority"] = 4
        meta["latency_ms"] = int((time.perf_counter() - start) * 1000)
        return result, meta
    except CircuitOpenError:
        raise
    except Exception as e:
        logger.warning(f"LLM translation failed: {e}")
        raise


# =============================================================================
# RACE-COMMIT PATTERN
# =============================================================================

async def race_translate(
    text: str,
    src: str,
    tgt: str,
    policy: TranslationPolicy
) -> Tuple[str, dict]:
    """
    Race multiple translation strategies and commit the best valid result.
    
    Priority (lower = higher):
    1. Exact match
    2. Vector similarity (if allowed)
    3. Argos
    4. LLM (if allowed)
    
    Returns first valid result by priority, cancels others.
    """
    tasks: List[asyncio.Task] = []
    results: Dict[int, Tuple[str, dict]] = {}
    
    # Create tasks based on policy
    # Priority 1: Exact match (always)
    tasks.append(asyncio.create_task(
        lookup_exact(text, src, tgt),
        name="exact"
    ))
    
    # Priority 2: Vector search (if allowed)
    if policy.allow_vector_search:
        tasks.append(asyncio.create_task(
            lookup_vector(text, src, tgt),
            name="vector"
        ))
    
    # Priority 3: Argos (always for live)
    tasks.append(asyncio.create_task(
        safe_call(
            translate_argos(text, src, tgt),
            timeout=policy.translate_timeout,
            fallback=None
        ),
        name="argos"
    ))
    
    # Priority 4: LLM (only if allowed)
    if policy.allow_llm and is_llm_available():
        tasks.append(asyncio.create_task(
            safe_call(
                translate_llm(text, src, tgt, timeout=policy.llm_timeout),
                timeout=policy.llm_timeout + 1,
                fallback=None
            ),
            name="llm"
        ))
    
    # Wait for first valid result
    winner = None
    winner_meta = {"source": "unknown"}
    
    try:
        # Wait with timeout
        done, pending = await asyncio.wait(
            tasks,
            timeout=policy.translate_timeout + 0.5,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Check results by priority
        for task in done:
            try:
                result = task.result()
                if result and result[0]:
                    translation, meta = result
                    priority = meta.get("priority", 99)
                    
                    # First valid wins if no current winner
                    if winner is None or priority < winner_meta.get("priority", 99):
                        winner = translation
                        winner_meta = meta
            except Exception:
                continue
        
        # If still no winner, wait a bit more for others
        if winner is None and pending:
            done2, pending2 = await asyncio.wait(
                pending,
                timeout=0.3,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done2:
                try:
                    result = task.result()
                    if result and result[0]:
                        winner, winner_meta = result
                        break
                except Exception:
                    continue
            
            # Cancel remaining
            for task in pending2:
                task.cancel()
        
        # Cancel any still pending
        for task in pending:
            task.cancel()
            
    except asyncio.TimeoutError:
        logger.warning("Race translate timed out")
        for task in tasks:
            task.cancel()
    
    if winner is None:
        # Ultimate fallback
        winner = text  # Return original
        winner_meta = {"source": "fallback", "priority": 99}
    
    return winner, winner_meta


# =============================================================================
# TTS SYNTHESIS
# =============================================================================

async def synthesize_async(text: str, voice_key: str = "hi_IN") -> Tuple[bytes, dict]:
    """Async TTS synthesis."""
    loop = asyncio.get_running_loop()
    
    def _synth():
        from .engine_tts import get_tts_engine
        start = time.perf_counter()
        tts = get_tts_engine()
        audio_array = tts.synthesize(text, voice_key=voice_key)
        
        # Convert to WAV bytes
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            audio_int16 = (audio_array * 32767).astype(np.int16)
            wav.writeframes(audio_int16.tobytes())
        
        return wav_buffer.getvalue(), {
            "voice": voice_key,
            "samples": len(audio_array),
            "latency_ms": int((time.perf_counter() - start) * 1000)
        }
    
    return await loop.run_in_executor(get_tts_executor(), _synth)


# =============================================================================
# CHUNK PROCESSING
# =============================================================================

async def process_chunk(
    sio,
    sid: str,
    chunk: Chunk,
    source_lang: str,
    target_lang: str,
    chunk_order: int,
    policy: TranslationPolicy
) -> Optional[dict]:
    """
    Process a semantic chunk through NMT → TTS pipeline.
    Uses race-commit pattern based on policy.
    """
    text = chunk.text.strip()
    if not text:
        return None
    
    start_time = time.perf_counter()
    
    try:
        # Translation with race pattern
        translated, nmt_meta = await race_translate(text, source_lang, target_lang, policy)
        
        if not translated:
            return None
        
        # TTS
        voice_key = "hi_IN" if target_lang == "hi" else "en_US"
        audio_bytes, tts_meta = await synthesize_async(translated, voice_key)
        
        # Encode audio
        audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
        
        # Calculate total latency
        total_ms = int((time.perf_counter() - start_time) * 1000)
        
        # Build result
        result = {
            "order": chunk_order,
            "original": text,
            "translated": translated,
            "audio_b64": audio_b64,
            "is_final": policy.mode == "final",
            "is_speculative": policy.speculative_tts,
            "has_named_entity": chunk.has_named_entity,
            "metrics": {
                "nmt_ms": nmt_meta.get("latency_ms", 0),
                "tts_ms": tts_meta.get("latency_ms", 0),
                "total_ms": total_ms,
                "source": nmt_meta.get("source", "unknown"),
            },
            "timestamp": time.time(),
        }
        
        # Emit result
        await sio.emit("chunk_ready", result, to=sid)
        
        logger.debug(
            f"[{sid}] Chunk {chunk_order}: '{text[:20]}' → '{translated[:20]}' "
            f"({nmt_meta.get('source')}, {total_ms}ms)"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Chunk processing failed: {e}")
        await sio.emit("error", {
            "type": "pipeline",
            "message": str(e),
            "chunk_order": chunk_order,
        }, to=sid)
        return None


# =============================================================================
# SESSION HANDLERS
# =============================================================================

_chunk_counters: Dict[str, int] = {}
_active_tasks: Dict[str, List[asyncio.Task]] = {}


async def handle_stt_partial(
    sio,
    sid: str,
    partial_text: str,
    confidence: float = 1.0
):
    """
    Handle partial STT text using semantic chunking.
    Enforces LIVE policy.
    """
    session = get_session(sid)
    policy = enforce_policy(sid, "translation")
    
    # Skip low-confidence transcriptions
    if confidence < STT_CONF_THRESHOLD:
        logger.debug(f"Skipping low confidence ({confidence:.2f}): {partial_text}")
        return
    
    # Get chunker and process
    chunker = get_chunker(sid)
    chunks = chunker.chunk(partial_text, confidence=confidence)
    
    # Process each chunk
    for chunk in chunks:
        if sid not in _chunk_counters:
            _chunk_counters[sid] = 0
        
        chunk_order = _chunk_counters[sid]
        _chunk_counters[sid] += 1
        
        # Update session
        session.append_text(chunk.text)
        session.total_chunks += 1
        
        # Process asynchronously
        task = asyncio.create_task(
            process_chunk(
                sio, sid, chunk,
                session.source_lang, session.target_lang,
                chunk_order, policy
            )
        )
        
        # Track task for cancellation
        if sid not in _active_tasks:
            _active_tasks[sid] = []
        _active_tasks[sid].append(task)
        
        # Check backpressure
        nmt_queue = get_queue("nmt", MAX_NMT_QUEUE)
        if nmt_queue.is_overloaded():
            signal = BackpressureSignal.from_queue(nmt_queue)
            await sio.emit("backpressure", {
                "recommendation": signal.recommendation,
                "queue_size": signal.queue_size,
            }, to=sid)


async def handle_final_pass(
    sio,
    sid: str,
    full_text: str = None
):
    """
    Handle final LLM polish when speech ends.
    Uses FINAL policy for high quality.
    """
    session = get_session(sid)
    
    # Use accumulated text if not provided
    if not full_text:
        full_text = session.accumulated_text
    
    if not full_text or not full_text.strip():
        return
    
    # Switch to FINAL mode
    session.switch_to_final()
    policy = FINAL_POLICY
    
    # Flush chunker
    chunker = get_chunker(sid)
    final_chunk = chunker.flush()
    
    # Create chunk for full text
    from .chunker import Chunk
    chunk = Chunk(
        text=full_text.strip(),
        confidence=1.0,
        is_complete=True
    )
    
    chunk_order = _chunk_counters.get(sid, 0)
    _chunk_counters[sid] = chunk_order + 1
    
    # Process with FINAL policy (LLM allowed)
    await process_chunk(
        sio, sid, chunk,
        session.source_lang, session.target_lang,
        chunk_order, policy
    )
    
    # Update stats
    session.llm_calls += 1


def reset_session(sid: str):
    """Reset session state and cancel pending tasks."""
    # Cancel active tasks
    if sid in _active_tasks:
        for task in _active_tasks[sid]:
            task.cancel()
        del _active_tasks[sid]
    
    _chunk_counters.pop(sid, None)
    remove_chunker(sid)


# =============================================================================
# STATS
# =============================================================================

def get_pipeline_stats() -> dict:
    """Get pipeline statistics."""
    from .circuit_breaker import get_llm_stats
    from .bounded_queue import get_all_queue_stats
    
    return {
        "nmt_workers": NMT_WORKERS,
        "tts_workers": TTS_WORKERS,
        "active_sessions": len(_chunk_counters),
        "queues": get_all_queue_stats(),
        "llm_circuit": get_llm_stats(),
    }
