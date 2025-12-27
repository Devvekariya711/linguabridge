"""
LinguaBridge Streaming STT
==========================
Sliding-window partial transcription for live translation.
Emits text diffs as speech is detected.
"""

import asyncio
import logging
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
from .constants import (
    get_env_float, get_env_int, get_env,
    STT_MODEL, STT_DEVICE, STT_COMPUTE_TYPE
)

# Streaming parameters
WINDOW_SECONDS = get_env_float("STT_WINDOW_SECONDS", 1.0)
STEP_SECONDS = get_env_float("STT_STEP_SECONDS", 0.25)
SAMPLE_RATE = 16000

# =============================================================================
# GLOBAL MODEL (loaded once)
# =============================================================================
_whisper_model = None
_stt_executor = None


def _get_whisper_model():
    """Get or load Whisper model with GPU support."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    
    from faster_whisper import WhisperModel
    
    device = STT_DEVICE
    compute_type = STT_COMPUTE_TYPE
    
    # Fallback to CPU if CUDA not available
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
            compute_type = "int8"
    except ImportError:
        if device == "cuda":
            logger.warning("PyTorch not found, falling back to CPU")
            device = "cpu"
            compute_type = "int8"
    
    logger.info(f"Loading Whisper model: {STT_MODEL} on {device} ({compute_type})")
    _whisper_model = WhisperModel(STT_MODEL, device=device, compute_type=compute_type)
    logger.info("Whisper model loaded for streaming")
    
    return _whisper_model


def _get_executor():
    """Get process pool for STT inference."""
    global _stt_executor
    if _stt_executor is None:
        _stt_executor = ProcessPoolExecutor(max_workers=1)
    return _stt_executor


# =============================================================================
# SESSION BUFFER
# =============================================================================
class SessionBuffer:
    """Per-session audio buffer for sliding window STT."""
    
    def __init__(self, sid: str):
        self.sid = sid
        self.frames: deque = deque()
        self.last_emitted_text = ""
        self.last_full_text = ""
        self.frame_order = 0
        self.lock = asyncio.Lock()
        self.created_at = time.time()
        
        # Stats
        self.frames_received = 0
        self.transcriptions_run = 0
        
    def append_frame(self, pcm_bytes: bytes, order: int = None):
        """Append audio frame (PCM16 bytes)."""
        self.frames.append(pcm_bytes)
        self.frames_received += 1
        
        if order is not None:
            self.frame_order = order
        
        # Keep only WINDOW_SECONDS of audio
        max_bytes = int(SAMPLE_RATE * WINDOW_SECONDS * 2)  # 2 bytes per sample
        total_bytes = sum(len(f) for f in self.frames)
        
        while total_bytes > max_bytes and len(self.frames) > 1:
            removed = self.frames.popleft()
            total_bytes -= len(removed)
    
    def get_audio_window(self) -> Optional[np.ndarray]:
        """Get current audio window as float32 numpy array."""
        if not self.frames:
            return None
        
        # Concatenate all frames
        data = b"".join(self.frames)
        if len(data) < 1600:  # Minimum ~0.1s
            return None
        
        # Convert PCM16 to float32
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
        return arr
    
    def compute_text_diff(self, new_text: str) -> str:
        """Compute new text that wasn't emitted before."""
        if not new_text:
            return ""
        
        new_text = new_text.strip()
        
        # If new text starts with old text, return only the new part
        if new_text.startswith(self.last_emitted_text):
            new_part = new_text[len(self.last_emitted_text):].strip()
            return new_part
        
        # Word-level diff
        old_words = self.last_emitted_text.split()
        new_words = new_text.split()
        
        if len(new_words) > len(old_words):
            # Find where they start to differ
            i = 0
            while i < len(old_words) and i < len(new_words):
                if old_words[i] != new_words[i]:
                    break
                i += 1
            return " ".join(new_words[i:])
        
        return new_text
    
    def get_stats(self) -> dict:
        return {
            "frames_received": self.frames_received,
            "transcriptions_run": self.transcriptions_run,
            "buffer_size": sum(len(f) for f in self.frames),
            "age_seconds": round(time.time() - self.created_at, 1),
        }


# =============================================================================
# SESSION MANAGER
# =============================================================================
SESSIONS: Dict[str, SessionBuffer] = {}


def get_session(sid: str) -> SessionBuffer:
    """Get or create session buffer."""
    if sid not in SESSIONS:
        SESSIONS[sid] = SessionBuffer(sid)
        logger.debug(f"Created session buffer: {sid}")
    return SESSIONS[sid]


def remove_session(sid: str):
    """Remove session buffer."""
    if sid in SESSIONS:
        del SESSIONS[sid]
        logger.debug(f"Removed session buffer: {sid}")


# =============================================================================
# TRANSCRIPTION
# =============================================================================
def transcribe_audio_sync(audio: np.ndarray) -> str:
    """Synchronous transcription (runs in executor)."""
    model = _get_whisper_model()
    
    segments, info = model.transcribe(
        audio,
        beam_size=1,
        vad_filter=True,
        word_timestamps=False,  # Faster without word timestamps
    )
    
    text = " ".join([seg.text.strip() for seg in segments]).strip()
    return text


async def transcribe_window(session: SessionBuffer) -> Optional[str]:
    """Transcribe current audio window and return new text diff."""
    async with session.lock:
        audio = session.get_audio_window()
    
    if audio is None or len(audio) < 1600:
        return None
    
    try:
        loop = asyncio.get_running_loop()
        
        # Run in executor to avoid blocking
        full_text = await loop.run_in_executor(
            None,  # Use default executor
            transcribe_audio_sync,
            audio
        )
        
        session.transcriptions_run += 1
        
        if not full_text:
            return None
        
        # Compute diff
        new_text = session.compute_text_diff(full_text)
        
        if new_text:
            session.last_emitted_text = full_text
            session.last_full_text = full_text
        
        return new_text
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return None


# =============================================================================
# STREAMING LOOP
# =============================================================================
async def session_transcribe_loop(sio, sid: str):
    """
    Background task: periodically transcribe audio window.
    Emits 'stt_partial' events with new text.
    """
    logger.info(f"Started transcribe loop for {sid}")
    
    while sid in SESSIONS:
        await asyncio.sleep(STEP_SECONDS)
        
        session = SESSIONS.get(sid)
        if session is None:
            break
        
        try:
            new_text = await transcribe_window(session)
            
            if new_text:
                # Emit partial transcription
                await sio.emit("stt_partial", {
                    "text": new_text,
                    "full_text": session.last_full_text,
                    "timestamp": time.time(),
                }, to=sid)
                
                logger.debug(f"STT partial [{sid}]: {new_text[:50]}...")
                
        except Exception as e:
            logger.error(f"Transcribe loop error [{sid}]: {e}")
    
    logger.info(f"Ended transcribe loop for {sid}")


# =============================================================================
# PUBLIC API
# =============================================================================
async def handle_audio_frame(sio, sid: str, pcm_bytes: bytes, order: int = 0):
    """Handle incoming audio frame from client."""
    session = get_session(sid)
    
    async with session.lock:
        session.append_frame(pcm_bytes, order)


async def start_session(sio, sid: str):
    """Start streaming session."""
    session = get_session(sid)
    
    # Start transcription loop
    asyncio.create_task(session_transcribe_loop(sio, sid))


def end_session(sid: str):
    """End streaming session."""
    remove_session(sid)


def get_session_stats(sid: str) -> Optional[dict]:
    """Get session statistics."""
    session = SESSIONS.get(sid)
    if session:
        return session.get_stats()
    return None
