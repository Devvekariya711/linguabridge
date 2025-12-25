"""
LinguaBridge Audio Utilities
============================
Helper functions for audio processing, conversion, and file management.
"""

import io
import wave
import struct
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def wav_bytes_to_numpy(wav_bytes: bytes) -> np.ndarray:
    """
    Convert WAV bytes to numpy float32 array.
    
    Args:
        wav_bytes: WAV file as bytes
        
    Returns:
        Audio samples as float32 numpy array (normalized to [-1, 1])
    
    Stress Tests:
        - Edge Case 1: Empty bytes -> returns empty array
        - Edge Case 2: Invalid WAV header -> raises ValueError
    """
    if not wav_bytes:
        return np.array([], dtype=np.float32)
    
    try:
        with io.BytesIO(wav_bytes) as buffer:
            with wave.open(buffer, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()
                
                raw_data = wav_file.readframes(n_frames)
                
                # Convert based on sample width
                if sample_width == 2:  # 16-bit
                    audio = np.frombuffer(raw_data, dtype=np.int16)
                    audio = audio.astype(np.float32) / 32768.0
                elif sample_width == 4:  # 32-bit
                    audio = np.frombuffer(raw_data, dtype=np.int32)
                    audio = audio.astype(np.float32) / 2147483648.0
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")
                
                return audio
                
    except wave.Error as e:
        raise ValueError(f"Invalid WAV data: {e}")


def numpy_to_wav_bytes(
    audio: np.ndarray,
    sample_rate: int = 22050,
    channels: int = 1,
) -> bytes:
    """
    Convert numpy array to WAV bytes.
    
    Args:
        audio: Audio samples as numpy array (int16 or float32)
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        
    Returns:
        WAV file as bytes
    
    Stress Tests:
        - Edge Case 1: Empty array -> returns valid empty WAV
        - Edge Case 2: Float32 array -> converts to int16
    """
    # Convert to int16 if float
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio = (audio * 32767).astype(np.int16)
    elif audio.dtype != np.int16:
        audio = audio.astype(np.int16)
    
    buffer = io.BytesIO()
    
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    
    return buffer.getvalue()


def pcm16_to_wav_bytes(
    pcm_data: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
) -> bytes:
    """
    Convert raw PCM16 bytes to WAV bytes.
    
    Args:
        pcm_data: Raw PCM16 audio bytes
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        
    Returns:
        WAV file as bytes
    """
    buffer = io.BytesIO()
    
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    
    return buffer.getvalue()


def resample_audio(
    audio: np.ndarray,
    src_rate: int,
    tgt_rate: int = 16000,
) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Uses linear interpolation for simplicity.
    For production, consider scipy.signal.resample.
    
    Args:
        audio: Audio samples
        src_rate: Source sample rate
        tgt_rate: Target sample rate (default 16000 for Whisper)
        
    Returns:
        Resampled audio array
    """
    if src_rate == tgt_rate:
        return audio
    
    # Simple linear interpolation
    ratio = tgt_rate / src_rate
    n_samples = int(len(audio) * ratio)
    
    if n_samples == 0:
        return np.array([], dtype=audio.dtype)
    
    # Use numpy interp for fast linear interpolation
    old_indices = np.arange(len(audio))
    new_indices = np.linspace(0, len(audio) - 1, n_samples)
    
    return np.interp(new_indices, old_indices, audio).astype(audio.dtype)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range.
    
    Args:
        audio: Audio samples
        
    Returns:
        Normalized audio
    """
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    
    max_val = np.abs(audio).max()
    if max_val > 0:
        return audio / max_val
    return audio


def cleanup_temp_files(
    directory: Path,
    max_age_hours: int = 24,
    extensions: tuple = (".wav", ".tmp"),
) -> int:
    """
    Remove old temporary files from a directory.
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum file age in hours
        extensions: File extensions to delete
        
    Returns:
        Number of files deleted
    
    Stress Tests:
        - Edge Case 1: Directory doesn't exist -> returns 0
        - Edge Case 2: Permission error -> logs warning, continues
    """
    if not directory.exists():
        return 0
    
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    deleted = 0
    
    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue
            
        if not file_path.suffix.lower() in extensions:
            continue
        
        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime < cutoff:
                file_path.unlink()
                deleted += 1
                logger.debug(f"Deleted old file: {file_path.name}")
        except PermissionError:
            logger.warning(f"Cannot delete (permission denied): {file_path}")
        except Exception as e:
            logger.warning(f"Error deleting {file_path}: {e}")
    
    if deleted > 0:
        logger.info(f"Cleaned up {deleted} old files from {directory}")
    
    return deleted


def generate_temp_filename(extension: str = ".wav") -> str:
    """
    Generate a unique temporary filename.
    
    Args:
        extension: File extension including dot
        
    Returns:
        Unique filename string
    """
    import uuid
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = str(uuid.uuid4())[:8]
    return f"temp_{timestamp}_{unique}{extension}"
