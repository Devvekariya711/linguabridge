"""
LinguaBridge Stereo Mixer
==========================
Merge two mono audio buffers into stereo output.
Left channel = English (for Speaker B)
Right channel = Hindi (for Speaker A)
"""

import logging
from collections import deque
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class StereoMixer:
    """
    Merge two mono audio streams into a single stereo output.
    
    Left channel (0): English TTS (translated Hindi)
    Right channel (1): Hindi TTS (translated English)
    """
    
    def __init__(self, sample_rate: int = 22050, buffer_seconds: float = 5.0):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * buffer_seconds)
        
        # Separate buffers for each channel
        self.left_buffer = deque(maxlen=self.max_samples)
        self.right_buffer = deque(maxlen=self.max_samples)
        
        # Track what has been played
        self.left_position = 0
        self.right_position = 0
    
    def add_left(self, audio: np.ndarray):
        """Add audio to left channel (English for Speaker B)."""
        audio = self._normalize(audio)
        for sample in audio:
            self.left_buffer.append(sample)
        logger.debug(f"Left buffer: {len(self.left_buffer)} samples")
    
    def add_right(self, audio: np.ndarray):
        """Add audio to right channel (Hindi for Speaker A)."""
        audio = self._normalize(audio)
        for sample in audio:
            self.right_buffer.append(sample)
        logger.debug(f"Right buffer: {len(self.right_buffer)} samples")
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio is 1D float32."""
        if audio.ndim > 1:
            audio = audio.flatten()
        return audio.astype(np.float32)
    
    def _drain(self, buffer: deque, samples: int) -> np.ndarray:
        """Drain samples from buffer, pad with silence if needed."""
        result = []
        for _ in range(samples):
            if buffer:
                result.append(buffer.popleft())
            else:
                result.append(0.0)  # Silence
        return np.array(result, dtype=np.float32)
    
    def get_stereo_chunk(self, samples: int) -> np.ndarray:
        """
        Get stereo audio chunk.
        
        Returns:
            numpy array of shape (samples, 2) with L/R channels
        """
        left = self._drain(self.left_buffer, samples)
        right = self._drain(self.right_buffer, samples)
        
        stereo = np.zeros((samples, 2), dtype=np.float32)
        stereo[:, 0] = left   # Left channel
        stereo[:, 1] = right  # Right channel
        
        return stereo
    
    def get_available(self) -> Tuple[int, int]:
        """Get available samples in each buffer."""
        return len(self.left_buffer), len(self.right_buffer)
    
    def has_audio(self) -> bool:
        """Check if any audio is available."""
        return len(self.left_buffer) > 0 or len(self.right_buffer) > 0
    
    def clear(self):
        """Clear both buffers."""
        self.left_buffer.clear()
        self.right_buffer.clear()


def merge_mono_to_stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Merge two mono audio arrays into stereo.
    
    Args:
        left: Mono audio for left channel
        right: Mono audio for right channel
    
    Returns:
        Stereo audio array of shape (max_len, 2)
    """
    # Normalize to 1D
    if left.ndim > 1:
        left = left.flatten()
    if right.ndim > 1:
        right = right.flatten()
    
    # Match lengths (pad shorter with silence)
    max_len = max(len(left), len(right))
    
    stereo = np.zeros((max_len, 2), dtype=np.float32)
    stereo[:len(left), 0] = left.astype(np.float32)
    stereo[:len(right), 1] = right.astype(np.float32)
    
    return stereo


def split_stereo_to_mono(stereo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split stereo audio into two mono arrays.
    
    Args:
        stereo: Stereo audio of shape (samples, 2)
    
    Returns:
        Tuple of (left, right) mono arrays
    """
    if stereo.ndim == 1:
        # Already mono
        return stereo, stereo
    
    return stereo[:, 0], stereo[:, 1]


def create_silence(samples: int, channels: int = 2) -> np.ndarray:
    """Create silence buffer."""
    if channels == 1:
        return np.zeros(samples, dtype=np.float32)
    return np.zeros((samples, channels), dtype=np.float32)
