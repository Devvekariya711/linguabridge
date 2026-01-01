"""
LinguaBridge Speaker Session
=============================
Per-speaker state management for dual-channel translation.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Literal
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSession:
    """
    State for one speaker in the conversation.
    
    speaker_id: Unique identifier (e.g., "a" or "b")
    language: Speaker's native language ("en" or "hi")
    target_language: Language to translate into
    output_channel: Which ear receives this speaker's translation
    """
    speaker_id: str
    language: str
    target_language: str
    output_channel: Literal["left", "right"]
    
    # Input device
    input_device: Optional[int] = None
    
    # Audio buffer for STT
    audio_buffer: deque = field(default_factory=lambda: deque(maxlen=16000 * 5))
    
    # Transcription state
    last_text: str = ""
    accumulated_text: str = ""
    
    # Statistics
    total_phrases: int = 0
    total_latency_ms: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # Active state
    is_speaking: bool = False
    is_active: bool = True
    
    def add_audio(self, audio: np.ndarray):
        """Add audio samples to buffer."""
        for sample in audio.flatten():
            self.audio_buffer.append(sample)
        self.last_activity = time.time()
    
    def get_audio(self) -> np.ndarray:
        """Get all buffered audio and clear."""
        audio = np.array(list(self.audio_buffer), dtype=np.float32)
        self.audio_buffer.clear()
        return audio
    
    def append_text(self, text: str):
        """Append transcribed text."""
        self.accumulated_text += " " + text
        self.last_text = text
        self.total_phrases += 1
    
    def clear_text(self):
        """Clear accumulated text."""
        self.accumulated_text = ""
        self.last_text = ""
    
    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            "speaker_id": self.speaker_id,
            "language": self.language,
            "target_language": self.target_language,
            "output_channel": self.output_channel,
            "total_phrases": self.total_phrases,
            "avg_latency_ms": self.total_latency_ms / max(1, self.total_phrases),
            "uptime_seconds": time.time() - self.created_at,
            "is_active": self.is_active,
        }


class DualSpeakerManager:
    """
    Manage two speakers for stereo translation.
    
    Speaker A: Hindi speaker -> English output -> LEFT channel
    Speaker B: English speaker -> Hindi output -> RIGHT channel
    """
    
    def __init__(self):
        self.speaker_a: Optional[SpeakerSession] = None
        self.speaker_b: Optional[SpeakerSession] = None
        self._sessions: Dict[str, SpeakerSession] = {}
    
    def setup_speakers(
        self,
        device_a: Optional[int] = None,
        device_b: Optional[int] = None
    ):
        """
        Initialize both speakers with devices.
        
        Args:
            device_a: Mic device ID for Hindi speaker
            device_b: Mic device ID for English speaker
        """
        # Speaker A: Hindi -> English -> LEFT
        self.speaker_a = SpeakerSession(
            speaker_id="a",
            language="hi",
            target_language="en",
            output_channel="left",
            input_device=device_a
        )
        
        # Speaker B: English -> Hindi -> RIGHT
        self.speaker_b = SpeakerSession(
            speaker_id="b",
            language="en",
            target_language="hi",
            output_channel="right",
            input_device=device_b
        )
        
        self._sessions["a"] = self.speaker_a
        self._sessions["b"] = self.speaker_b
        
        logger.info("Dual speakers configured: A(HI->EN->L), B(EN->HI->R)")
    
    def get_speaker(self, speaker_id: str) -> Optional[SpeakerSession]:
        """Get speaker by ID."""
        return self._sessions.get(speaker_id)
    
    def get_output_channel(self, speaker_id: str) -> str:
        """Get output channel for a speaker's translation."""
        session = self.get_speaker(speaker_id)
        if session:
            return session.output_channel
        return "left"  # Default
    
    def get_translation_pair(self, speaker_id: str) -> tuple:
        """Get source->target language pair for speaker."""
        session = self.get_speaker(speaker_id)
        if session:
            return session.language, session.target_language
        return "en", "hi"
    
    def process_audio(self, speaker_id: str, audio: np.ndarray):
        """Add audio to speaker's buffer."""
        session = self.get_speaker(speaker_id)
        if session:
            session.add_audio(audio)
    
    def get_stats(self) -> dict:
        """Get stats for both speakers."""
        return {
            "speaker_a": self.speaker_a.get_stats() if self.speaker_a else None,
            "speaker_b": self.speaker_b.get_stats() if self.speaker_b else None,
        }
    
    def stop(self):
        """Stop both speakers."""
        if self.speaker_a:
            self.speaker_a.is_active = False
        if self.speaker_b:
            self.speaker_b.is_active = False
    
    def reset(self):
        """Reset all state."""
        self.speaker_a = None
        self.speaker_b = None
        self._sessions.clear()


# Singleton manager
_manager: Optional[DualSpeakerManager] = None


def get_speaker_manager() -> DualSpeakerManager:
    """Get the singleton speaker manager."""
    global _manager
    if _manager is None:
        _manager = DualSpeakerManager()
    return _manager
