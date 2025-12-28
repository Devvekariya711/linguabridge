"""
LinguaBridge Translation Policy
================================
Defines LIVE vs FINAL modes and enforces behavior.
"""

from dataclasses import dataclass
from typing import Literal, Optional
import time

# =============================================================================
# TRANSLATION POLICY
# =============================================================================

@dataclass(frozen=True)
class TranslationPolicy:
    """
    Immutable policy that controls translation behavior.
    
    LIVE mode: Fast, speculative, no LLM
    FINAL mode: High-quality, LLM allowed
    """
    mode: Literal["live", "final"]
    max_words: int = 3
    allow_llm: bool = False
    allow_embeddings: bool = False
    allow_vector_search: bool = False
    speculative_tts: bool = True
    final_pass_enabled: bool = True
    
    # Timeouts
    translate_timeout: float = 0.5  # seconds for live
    llm_timeout: float = 5.0  # seconds for final
    
    # Confidence thresholds
    stt_confidence_threshold: float = 0.60
    vector_similarity_threshold: float = 0.85
    
    def __post_init__(self):
        if self.mode == "live" and self.allow_llm:
            raise ValueError("LIVE mode cannot allow LLM - use FINAL mode")


# =============================================================================
# PREDEFINED POLICIES
# =============================================================================

LIVE_POLICY = TranslationPolicy(
    mode="live",
    max_words=3,
    allow_llm=False,
    allow_embeddings=False,
    allow_vector_search=False,  # Disabled for speed
    speculative_tts=True,
    final_pass_enabled=True,
    translate_timeout=0.5,
)

FINAL_POLICY = TranslationPolicy(
    mode="final",
    max_words=50,  # Full sentences
    allow_llm=True,
    allow_embeddings=True,
    allow_vector_search=True,
    speculative_tts=False,
    final_pass_enabled=False,  # This IS the final pass
    translate_timeout=5.0,
    llm_timeout=20.0,
)

FAST_POLICY = TranslationPolicy(
    mode="live",
    max_words=4,
    allow_llm=False,
    allow_embeddings=False,
    allow_vector_search=False,
    speculative_tts=True,
    final_pass_enabled=False,  # No correction
    translate_timeout=0.3,
)


# =============================================================================
# SESSION STATE
# =============================================================================

@dataclass
class SessionState:
    """Per-session state tracking."""
    sid: str
    source_lang: str = "en"
    target_lang: str = "hi"
    speaking: bool = False
    last_audio_time: float = 0.0
    silence_start: float = 0.0
    policy: TranslationPolicy = LIVE_POLICY
    
    # Buffers
    accumulated_text: str = ""
    chunks_sent: int = 0
    
    # Stats
    total_frames: int = 0
    total_chunks: int = 0
    cache_hits: int = 0
    argos_calls: int = 0
    llm_calls: int = 0
    
    def update_audio_time(self):
        """Called when audio is received."""
        self.last_audio_time = time.time()
        if not self.speaking:
            self.speaking = True
            self.silence_start = 0.0
    
    def check_silence(self, threshold_ms: float = 700) -> bool:
        """Check if silence threshold exceeded."""
        if self.last_audio_time == 0:
            return False
        
        elapsed = (time.time() - self.last_audio_time) * 1000
        return elapsed >= threshold_ms
    
    def switch_to_final(self):
        """Switch to FINAL mode for polish pass."""
        self.speaking = False
        self.policy = FINAL_POLICY
    
    def switch_to_live(self):
        """Switch back to LIVE mode."""
        self.speaking = True
        self.policy = LIVE_POLICY
        self.accumulated_text = ""
    
    def append_text(self, text: str):
        """Accumulate transcribed text."""
        if self.accumulated_text:
            self.accumulated_text += " " + text
        else:
            self.accumulated_text = text


# =============================================================================
# SESSION MANAGER
# =============================================================================

_sessions: dict[str, SessionState] = {}

SILENCE_MS_FINAL = 700  # Silence threshold to trigger final pass


def get_session(sid: str) -> SessionState:
    """Get or create session state."""
    if sid not in _sessions:
        _sessions[sid] = SessionState(sid=sid)
    return _sessions[sid]


def remove_session(sid: str):
    """Remove session state."""
    _sessions.pop(sid, None)


def get_policy(sid: str) -> TranslationPolicy:
    """Get current policy for session."""
    return get_session(sid).policy


def enforce_policy(sid: str, operation: str) -> TranslationPolicy:
    """
    Get policy and validate operation is allowed.
    Raises ValueError if operation violates policy.
    """
    policy = get_policy(sid)
    
    if operation == "llm" and not policy.allow_llm:
        raise ValueError(f"LLM not allowed in {policy.mode} mode")
    
    if operation == "vector_search" and not policy.allow_vector_search:
        raise ValueError(f"Vector search not allowed in {policy.mode} mode")
    
    if operation == "embeddings" and not policy.allow_embeddings:
        raise ValueError(f"Embeddings not allowed in {policy.mode} mode")
    
    return policy
