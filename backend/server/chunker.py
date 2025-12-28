"""
LinguaBridge Semantic Chunker
==============================
Splits text into meaning-bearing micro-units (2-4 words).
Not word-by-word, but linguistically coherent chunks.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time

logger = logging.getLogger(__name__)


# =============================================================================
# CHUNK DATA
# =============================================================================

@dataclass
class Chunk:
    """A semantic chunk with metadata."""
    text: str
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 1.0
    is_complete: bool = True
    has_named_entity: bool = False
    word_count: int = 0
    
    def __post_init__(self):
        self.word_count = len(self.text.split())


# =============================================================================
# LINGUISTIC PATTERNS
# =============================================================================

# Auxiliaries that should wait for next word
AUXILIARIES = {
    'am', 'is', 'are', 'was', 'were', 'be', 'being', 'been',
    'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must',
    'a', 'an', 'the', 'to', 'of', 'in', 'on', 'at', 'for', 'with', 'by'
}

# Conjunctions that trigger chunk boundary
CONJUNCTIONS = {
    'and', 'but', 'or', 'nor', 'yet', 'so', 'because', 'although',
    'while', 'when', 'if', 'then', 'that', 'which', 'who'
}

# Punctuation that triggers chunk boundary
BOUNDARY_PUNCT = {'.', ',', '!', '?', ';', ':', '-', '—'}

# Common phrases to keep together
COMMON_PHRASES = {
    'i am', 'you are', 'he is', 'she is', 'it is', 'we are', 'they are',
    'my name', 'your name', 'good morning', 'good evening', 'thank you',
    'how are', 'what is', 'where is', 'can you', 'could you', 'would you',
    'i have', 'i will', 'i would', 'i can', 'i could',
    'on behalf of', 'in order to', 'as well as', 'in front of'
}


# =============================================================================
# NAMED ENTITY DETECTION (Simple)
# =============================================================================

def detect_named_entity(text: str) -> bool:
    """
    Simple named entity detection using heuristics.
    
    Detects:
    - Capitalized words (not at sentence start)
    - Words with mixed case (iPhone, McDonald's)
    - All-caps words (USA, NASA)
    """
    words = text.split()
    
    for i, word in enumerate(words):
        # Skip first word (sentence start)
        if i == 0:
            continue
        
        # Remove punctuation for check
        clean = word.strip('.,!?;:\'"')
        
        if not clean:
            continue
        
        # Capitalized word (not all caps, not all lower)
        if clean[0].isupper() and len(clean) > 1:
            if clean[1:].islower():  # "John", "Paris"
                return True
        
        # All caps (likely acronym)
        if clean.isupper() and len(clean) > 1:
            return True
        
        # Mixed case (iPhone, McKinsey)
        if any(c.isupper() for c in clean[1:]) and any(c.islower() for c in clean):
            return True
    
    return False


# =============================================================================
# SEMANTIC CHUNKER
# =============================================================================

class SemanticChunker:
    """
    Chunks text into meaning-bearing units.
    
    Rules:
    - Minimum chunk: 2 content words
    - Maximum chunk: 4 words
    - Don't emit chunk that is only auxiliary/particle
    - Emit on: pause, conjunction, punctuation, or content words >= min
    """
    
    def __init__(
        self,
        min_words: int = 2,
        max_words: int = 4,
        min_content_words: int = 1
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.min_content_words = min_content_words
        
        # Buffer for held words
        self.buffer: List[str] = []
        self.buffer_start_time: float = 0.0
    
    def is_content_word(self, word: str) -> bool:
        """Check if word is a content word (noun/verb/adj/adv)."""
        clean = word.lower().strip('.,!?;:\'"')
        return clean not in AUXILIARIES and len(clean) > 1
    
    def count_content_words(self, words: List[str]) -> int:
        """Count content words in list."""
        return sum(1 for w in words if self.is_content_word(w))
    
    def should_emit(self, words: List[str], force: bool = False) -> bool:
        """Decide if current buffer should be emitted as chunk."""
        if not words:
            return False
        
        # Force emit (end of stream)
        if force and len(words) >= 1:
            return True
        
        # Max words reached
        if len(words) >= self.max_words:
            return True
        
        # Check last word for boundary markers
        last = words[-1].lower().strip()
        
        # Punctuation boundary
        if any(p in words[-1] for p in BOUNDARY_PUNCT):
            return len(words) >= self.min_words
        
        # Conjunction boundary
        if last in CONJUNCTIONS and len(words) > 1:
            return True
        
        # Enough content words
        content_count = self.count_content_words(words)
        if content_count >= self.min_content_words and len(words) >= self.min_words:
            # Don't emit if last word is auxiliary (wait for more)
            if last not in AUXILIARIES:
                return True
        
        return False
    
    def chunk(
        self,
        partial_text: str,
        confidence: float = 1.0,
        timestamp: float = None,
        force_emit: bool = False
    ) -> List[Chunk]:
        """
        Process partial text and return complete chunks.
        
        Args:
            partial_text: New text from STT
            confidence: STT confidence score
            timestamp: Current timestamp
            force_emit: Force emit all buffered content
        
        Returns:
            List of complete chunks (may be empty)
        """
        if timestamp is None:
            timestamp = time.time()
        
        if not self.buffer:
            self.buffer_start_time = timestamp
        
        # Add new words to buffer
        new_words = partial_text.split()
        self.buffer.extend(new_words)
        
        chunks: List[Chunk] = []
        
        # Process buffer
        while self.buffer:
            # Check for common phrases first
            phrase_found = self._check_common_phrase()
            if phrase_found:
                chunks.append(phrase_found)
                continue
            
            # Decide how many words to emit
            emit_count = self._find_emit_point(force_emit)
            
            if emit_count > 0:
                chunk_words = self.buffer[:emit_count]
                self.buffer = self.buffer[emit_count:]
                
                chunk_text = ' '.join(chunk_words)
                
                chunks.append(Chunk(
                    text=chunk_text,
                    start_time=self.buffer_start_time,
                    end_time=timestamp,
                    confidence=confidence,
                    is_complete=True,
                    has_named_entity=detect_named_entity(chunk_text),
                ))
                
                self.buffer_start_time = timestamp
            else:
                break
        
        return chunks
    
    def _check_common_phrase(self) -> Optional[Chunk]:
        """Check if buffer starts with a common phrase."""
        if len(self.buffer) < 2:
            return None
        
        # Check 2-4 word phrases
        for length in range(min(4, len(self.buffer)), 1, -1):
            phrase = ' '.join(self.buffer[:length]).lower()
            if phrase in COMMON_PHRASES:
                chunk_words = self.buffer[:length]
                self.buffer = self.buffer[length:]
                
                return Chunk(
                    text=' '.join(chunk_words),
                    start_time=self.buffer_start_time,
                    end_time=time.time(),
                    confidence=1.0,
                    is_complete=True,
                )
        
        return None
    
    def _find_emit_point(self, force: bool = False) -> int:
        """Find how many words to emit from buffer."""
        if not self.buffer:
            return 0
        
        # Force emit all
        if force:
            return len(self.buffer)
        
        # Not enough words yet
        if len(self.buffer) < self.min_words:
            # Unless last word ends with punctuation
            if any(p in self.buffer[-1] for p in BOUNDARY_PUNCT):
                return len(self.buffer)
            return 0
        
        # Find best emit point (boundaries)
        for i in range(min(self.max_words, len(self.buffer)), self.min_words - 1, -1):
            words_to_check = self.buffer[:i]
            
            if self.should_emit(words_to_check):
                return i
        
        # Max words reached, force emit
        if len(self.buffer) >= self.max_words:
            return self.max_words
        
        return 0
    
    def flush(self) -> Optional[Chunk]:
        """Flush remaining buffer as final chunk."""
        if not self.buffer:
            return None
        
        chunk_text = ' '.join(self.buffer)
        chunk = Chunk(
            text=chunk_text,
            start_time=self.buffer_start_time,
            end_time=time.time(),
            confidence=1.0,
            is_complete=True,
            has_named_entity=detect_named_entity(chunk_text),
        )
        
        self.buffer = []
        return chunk
    
    def reset(self):
        """Reset chunker state."""
        self.buffer = []
        self.buffer_start_time = 0.0


# =============================================================================
# SESSION CHUNKERS
# =============================================================================

_chunkers: dict[str, SemanticChunker] = {}


def get_chunker(sid: str) -> SemanticChunker:
    """Get or create chunker for session."""
    if sid not in _chunkers:
        _chunkers[sid] = SemanticChunker()
    return _chunkers[sid]


def remove_chunker(sid: str):
    """Remove session chunker."""
    _chunkers.pop(sid, None)
