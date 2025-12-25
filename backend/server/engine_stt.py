"""
LinguaBridge STT Engine
=======================
Speech-to-Text using Faster-Whisper (CPU-optimized).

Model: whisper-base (~150 MB)
Stored at: backend/storage/voice_models/whisper/base/
"""

import logging
from pathlib import Path
from typing import Optional, Generator, Tuple
import numpy as np

from .constants import (
    WHISPER_MODEL_NAME,
    WHISPER_MODELS_DIR,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_BEAM_SIZE,
    WHISPER_VAD_FILTER,
    AUDIO_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)


class STTEngine:
    """
    Speech-to-Text engine using Faster-Whisper.
    
    Converts microphone audio → raw text.
    CPU-first design with int8 quantization for performance.
    """
    
    def __init__(self):
        self._model = None
        self._model_loaded = False
        
    def _ensure_model_loaded(self) -> None:
        """Lazy-load the Whisper model on first use."""
        if self._model_loaded:
            return
            
        try:
            from faster_whisper import WhisperModel
            
            logger.info(f"Loading Whisper model: {WHISPER_MODEL_NAME}")
            logger.info(f"Model directory: {WHISPER_MODELS_DIR}")
            logger.info(f"Device: {WHISPER_DEVICE}, Compute type: {WHISPER_COMPUTE_TYPE}")
            
            # Ensure model directory exists
            WHISPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Load model - faster-whisper auto-downloads if not present
            self._model = WhisperModel(
                WHISPER_MODEL_NAME,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                download_root=str(WHISPER_MODELS_DIR),
            )
            
            self._model_loaded = True
            logger.info("✅ Whisper model loaded successfully")
            
        except ImportError as e:
            logger.error(f"❌ faster-whisper not installed: {e}")
            raise RuntimeError("faster-whisper package not installed. Run: pip install faster-whisper")
        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            raise
    
    def transcribe(
        self,
        audio_data: np.ndarray,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio_data: Audio as numpy array (float32, mono, 16kHz)
            language: Optional language code (e.g., 'en', 'ja'). 
                      If None, auto-detects language.
        
        Returns:
            Transcribed text as string.
        
        Stress Tests:
            - Edge Case 1: Empty audio → returns empty string
            - Edge Case 2: Very short audio (<0.5s) → may return empty or partial
        """
        self._ensure_model_loaded()
        
        # Handle empty audio
        if audio_data is None or len(audio_data) == 0:
            logger.warning("Empty audio data received")
            return ""
        
        # Ensure correct dtype
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed (Whisper expects -1.0 to 1.0 range)
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        try:
            segments, info = self._model.transcribe(
                audio_data,
                language=language,
                beam_size=WHISPER_BEAM_SIZE,
                vad_filter=WHISPER_VAD_FILTER,
            )
            
            # Combine all segments
            text = " ".join(segment.text.strip() for segment in segments)
            
            if info.language:
                logger.debug(f"Detected language: {info.language} (prob: {info.language_probability:.2f})")
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def transcribe_streaming(
        self,
        audio_data: np.ndarray,
        language: Optional[str] = None,
    ) -> Generator[Tuple[str, bool], None, None]:
        """
        Stream transcription results as they become available.
        
        Yields:
            Tuple of (text_segment, is_final)
        
        Stress Tests:
            - Edge Case 1: Network interruption during stream → handle gracefully
            - Edge Case 2: Audio ends mid-word → yield partial with is_final=True
        """
        self._ensure_model_loaded()
        
        if audio_data is None or len(audio_data) == 0:
            return
        
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        try:
            segments, _ = self._model.transcribe(
                audio_data,
                language=language,
                beam_size=WHISPER_BEAM_SIZE,
                vad_filter=WHISPER_VAD_FILTER,
            )
            
            segments_list = list(segments)
            for i, segment in enumerate(segments_list):
                is_final = (i == len(segments_list) - 1)
                yield (segment.text.strip(), is_final)
                
        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            raise
    
    def transcribe_file(self, audio_path: Path, language: Optional[str] = None) -> str:
        """
        Transcribe audio from a file path.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            language: Optional language code
        
        Returns:
            Transcribed text
        """
        self._ensure_model_loaded()
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            segments, info = self._model.transcribe(
                str(audio_path),
                language=language,
                beam_size=WHISPER_BEAM_SIZE,
                vad_filter=WHISPER_VAD_FILTER,
            )
            
            text = " ".join(segment.text.strip() for segment in segments)
            return text.strip()
            
        except Exception as e:
            logger.error(f"File transcription failed: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Return information about the loaded model."""
        return {
            "model_name": WHISPER_MODEL_NAME,
            "device": WHISPER_DEVICE,
            "compute_type": WHISPER_COMPUTE_TYPE,
            "loaded": self._model_loaded,
            "sample_rate": AUDIO_SAMPLE_RATE,
        }
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_loaded = False
            logger.info("Whisper model unloaded")


# Singleton instance for global access
_stt_engine: Optional[STTEngine] = None


def get_stt_engine() -> STTEngine:
    """Get or create the global STT engine instance."""
    global _stt_engine
    if _stt_engine is None:
        _stt_engine = STTEngine()
    return _stt_engine
