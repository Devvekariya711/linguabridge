"""
LinguaBridge TTS Engine
=======================
Text-to-Speech using Piper TTS (ONNX-based, CPU-optimized).

Models: ~60-100 MB per voice
Stored at: backend/storage/voice_models/piper/
Required files per voice: .onnx + .onnx.json
"""

import logging
import wave
import io
from pathlib import Path
from typing import Optional, Union
import numpy as np

from .constants import (
    PIPER_MODELS_DIR,
    PIPER_VOICES,
    PIPER_DEFAULT_VOICE,
    PIPER_SAMPLE_RATE,
    PIPER_LENGTH_SCALE,
    PIPER_VOICE_URLS,
    TEMP_AUDIO_DIR,
)

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Text-to-Speech engine using Piper TTS.
    
    Converts translated text → natural-sounding speech.
    Uses ONNX models for CPU-optimized inference.
    
    Each voice requires two files:
        - .onnx (model weights)
        - .onnx.json (voice configuration)
    """
    
    def __init__(self):
        self._voices: dict = {}  # Loaded voice instances
        self._piper_available = False
        self._check_piper_installed()
    
    def _check_piper_installed(self) -> None:
        """Check if piper-tts is installed."""
        try:
            from piper import PiperVoice
            self._piper_available = True
            logger.info("✅ Piper TTS is available")
        except ImportError:
            self._piper_available = False
            logger.warning("⚠️ piper-tts not installed. Run: pip install piper-tts")
    
    def _ensure_dirs(self) -> None:
        """Ensure required directories exist."""
        PIPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    
    def _get_voice_paths(self, voice_key: str) -> tuple:
        """
        Get paths for voice model files.
        
        Returns:
            Tuple of (onnx_path, json_path)
        """
        if voice_key not in PIPER_VOICES:
            raise ValueError(f"Unknown voice: {voice_key}. Available: {list(PIPER_VOICES.keys())}")
        
        voice_config = PIPER_VOICES[voice_key]
        onnx_path = PIPER_MODELS_DIR / voice_config["onnx_file"]
        json_path = PIPER_MODELS_DIR / voice_config["json_file"]
        
        return onnx_path, json_path
    
    def is_voice_installed(self, voice_key: str) -> bool:
        """
        Check if a voice is installed (both .onnx and .onnx.json present).
        
        Stress Tests:
            - Edge Case 1: Only .onnx exists without .json → returns False
            - Edge Case 2: Files exist but are corrupted → returns True (checked at load time)
        """
        try:
            onnx_path, json_path = self._get_voice_paths(voice_key)
            return onnx_path.exists() and json_path.exists()
        except ValueError:
            return False
    
    def download_voice(self, voice_key: str) -> bool:
        """
        Download a voice model from Hugging Face.
        
        Args:
            voice_key: Voice identifier (e.g., 'en_US')
        
        Returns:
            True if successful, False otherwise
        
        Stress Tests:
            - Edge Case 1: Network timeout → return False, cleanup partial files
            - Edge Case 2: Disk full → return False, don't corrupt state
        """
        self._ensure_dirs()
        
        if voice_key not in PIPER_VOICE_URLS:
            logger.error(f"No download URL configured for voice: {voice_key}")
            return False
        
        if self.is_voice_installed(voice_key):
            logger.info(f"Voice {voice_key} already installed")
            return True
        
        try:
            import requests
            from tqdm import tqdm
            
            urls = PIPER_VOICE_URLS[voice_key]
            onnx_path, json_path = self._get_voice_paths(voice_key)
            
            # Download .onnx file
            logger.info(f"Downloading {voice_key} ONNX model...")
            response = requests.get(urls["onnx"], stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(onnx_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"{voice_key}.onnx") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Download .onnx.json file
            logger.info(f"Downloading {voice_key} config...")
            response = requests.get(urls["json"], timeout=30)
            response.raise_for_status()
            
            with open(json_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"✅ Voice {voice_key} downloaded successfully")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed: {e}")
            # Cleanup partial downloads
            for path in [onnx_path, json_path]:
                if path.exists():
                    try:
                        path.unlink()
                    except Exception:
                        pass
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading voice: {e}")
            return False
    
    def load_voice(self, voice_key: str, auto_download: bool = False) -> bool:
        """
        Load a voice into memory.
        
        Args:
            voice_key: Voice identifier
            auto_download: If True, download voice if not present
        
        Returns:
            True if voice loaded successfully
        """
        if not self._piper_available:
            logger.error("Piper TTS not installed")
            return False
        
        # Already loaded?
        if voice_key in self._voices:
            return True
        
        # Check if installed
        if not self.is_voice_installed(voice_key):
            if auto_download:
                if not self.download_voice(voice_key):
                    return False
            else:
                logger.error(f"Voice {voice_key} not installed. Call download_voice() first.")
                return False
        
        try:
            from piper import PiperVoice
            
            onnx_path, json_path = self._get_voice_paths(voice_key)
            
            logger.info(f"Loading voice: {voice_key}")
            self._voices[voice_key] = PiperVoice.load(
                str(onnx_path),
                config_path=str(json_path),
            )
            
            logger.info(f"✅ Voice {voice_key} loaded")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load voice {voice_key}: {e}")
            return False
    
    def synthesize(
        self,
        text: str,
        voice_key: Optional[str] = None,
        length_scale: Optional[float] = None,
        auto_load: bool = True,
    ) -> np.ndarray:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice_key: Voice to use (default: PIPER_DEFAULT_VOICE)
            length_scale: Speech speed (1.0 = normal, <1.0 = faster, >1.0 = slower)
            auto_load: If True, automatically load voice if not loaded
        
        Returns:
            Audio as numpy array (int16)
        
        Stress Tests:
            - Edge Case 1: Empty text → return empty array
            - Edge Case 2: Very long text (>10000 chars) → may be slow, handle gracefully
        """
        if not self._piper_available:
            raise RuntimeError("Piper TTS not installed")
        
        voice_key = voice_key or PIPER_DEFAULT_VOICE
        length_scale = length_scale or PIPER_LENGTH_SCALE
        
        # Handle empty text
        if not text or not text.strip():
            return np.array([], dtype=np.int16)
        
        # Ensure voice is loaded
        if voice_key not in self._voices:
            if auto_load:
                if not self.load_voice(voice_key, auto_download=True):
                    raise RuntimeError(f"Failed to load voice: {voice_key}")
            else:
                raise RuntimeError(f"Voice {voice_key} not loaded")
        
        voice = self._voices[voice_key]
        
        try:
            # Synthesize - Piper returns generator of AudioChunk objects
            audio_chunks = []
            for audio_chunk in voice.synthesize(text):
                # AudioChunk has audio_int16_array (numpy array)
                audio_chunks.append(audio_chunk.audio_int16_array)
            
            # Combine all chunks
            if not audio_chunks:
                return np.array([], dtype=np.int16)
            
            audio_data = np.concatenate(audio_chunks)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise RuntimeError(f"Speech synthesis failed: {e}")
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: Union[str, Path],
        voice_key: Optional[str] = None,
        length_scale: Optional[float] = None,
    ) -> Path:
        """
        Synthesize speech and save to WAV file.
        
        Args:
            text: Text to synthesize
            output_path: Path to save WAV file
            voice_key: Voice to use
            length_scale: Speech speed
        
        Returns:
            Path to the saved file
        """
        if not self._piper_available:
            raise RuntimeError("Piper TTS not installed")
        
        voice_key = voice_key or PIPER_DEFAULT_VOICE
        length_scale = length_scale or PIPER_LENGTH_SCALE
        output_path = Path(output_path)
        
        # Ensure voice is loaded
        if voice_key not in self._voices:
            if not self.load_voice(voice_key, auto_download=True):
                raise RuntimeError(f"Failed to load voice: {voice_key}")
        
        voice = self._voices[voice_key]
        
        try:
            with wave.open(str(output_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(PIPER_SAMPLE_RATE)
                
                # Iterate AudioChunk generator and write int16 array as bytes
                for audio_chunk in voice.synthesize(text):
                    wav_file.writeframes(audio_chunk.audio_int16_array.tobytes())
            
            logger.info(f"Audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise
    
    def get_available_voices(self) -> list:
        """Get list of configured voices."""
        return [
            {
                "key": key,
                "model_name": config["model_name"],
                "installed": self.is_voice_installed(key),
                "loaded": key in self._voices,
            }
            for key, config in PIPER_VOICES.items()
        ]
    
    def get_model_info(self) -> dict:
        """Return information about TTS engine."""
        return {
            "available": self._piper_available,
            "models_dir": str(PIPER_MODELS_DIR),
            "sample_rate": PIPER_SAMPLE_RATE,
            "voices": self.get_available_voices(),
            "loaded_voices": list(self._voices.keys()),
        }
    
    def unload_voice(self, voice_key: str) -> None:
        """Unload a voice from memory."""
        if voice_key in self._voices:
            del self._voices[voice_key]
            logger.info(f"Voice {voice_key} unloaded")
    
    def unload_all(self) -> None:
        """Unload all voices from memory."""
        self._voices.clear()
        logger.info("All voices unloaded")


# Singleton instance for global access
_tts_engine: Optional[TTSEngine] = None


def get_tts_engine() -> TTSEngine:
    """Get or create the global TTS engine instance."""
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine()
    return _tts_engine
