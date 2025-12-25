"""
LinguaBridge Model Constants
============================
Single source of truth for all model paths and configurations.
Rule 4 compliant: No magic numbers/strings in logic files.
"""

from pathlib import Path

# =============================================================================
# BASE PATHS
# =============================================================================
BACKEND_DIR = Path(__file__).parent.parent
STORAGE_DIR = BACKEND_DIR / "storage"
VOICE_MODELS_DIR = STORAGE_DIR / "voice_models"

# =============================================================================
# STT (Speech-to-Text) - Faster-Whisper
# =============================================================================
WHISPER_MODELS_DIR = VOICE_MODELS_DIR / "whisper"
WHISPER_MODEL_NAME = "base"  # Options: tiny, base, small, medium, large-v3
WHISPER_MODEL_PATH = WHISPER_MODELS_DIR / WHISPER_MODEL_NAME
WHISPER_COMPUTE_TYPE = "int8"  # CPU-optimized, options: int8, float16, float32
WHISPER_DEVICE = "cpu"  # CPU-first design
WHISPER_BEAM_SIZE = 5
WHISPER_VAD_FILTER = True  # Voice Activity Detection for better accuracy

# =============================================================================
# NMT (Neural Machine Translation) - Argos Translate
# =============================================================================
ARGOS_MODELS_DIR = VOICE_MODELS_DIR / "argos"

# Supported language pairs (ISO 639-1 codes)
ARGOS_LANGUAGE_PAIRS = [
    ("en", "ja"),  # English → Japanese (primary)
    ("ja", "en"),  # Japanese → English (optional)
    ("en", "hi"),  # English → Hindi (optional)
    ("hi", "en"),  # Hindi → English (optional)
]

# Language display names for UI
LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ko": "Korean",
}

# =============================================================================
# TTS (Text-to-Speech) - Piper
# =============================================================================
PIPER_MODELS_DIR = VOICE_MODELS_DIR / "piper"

# Default voice configurations
# Each voice requires: .onnx file + .onnx.json config file
PIPER_VOICES = {
    "en_US": {
        "model_name": "en_US-lessac-medium",
        "onnx_file": "en_US-lessac-medium.onnx",
        "json_file": "en_US-lessac-medium.onnx.json",
    },
    "hi_IN": {
        "model_name": "hi_IN-pratham-medium",
        "onnx_file": "hi_IN-pratham-medium.onnx",
        "json_file": "hi_IN-pratham-medium.onnx.json",
    },
}

PIPER_DEFAULT_VOICE = "en_US"
PIPER_SAMPLE_RATE = 22050  # Hz, standard for Piper voices
PIPER_LENGTH_SCALE = 1.0  # Speech speed (1.0 = normal)

# =============================================================================
# AUDIO SETTINGS
# =============================================================================
AUDIO_SAMPLE_RATE = 16000  # Hz, required for Whisper
AUDIO_CHANNELS = 1  # Mono
AUDIO_CHUNK_DURATION_MS = 30  # Milliseconds per chunk for streaming
TEMP_AUDIO_DIR = STORAGE_DIR / "temp_audio"

# =============================================================================
# MODEL DOWNLOAD URLS
# =============================================================================
# Whisper models are auto-downloaded by faster-whisper
# Argos models are auto-downloaded by argostranslate

# Piper voice download URLs (Hugging Face)
PIPER_DOWNLOAD_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
PIPER_VOICE_URLS = {
    "en_US": {  # English voice
        "onnx": f"{PIPER_DOWNLOAD_BASE_URL}/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "json": f"{PIPER_DOWNLOAD_BASE_URL}/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    },
    "hi_IN": {  # Hindi voice
        "onnx": f"{PIPER_DOWNLOAD_BASE_URL}/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx",
        "json": f"{PIPER_DOWNLOAD_BASE_URL}/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx.json",
    },
}
