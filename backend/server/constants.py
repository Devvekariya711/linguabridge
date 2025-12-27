"""
LinguaBridge Configuration
==========================
Single source of truth for all settings.
Reads from .env file - all settings are user-changeable.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# LOAD ENVIRONMENT
# =============================================================================
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)

def get_env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.getenv(key, default)

def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment."""
    return get_env(key, str(default)).lower() in ("true", "1", "yes")

def get_env_int(key: str, default: int = 0) -> int:
    """Get integer from environment."""
    try:
        return int(get_env(key, str(default)))
    except ValueError:
        return default

def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float from environment."""
    try:
        return float(get_env(key, str(default)))
    except ValueError:
        return default

# =============================================================================
# BASE PATHS
# =============================================================================
BACKEND_DIR = Path(__file__).parent.parent
STORAGE_DIR = BACKEND_DIR / "storage"
VOICE_MODELS_DIR = STORAGE_DIR / "voice_models"

# =============================================================================
# SERVER
# =============================================================================
SERVER_HOST = get_env("HOST", "0.0.0.0")
SERVER_PORT = get_env_int("PORT", 8000)
DEBUG = get_env_bool("DEBUG", False)
CORS_ALLOWED_ORIGINS = get_env("CORS_ALLOWED_ORIGINS", "*").split(",")

# =============================================================================
# STT (Speech-to-Text) - Faster-Whisper
# =============================================================================
# User can change model in .env: tiny, base, small, medium, large-v2, large-v3
STT_MODEL = get_env("STT_MODEL", "base")
STT_DEVICE = get_env("STT_DEVICE", "cuda")
STT_COMPUTE_TYPE = get_env("STT_COMPUTE_TYPE", "float16")

WHISPER_MODELS_DIR = Path(get_env("WHISPER_CACHE_DIR", str(VOICE_MODELS_DIR / "whisper")))
WHISPER_MODEL_NAME = STT_MODEL
WHISPER_MODEL_PATH = WHISPER_MODELS_DIR / WHISPER_MODEL_NAME
WHISPER_COMPUTE_TYPE = STT_COMPUTE_TYPE
WHISPER_DEVICE = STT_DEVICE
WHISPER_BEAM_SIZE = 1  # Faster for streaming
WHISPER_VAD_FILTER = True

# Streaming STT Parameters
STT_WINDOW_SECONDS = get_env_float("STT_WINDOW_SECONDS", 1.0)
STT_STEP_SECONDS = get_env_float("STT_STEP_SECONDS", 0.25)
CLIENT_FRAME_MS = get_env_int("CLIENT_FRAME_MS", 250)

# =============================================================================
# PIPELINE WORKERS
# =============================================================================
NMT_WORKERS = get_env_int("NMT_WORKERS", 2)
TTS_WORKERS = get_env_int("TTS_WORKERS", 2)
LIVE_CHUNK_MAX_WORDS = get_env_int("LIVE_CHUNK_MAX_WORDS", 4)

# =============================================================================
# NMT (Neural Machine Translation) - Argos Translate
# =============================================================================
ARGOS_MODELS_DIR = Path(get_env("ARGOS_DATA_DIR", str(VOICE_MODELS_DIR / "argos")))

# Parse language pairs from env (format: en-hi,en-ja,hi-en)
_argos_pairs_str = get_env("ARGOS_LANGUAGE_PAIRS", "en-hi,en-ja,hi-en,ja-en")
ARGOS_LANGUAGE_PAIRS = [
    tuple(pair.split("-")) for pair in _argos_pairs_str.split(",")
    if "-" in pair
]

# Language display names
LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "zh": "Chinese",
    "ko": "Korean",
    "ar": "Arabic",
    "ru": "Russian",
}

# =============================================================================
# LLM (Large Language Model) - Ollama
# =============================================================================
LLM_ENABLED = get_env_bool("LLM_ENABLED", True)
LLM_MODEL = get_env("LLM_MODEL", "huihui_ai/qwen2.5-coder-abliterate:14b")
LLM_TIMEOUT = get_env_int("LLM_TIMEOUT", 60)
LLM_MIN_WORDS = get_env_int("LLM_MIN_WORDS", 4)  # Use LLM for phrases > N words

# =============================================================================
# TTS (Text-to-Speech) - Piper
# =============================================================================
PIPER_MODELS_DIR = Path(get_env("PIPER_MODEL_DIR", str(VOICE_MODELS_DIR / "piper")))
TTS_SPEED = get_env_float("TTS_SPEED", 1.0)

# Voice configurations - parsed from env
TTS_VOICE_EN = get_env("TTS_VOICE_EN", "en_US-lessac-medium")
TTS_VOICE_HI = get_env("TTS_VOICE_HI", "hi_IN-pratham-medium")

PIPER_VOICES = {
    "en_US": {
        "model_name": TTS_VOICE_EN,
        "onnx_file": f"{TTS_VOICE_EN}.onnx",
        "json_file": f"{TTS_VOICE_EN}.onnx.json",
    },
    "hi_IN": {
        "model_name": TTS_VOICE_HI,
        "onnx_file": f"{TTS_VOICE_HI}.onnx",
        "json_file": f"{TTS_VOICE_HI}.onnx.json",
    },
}

PIPER_DEFAULT_VOICE = "en_US"
PIPER_SAMPLE_RATE = 22050
PIPER_LENGTH_SCALE = TTS_SPEED

# =============================================================================
# TRANSLATION MEMORY
# =============================================================================
MEMORY_ENABLED = get_env_bool("MEMORY_ENABLED", True)
MEMORY_DB_PATH = Path(get_env("MEMORY_DB_PATH", str(BACKEND_DIR / "database" / "translation_memory.db")))
MEMORY_CONTEXT_LIMIT = get_env_int("MEMORY_CONTEXT_LIMIT", 5)

# =============================================================================
# DATABASE
# =============================================================================
SQLITE_PATH = Path(get_env("SQLITE_PATH", str(BACKEND_DIR / "database" / "conversations.db")))

# =============================================================================
# AUDIO SETTINGS
# =============================================================================
AUDIO_SAMPLE_RATE = 16000  # Required for Whisper
AUDIO_CHANNELS = 1  # Mono
AUDIO_CHUNK_DURATION_MS = 30
TEMP_AUDIO_DIR = STORAGE_DIR / "temp_audio"

# =============================================================================
# PERFORMANCE
# =============================================================================
PRELOAD_MODELS = get_env_bool("PRELOAD_MODELS", True)
WORKERS = get_env_int("WORKERS", 4)

# =============================================================================
# MODEL DOWNLOAD URLS
# =============================================================================
PIPER_DOWNLOAD_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
PIPER_VOICE_URLS = {
    "en_US": {
        "onnx": f"{PIPER_DOWNLOAD_BASE_URL}/en/en_US/lessac/medium/{TTS_VOICE_EN}.onnx",
        "json": f"{PIPER_DOWNLOAD_BASE_URL}/en/en_US/lessac/medium/{TTS_VOICE_EN}.onnx.json",
    },
    "hi_IN": {
        "onnx": f"{PIPER_DOWNLOAD_BASE_URL}/hi/hi_IN/pratham/medium/{TTS_VOICE_HI}.onnx",
        "json": f"{PIPER_DOWNLOAD_BASE_URL}/hi/hi_IN/pratham/medium/{TTS_VOICE_HI}.onnx.json",
    },
}

# =============================================================================
# PRINT CONFIG (for debugging)
# =============================================================================
def print_config():
    """Print current configuration."""
    print("=" * 50)
    print("LinguaBridge Configuration")
    print("=" * 50)
    print(f"STT Model:     {STT_MODEL}")
    print(f"LLM Enabled:   {LLM_ENABLED}")
    print(f"LLM Model:     {LLM_MODEL}")
    print(f"LLM Timeout:   {LLM_TIMEOUT}s")
    print(f"LLM Min Words: {LLM_MIN_WORDS}")
    print(f"TTS Voice EN:  {TTS_VOICE_EN}")
    print(f"TTS Voice HI:  {TTS_VOICE_HI}")
    print(f"TTS Speed:     {TTS_SPEED}")
    print(f"Memory:        {MEMORY_ENABLED}")
    print(f"Preload:       {PRELOAD_MODELS}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
