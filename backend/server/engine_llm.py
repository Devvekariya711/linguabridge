"""
LinguaBridge LLM Engine
=======================
Wrapper for Ollama LLM translation using local models.
Supports Qwen2.5, Deepseek, Mistral, Llama, etc.

Configuration via .env file:
- LLM_MODEL: Model to use
- LLM_TIMEOUT: Timeout in seconds
- LLM_ENABLED: Enable/disable LLM
"""

import subprocess
import json
import time
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION FROM .ENV
# =============================================================================
from .constants import LLM_MODEL, LLM_TIMEOUT, LLM_ENABLED

# Default model (from .env)
DEFAULT_MODEL = LLM_MODEL

# Timeout for LLM calls (from .env)
LLM_TIMEOUT_SEC = LLM_TIMEOUT

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

TRANSLATION_PROMPT = """You are a professional translator. Translate the following text from {src_lang} to {tgt_lang}.

Rules:
- Output ONLY the translation, nothing else
- Keep proper names as-is or transliterate appropriately
- Maintain the tone and formality of the original
- For Hindi output, use Devanagari script

Text to translate: "{text}"

Translation:"""

TRANSLATION_WITH_CONTEXT_PROMPT = """You are a professional translator. Use the context below for consistency.

Previous translations for reference:
{context}

Now translate from {src_lang} to {tgt_lang}:

Text: "{text}"

Translation:"""


# =============================================================================
# OLLAMA API
# =============================================================================

def check_ollama_available() -> bool:
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def list_models() -> list:
    """List available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return []
        
        lines = result.stdout.strip().split("\n")[1:]  # Skip header
        models = []
        for line in lines:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except Exception:
        return []


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: int = LLM_TIMEOUT_SEC
) -> Tuple[str, float]:
    """
    Generate text using Ollama.
    
    Returns: (response_text, latency_seconds)
    """
    start = time.perf_counter()
    
    try:
        # Use ollama run with prompt as argument
        # This works better than piping to stdin
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace'
        )
        
        latency = time.perf_counter() - start
        
        if result.returncode != 0:
            raise RuntimeError(f"Ollama error: {result.stderr.strip()}")
        
        response = result.stdout.strip()
        
        # Clean up response (remove quotes if present)
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        
        return response, latency
        
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"LLM generation timed out after {timeout}s")
    except FileNotFoundError:
        raise RuntimeError("Ollama not found. Install from https://ollama.ai")


# =============================================================================
# TRANSLATION API
# =============================================================================

def translate(
    text: str,
    src_lang: str,
    tgt_lang: str,
    context: Optional[str] = None,
    model: str = DEFAULT_MODEL
) -> Tuple[str, dict]:
    """
    Translate text using LLM.
    
    Args:
        text: Text to translate
        src_lang: Source language code (en, hi, ja, etc.)
        tgt_lang: Target language code
        context: Optional context from previous translations
        model: Ollama model to use
        
    Returns:
        (translated_text, metadata_dict)
    """
    # Map language codes to full names for better prompts
    lang_names = {
        "en": "English",
        "hi": "Hindi",
        "ja": "Japanese",
        "zh": "Chinese",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ko": "Korean",
        "ar": "Arabic",
        "ru": "Russian",
    }
    
    src_name = lang_names.get(src_lang, src_lang)
    tgt_name = lang_names.get(tgt_lang, tgt_lang)
    
    # Build prompt
    if context:
        prompt = TRANSLATION_WITH_CONTEXT_PROMPT.format(
            context=context,
            src_lang=src_name,
            tgt_lang=tgt_name,
            text=text
        )
    else:
        prompt = TRANSLATION_PROMPT.format(
            src_lang=src_name,
            tgt_lang=tgt_name,
            text=text
        )
    
    # Generate
    try:
        response, latency = generate(prompt, model=model)
        
        # Clean up response
        # Sometimes LLM adds quotes or explanations
        lines = response.strip().split("\n")
        # Take first non-empty line as translation
        translation = ""
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Translation:"):
                translation = line
                break
        
        if not translation:
            translation = response.strip()
        
        metadata = {
            "source": "llm",
            "model": model,
            "latency": round(latency, 2),
        }
        
        logger.info(f"LLM translate: '{text[:30]}...' -> '{translation[:30]}...' ({latency:.2f}s)")
        
        return translation, metadata
        
    except Exception as e:
        logger.error(f"LLM translation failed: {e}")
        raise


# =============================================================================
# SINGLETON
# =============================================================================

_llm_available = None

def is_llm_available() -> bool:
    """Check if LLM is available (cached)."""
    global _llm_available
    if _llm_available is None:
        _llm_available = check_ollama_available()
    return _llm_available


def get_model_info() -> dict:
    """Get LLM engine info."""
    return {
        "available": is_llm_available(),
        "default_model": DEFAULT_MODEL,
        "installed_models": list_models() if is_llm_available() else [],
    }
