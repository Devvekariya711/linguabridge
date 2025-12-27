"""
LinguaBridge Embeddings Engine
==============================
Sentence-transformer embeddings for vector search.
Uses GPU when available for speed.
"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
from .constants import get_env, get_env_bool

EMBEDDING_MODEL = get_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
USE_GPU = get_env_bool("EMBEDDING_USE_GPU", True)

# =============================================================================
# SINGLETON MODEL
# =============================================================================
_model = None
_device = None


def _get_device() -> str:
    """Detect best available device."""
    global _device
    if _device is not None:
        return _device
    
    if USE_GPU:
        try:
            import torch
            if torch.cuda.is_available():
                _device = "cuda"
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                _device = "cpu"
                logger.info("GPU not available, using CPU")
        except ImportError:
            _device = "cpu"
            logger.info("PyTorch not found, using CPU")
    else:
        _device = "cpu"
        logger.info("GPU disabled in config, using CPU")
    
    return _device


def _get_model():
    """Get or load the embedding model."""
    global _model
    if _model is not None:
        return _model
    
    try:
        from sentence_transformers import SentenceTransformer
        
        device = _get_device()
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        
        logger.info(f"Embedding model loaded on {device}")
        return _model
        
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        raise
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


# =============================================================================
# PUBLIC API
# =============================================================================

def get_embedding(text: str) -> List[float]:
    """
    Get embedding vector for text.
    
    Args:
        text: Input text
        
    Returns:
        List of floats (embedding vector)
    """
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for multiple texts (batch).
    
    Args:
        texts: List of input texts
        
    Returns:
        List of embedding vectors
    """
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def get_embedding_dimension() -> int:
    """Get the dimension of embeddings."""
    model = _get_model()
    return model.get_sentence_embedding_dimension()


def is_available() -> bool:
    """Check if embeddings are available."""
    try:
        _get_model()
        return True
    except Exception:
        return False


def get_info() -> dict:
    """Get embedding engine info."""
    try:
        model = _get_model()
        return {
            "available": True,
            "model": EMBEDDING_MODEL,
            "device": _device,
            "dimension": model.get_sentence_embedding_dimension(),
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }
