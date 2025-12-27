"""
LinguaBridge Vector Database
============================
ChromaDB-based vector store for translation memory.
Enables fuzzy matching via semantic similarity.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple
import time

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
from .constants import get_env, get_env_float, BACKEND_DIR

CHROMA_DB_PATH = Path(get_env("CHROMA_DB_PATH", str(BACKEND_DIR / "database" / "chroma")))
SIMILARITY_THRESHOLD = get_env_float("VECTOR_SIMILARITY_THRESHOLD", 0.85)
COLLECTION_NAME = "translations"

# =============================================================================
# CHROMADB CLIENT
# =============================================================================
_client = None
_collection = None


def _get_client():
    """Get or create ChromaDB client."""
    global _client
    if _client is not None:
        return _client
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Ensure directory exists
        CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
        
        # Create persistent client
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        
        logger.info(f"ChromaDB initialized at: {CHROMA_DB_PATH}")
        return _client
        
    except ImportError:
        logger.error("chromadb not installed. Run: pip install chromadb")
        raise


def _get_collection():
    """Get or create translations collection."""
    global _collection
    if _collection is not None:
        return _collection
    
    client = _get_client()
    
    # Create collection with cosine similarity
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    logger.info(f"ChromaDB collection '{COLLECTION_NAME}' ready with {_collection.count()} entries")
    return _collection


# =============================================================================
# PUBLIC API
# =============================================================================

def add_translation(
    doc_id: str,
    src_text: str,
    tgt_text: str,
    src_lang: str,
    tgt_lang: str,
    embedding: List[float],
    source: str = "unknown"
):
    """
    Add a translation to the vector store.
    
    Args:
        doc_id: Unique ID for this entry
        src_text: Source text
        tgt_text: Target (translated) text
        src_lang: Source language code
        tgt_lang: Target language code
        embedding: Pre-computed embedding vector
        source: Translation source (llm, argos, user)
    """
    collection = _get_collection()
    
    try:
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[src_text],
            metadatas=[{
                "src_text": src_text,
                "tgt_text": tgt_text,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "source": source,
                "timestamp": str(int(time.time())),
            }]
        )
        logger.debug(f"Added to vector DB: {doc_id}")
    except Exception as e:
        # Might already exist, try update
        logger.debug(f"Add failed, trying update: {e}")
        try:
            collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[src_text],
                metadatas=[{
                    "src_text": src_text,
                    "tgt_text": tgt_text,
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "source": source,
                    "timestamp": str(int(time.time())),
                }]
            )
        except Exception as e2:
            logger.error(f"Vector DB add/update failed: {e2}")


def search_similar(
    query_embedding: List[float],
    src_lang: str,
    tgt_lang: str,
    n_results: int = 5,
    threshold: float = None
) -> List[Tuple[str, str, float]]:
    """
    Search for similar translations.
    
    Args:
        query_embedding: Query text embedding
        src_lang: Filter by source language
        tgt_lang: Filter by target language
        n_results: Max results to return
        threshold: Minimum similarity (0-1, higher = more similar)
        
    Returns:
        List of (src_text, tgt_text, similarity) tuples
    """
    collection = _get_collection()
    threshold = threshold or SIMILARITY_THRESHOLD
    
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={
                "$and": [
                    {"src_lang": {"$eq": src_lang}},
                    {"tgt_lang": {"$eq": tgt_lang}},
                ]
            },
            include=["metadatas", "distances"]
        )
        
        matches = []
        if results and results["metadatas"] and results["metadatas"][0]:
            for i, meta in enumerate(results["metadatas"][0]):
                # ChromaDB returns distance, convert to similarity
                # For cosine: similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 1.0
                similarity = 1.0 - distance
                
                if similarity >= threshold:
                    matches.append((
                        meta.get("src_text", ""),
                        meta.get("tgt_text", ""),
                        round(similarity, 3)
                    ))
        
        return matches
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


def find_best_match(
    query_embedding: List[float],
    src_lang: str,
    tgt_lang: str,
    threshold: float = None
) -> Optional[Tuple[str, str, float]]:
    """
    Find the best matching translation above threshold.
    
    Returns:
        (src_text, tgt_text, similarity) or None if no match
    """
    matches = search_similar(query_embedding, src_lang, tgt_lang, n_results=1, threshold=threshold)
    return matches[0] if matches else None


def get_count() -> int:
    """Get total entries in vector DB."""
    try:
        collection = _get_collection()
        return collection.count()
    except Exception:
        return 0


def get_stats() -> dict:
    """Get vector DB statistics."""
    try:
        collection = _get_collection()
        return {
            "available": True,
            "db_path": str(CHROMA_DB_PATH),
            "collection": COLLECTION_NAME,
            "total_entries": collection.count(),
            "similarity_threshold": SIMILARITY_THRESHOLD,
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
        }


def delete_all():
    """Delete all entries (use with caution)."""
    global _collection
    try:
        client = _get_client()
        client.delete_collection(COLLECTION_NAME)
        _collection = None
        logger.warning("Vector DB cleared!")
    except Exception as e:
        logger.error(f"Failed to clear vector DB: {e}")
