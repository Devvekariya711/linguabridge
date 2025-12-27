"""
LinguaBridge Translation Memory
===============================
SQLite + ChromaDB powered translation cache.
Supports exact match AND fuzzy vector search.
"""

import sqlite3
import logging
import hashlib
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
from .constants import get_env_bool, MEMORY_DB_PATH, MEMORY_ENABLED

DB_PATH = MEMORY_DB_PATH
VECTOR_SEARCH_ENABLED = get_env_bool("VECTOR_SEARCH_ENABLED", True)

# =============================================================================
# DATABASE SCHEMA
# =============================================================================

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS translations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT UNIQUE,
    src_lang TEXT NOT NULL,
    tgt_lang TEXT NOT NULL,
    src_text TEXT NOT NULL,
    src_normalized TEXT NOT NULL,
    tgt_text TEXT NOT NULL,
    source TEXT DEFAULT 'unknown',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lookup 
ON translations(src_normalized, src_lang, tgt_lang);

CREATE INDEX IF NOT EXISTS idx_lang_pair
ON translations(src_lang, tgt_lang);

CREATE INDEX IF NOT EXISTS idx_doc_id
ON translations(doc_id);
"""

# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database schema and vector DB."""
    # SQLite
    conn = get_connection()
    try:
        conn.executescript(CREATE_SQL)
        conn.commit()
        logger.info(f"Translation memory initialized: {DB_PATH}")
    finally:
        conn.close()
    
    # Vector DB (ChromaDB)
    if VECTOR_SEARCH_ENABLED:
        try:
            from . import vector_db
            vector_db._get_collection()  # Initialize
            logger.info("Vector search initialized (ChromaDB)")
        except Exception as e:
            logger.warning(f"Vector search init failed: {e}")


# =============================================================================
# HELPERS
# =============================================================================

def normalize(text: str) -> str:
    """Normalize text for lookup (lowercase, strip, single spaces)."""
    return " ".join(text.strip().lower().split())


def generate_doc_id(src_text: str, src_lang: str, tgt_lang: str) -> str:
    """Generate unique document ID for vector DB."""
    content = f"{src_lang}:{tgt_lang}:{normalize(src_text)}"
    return hashlib.md5(content.encode()).hexdigest()


# =============================================================================
# EXACT MATCH (SQLite)
# =============================================================================

def find_exact(
    src_text: str,
    src_lang: str,
    tgt_lang: str
) -> Optional[str]:
    """
    Find exact match in translation memory.
    
    Returns: translated text if found, None otherwise
    """
    normalized = normalize(src_text)
    
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT tgt_text FROM translations 
            WHERE src_normalized = ? AND src_lang = ? AND tgt_lang = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (normalized, src_lang, tgt_lang)
        )
        row = cursor.fetchone()
        if row:
            logger.debug(f"Exact match: '{src_text[:30]}...'")
            return row["tgt_text"]
        return None
    finally:
        conn.close()


# =============================================================================
# VECTOR SEARCH (ChromaDB)
# =============================================================================

def find_similar(
    src_text: str,
    src_lang: str,
    tgt_lang: str,
    threshold: float = 0.85
) -> Optional[Tuple[str, str, float]]:
    """
    Find similar translation using vector search.
    
    Returns: (matched_src, tgt_text, similarity) or None
    """
    if not VECTOR_SEARCH_ENABLED:
        return None
    
    try:
        from . import embeddings
        from . import vector_db
        
        # Get embedding for query
        query_embedding = embeddings.get_embedding(src_text)
        
        # Search vector DB
        match = vector_db.find_best_match(
            query_embedding, src_lang, tgt_lang, threshold=threshold
        )
        
        if match:
            logger.debug(f"Vector match ({match[2]:.2f}): '{src_text[:30]}...' ~ '{match[0][:30]}...'")
        
        return match
        
    except Exception as e:
        logger.debug(f"Vector search failed: {e}")
        return None


# =============================================================================
# SAVE TRANSLATION
# =============================================================================

def save_translation(
    src_text: str,
    src_lang: str,
    tgt_text: str,
    tgt_lang: str,
    source: str = "unknown"
):
    """Save translation to memory (SQLite + ChromaDB)."""
    normalized = normalize(src_text)
    doc_id = generate_doc_id(src_text, src_lang, tgt_lang)
    
    # Save to SQLite
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO translations 
            (doc_id, src_lang, tgt_lang, src_text, src_normalized, tgt_text, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, src_lang, tgt_lang, src_text, normalized, tgt_text, source)
        )
        conn.commit()
        logger.debug(f"Saved to SQLite: '{src_text[:30]}...'")
    finally:
        conn.close()
    
    # Save to Vector DB
    if VECTOR_SEARCH_ENABLED:
        try:
            from . import embeddings
            from . import vector_db
            
            embedding = embeddings.get_embedding(src_text)
            vector_db.add_translation(
                doc_id=doc_id,
                src_text=src_text,
                tgt_text=tgt_text,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                embedding=embedding,
                source=source
            )
            logger.debug(f"Saved to ChromaDB: '{src_text[:30]}...'")
        except Exception as e:
            logger.debug(f"Vector save failed: {e}")


# =============================================================================
# CONTEXT & STATS
# =============================================================================

def get_recent(
    src_lang: str,
    tgt_lang: str,
    limit: int = 5
) -> List[Tuple[str, str]]:
    """Get recent translations for context."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT src_text, tgt_text FROM translations
            WHERE src_lang = ? AND tgt_lang = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (src_lang, tgt_lang, limit)
        )
        return [(row["src_text"], row["tgt_text"]) for row in cursor.fetchall()]
    finally:
        conn.close()


def build_context(src_lang: str, tgt_lang: str, limit: int = 5) -> Optional[str]:
    """Build context string from recent translations."""
    recent = get_recent(src_lang, tgt_lang, limit)
    if not recent:
        return None
    
    lines = []
    for src, tgt in recent:
        lines.append(f'"{src}" -> "{tgt}"')
    
    return "\n".join(lines)


def get_stats() -> dict:
    """Get memory statistics."""
    conn = get_connection()
    try:
        cursor = conn.execute("SELECT COUNT(*) as total FROM translations")
        total = cursor.fetchone()["total"]
        
        cursor = conn.execute(
            """
            SELECT src_lang || '->' || tgt_lang as pair, COUNT(*) as count
            FROM translations
            GROUP BY src_lang, tgt_lang
            ORDER BY count DESC
            """
        )
        pairs = {row["pair"]: row["count"] for row in cursor.fetchall()}
        
        stats = {
            "total_entries": total,
            "language_pairs": pairs,
            "db_path": str(DB_PATH),
            "vector_search_enabled": VECTOR_SEARCH_ENABLED,
        }
        
        # Add vector DB stats if enabled
        if VECTOR_SEARCH_ENABLED:
            try:
                from . import vector_db
                stats["vector_db"] = vector_db.get_stats()
            except Exception:
                pass
        
        return stats
    finally:
        conn.close()


def clear_memory():
    """Clear all translations (use with caution)."""
    # Clear SQLite
    conn = get_connection()
    try:
        conn.execute("DELETE FROM translations")
        conn.commit()
        logger.warning("Translation memory cleared!")
    finally:
        conn.close()
    
    # Clear Vector DB
    if VECTOR_SEARCH_ENABLED:
        try:
            from . import vector_db
            vector_db.delete_all()
        except Exception as e:
            logger.error(f"Vector DB clear failed: {e}")
