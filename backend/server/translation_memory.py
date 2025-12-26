"""
LinguaBridge Translation Memory
===============================
SQLite-backed translation cache for consistency and speed.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database path
DB_PATH = Path(__file__).parent.parent / "database" / "translation_memory.db"

# =============================================================================
# DATABASE SCHEMA
# =============================================================================

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS translations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    """Initialize database schema."""
    conn = get_connection()
    try:
        conn.executescript(CREATE_SQL)
        conn.commit()
        logger.info(f"Translation memory initialized: {DB_PATH}")
    finally:
        conn.close()


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize(text: str) -> str:
    """Normalize text for lookup (lowercase, strip, single spaces)."""
    return " ".join(text.strip().lower().split())


# =============================================================================
# MEMORY OPERATIONS
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
            logger.debug(f"Memory hit: '{src_text[:30]}...'")
            return row["tgt_text"]
        return None
    finally:
        conn.close()


def save_translation(
    src_text: str,
    src_lang: str,
    tgt_text: str,
    tgt_lang: str,
    source: str = "unknown"
):
    """Save translation to memory."""
    normalized = normalize(src_text)
    
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO translations 
            (src_lang, tgt_lang, src_text, src_normalized, tgt_text, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (src_lang, tgt_lang, src_text, normalized, tgt_text, source)
        )
        conn.commit()
        logger.debug(f"Memory saved: '{src_text[:30]}...' -> '{tgt_text[:30]}...'")
    finally:
        conn.close()


def get_recent(
    src_lang: str,
    tgt_lang: str,
    limit: int = 5
) -> List[Tuple[str, str]]:
    """
    Get recent translations for context.
    
    Returns: List of (src_text, tgt_text) tuples
    """
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
        
        return {
            "total_entries": total,
            "language_pairs": pairs,
            "db_path": str(DB_PATH),
        }
    finally:
        conn.close()


def clear_memory():
    """Clear all translations (use with caution)."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM translations")
        conn.commit()
        logger.warning("Translation memory cleared!")
    finally:
        conn.close()
