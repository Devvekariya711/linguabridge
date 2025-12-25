"""
LinguaBridge Database Manager
=============================
Async SQLite operations for conversation storage.
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import aiosqlite

logger = logging.getLogger(__name__)

# Database path from environment or default
DB_PATH = Path(os.getenv(
    "SQLITE_PATH",
    Path(__file__).parent / "conversations.db"
))


async def init_db() -> None:
    """
    Initialize the database schema.
    Creates tables if they don't exist.
    
    Stress Tests:
        - Edge Case 1: DB file doesn't exist -> creates it
        - Edge Case 2: Schema already exists -> no-op
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                direction TEXT NOT NULL,
                original_text TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                audio_path TEXT,
                source_lang TEXT DEFAULT 'en',
                target_lang TEXT DEFAULT 'hi'
            )
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON conversations(timestamp)
        """)
        
        await db.commit()
        logger.info(f"Database initialized: {DB_PATH}")


async def save_conversation(
    original: str,
    translated: str,
    audio_path: Optional[str] = None,
    direction: str = "A->B",
    source_lang: str = "en",
    target_lang: str = "hi",
) -> int:
    """
    Save a conversation to the database.
    
    Args:
        original: Original text
        translated: Translated text
        audio_path: Optional path to audio file
        direction: Direction label (e.g., "A->B" or "B->A")
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        Row ID of inserted record
    
    Stress Tests:
        - Edge Case 1: Empty strings -> still saves
        - Edge Case 2: Very long text -> handles up to SQLite max
    """
    timestamp = datetime.now().isoformat()
    
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """
            INSERT INTO conversations 
            (timestamp, direction, original_text, translated_text, audio_path, source_lang, target_lang)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, direction, original, translated, audio_path, source_lang, target_lang)
        )
        await db.commit()
        
        row_id = cursor.lastrowid
        logger.debug(f"Saved conversation {row_id}: {original[:30]}...")
        return row_id


async def get_recent_conversations(
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Get recent conversations.
    
    Args:
        limit: Maximum number of records
        offset: Number of records to skip
        
    Returns:
        List of conversation dicts
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        
        cursor = await db.execute(
            """
            SELECT id, timestamp, direction, original_text, translated_text, 
                   audio_path, source_lang, target_lang
            FROM conversations
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        )
        
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def get_conversation_by_id(conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a specific conversation by ID.
    
    Args:
        conversation_id: The conversation ID
        
    Returns:
        Conversation dict or None
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        
        cursor = await db.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,)
        )
        
        row = await cursor.fetchone()
        return dict(row) if row else None


async def delete_conversation(conversation_id: int) -> bool:
    """
    Delete a conversation by ID.
    
    Args:
        conversation_id: The conversation ID
        
    Returns:
        True if deleted, False if not found
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,)
        )
        await db.commit()
        
        return cursor.rowcount > 0


async def prune_old_conversations(days: int = 30) -> int:
    """
    Delete conversations older than N days.
    
    Args:
        days: Maximum age in days
        
    Returns:
        Number of deleted records
    """
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM conversations WHERE timestamp < ?",
            (cutoff,)
        )
        await db.commit()
        
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"Pruned {deleted} conversations older than {days} days")
        return deleted


async def count_conversations() -> int:
    """Get total number of conversations."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM conversations")
        row = await cursor.fetchone()
        return row[0] if row else 0


async def search_conversations(
    query: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Search conversations by text content.
    
    Args:
        query: Search query
        limit: Maximum results
        
    Returns:
        List of matching conversations
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        
        cursor = await db.execute(
            """
            SELECT * FROM conversations
            WHERE original_text LIKE ? OR translated_text LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", limit)
        )
        
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
