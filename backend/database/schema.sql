-- LinguaBridge Database Schema
-- =============================
-- Run: sqlite3 backend/database/conversations.db < backend/database/schema.sql

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    direction TEXT NOT NULL,
    original_text TEXT NOT NULL,
    translated_text TEXT NOT NULL,
    audio_path TEXT,
    source_lang TEXT DEFAULT 'en',
    target_lang TEXT DEFAULT 'hi'
);

CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp);
CREATE INDEX IF NOT EXISTS idx_source_lang ON conversations(source_lang);
CREATE INDEX IF NOT EXISTS idx_target_lang ON conversations(target_lang);
