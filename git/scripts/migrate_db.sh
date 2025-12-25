#!/bin/bash
# LinguaBridge Database Migration Script
# ======================================
# Usage: ./git/scripts/migrate_db.sh

set -e

DB_PATH="backend/database/conversations.db"
SCHEMA_PATH="backend/database/schema.sql"

echo "=== LinguaBridge DB Migration ==="

# Create directory if needed
mkdir -p "$(dirname "$DB_PATH")"

# Apply schema
if [ -f "$SCHEMA_PATH" ]; then
    echo "Applying schema from $SCHEMA_PATH..."
    sqlite3 "$DB_PATH" < "$SCHEMA_PATH"
    echo "✅ Schema applied successfully!"
else
    echo "⚠️ Schema file not found: $SCHEMA_PATH"
    exit 1
fi

# Show tables
echo ""
echo "Tables in database:"
sqlite3 "$DB_PATH" ".tables"

echo ""
echo "=== Migration Complete ==="
