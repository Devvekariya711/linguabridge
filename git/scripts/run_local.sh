#!/bin/bash
# Run Local Development Server
# ============================
# Start the LinguaBridge server locally

set -e

cd "$(dirname "$0")/../.."

echo "=== LinguaBridge Local Development ==="

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check dependencies
pip install -q -r requirements.txt

# Start server
echo "Starting server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

python -m uvicorn backend.server.server_main:asgi_app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload
