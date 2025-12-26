#!/bin/bash
# Build Frontend Script
# =====================
# Build the React frontend and copy to backend/server/static

set -e

echo "=== Building LinguaBridge Frontend ==="

# Navigate to frontend
cd "$(dirname "$0")/../../frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Build
echo "Building production bundle..."
npm run build

# Copy to static folder
echo "Copying to backend/server/static..."
rm -rf ../backend/server/static/*
cp -r dist/* ../backend/server/static/

echo "=== Frontend Build Complete ==="
echo "Files copied to: backend/server/static/"
