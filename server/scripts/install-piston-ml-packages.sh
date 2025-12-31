#!/bin/bash
################################################################################
# Install ML packages in Piston for Python AI/ML courses
# Run this after Piston container is healthy
################################################################################

PISTON_URL="${PISTON_URL:-http://localhost:2000}"

echo "🔧 Installing ML packages in Piston..."
echo "   Piston URL: $PISTON_URL"

# Wait for Piston to be healthy
echo ""
echo "⏳ Waiting for Piston to be ready..."
until curl -sf "$PISTON_URL/api/v2/runtimes" > /dev/null 2>&1; do
    echo "   Piston not ready, waiting 5 seconds..."
    sleep 5
done
echo "✅ Piston is ready!"

# Check available runtimes
echo ""
echo "📋 Available runtimes:"
curl -sf "$PISTON_URL/api/v2/runtimes" | jq -r '.[].language' | sort | uniq | head -20

# Install Python runtime if not available
echo ""
echo "🐍 Checking Python runtime..."
PYTHON_AVAILABLE=$(curl -sf "$PISTON_URL/api/v2/runtimes" | jq -r '.[] | select(.language == "python") | .version' | head -1)

if [ -z "$PYTHON_AVAILABLE" ]; then
    echo "   Installing Python runtime..."
    docker exec kodla_piston piston ppman install python || echo "   Python might already be installed"
else
    echo "   ✅ Python $PYTHON_AVAILABLE is available"
fi

# Note: Piston's ppman doesn't support pip packages directly
# ML packages need to be installed in a custom Python environment
echo ""
echo "⚠️  Note: Piston uses isolated environments for each execution."
echo "   For ML libraries, you need a custom Python image with pre-installed packages."
echo ""
echo "📝 To add ML libraries, create a custom Piston package or use:"
echo "   1. Build custom Python runtime with ML libs"
echo "   2. Or use Docker volumes to mount pre-installed packages"
echo ""

# Check if Python is working
echo "🧪 Testing Python execution..."
RESULT=$(curl -sf -X POST "$PISTON_URL/api/v2/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "language": "python",
    "version": "*",
    "files": [{"content": "print(\"Hello from Python!\")"}]
  }')

if echo "$RESULT" | jq -e '.run.stdout' > /dev/null 2>&1; then
    echo "   ✅ Python is working!"
    echo "$RESULT" | jq -r '.run.stdout'
else
    echo "   ❌ Python execution failed"
    echo "$RESULT" | jq .
fi

echo ""
echo "✅ Piston setup complete!"
