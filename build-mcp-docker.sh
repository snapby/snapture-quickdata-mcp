#!/bin/bash
# Build script for MCP server Docker image

set -euo pipefail

# Configuration
IMAGE_NAME="snapture-quickdata-mcp"
TAG="${1:-latest}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo "🐳 Building MCP server Docker image: ${FULL_IMAGE_NAME}"

# Build the image
docker build -t "${FULL_IMAGE_NAME}" .

echo "✅ Successfully built Docker image: ${FULL_IMAGE_NAME}"

# Show image info
echo ""
echo "📊 Image information:"
docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedSince}}"

echo ""
echo "🚀 To run the MCP server:"
echo "docker run -p 3000:3000 --env-file .env --name snapture-quickdata-mcp-container ${FULL_IMAGE_NAME}"
echo ""
echo "🔍 To run with custom environment variables:"
echo "docker run -p 8080:8080 -e PORT=8080 -e WORKERS=4 ${FULL_IMAGE_NAME}"
echo ""
