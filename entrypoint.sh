#!/bin/sh
set -e

# Set default values if environment variables are not provided
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-3000}
WORKERS_COUNT=${WORKERS:-1}

# Start the server with the configured values
exec uvicorn mcp_server.server:app --host ${HOST} --port ${PORT} --workers ${WORKERS_COUNT}
