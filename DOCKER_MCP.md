# 🐳 Docker MCP Server

This document explains how to build and run the QuickDataMCP server using Docker.

## 📦 Dockerfile Overview

The `Dockerfile` creates a containerized version of the MCP server with the following features:

- **Multi-stage build** for optimized image size
- **Python 3.13** runtime with UV package manager
- **Non-root user** for enhanced security
- **Configurable entrypoint** for flexible runtime settings
- **Port 3000** exposed by default for HTTP transport

### Build Architecture

```dockerfile
Stage 1 (builder): ghcr.io/astral-sh/uv:python3.13-bookworm-slim
├── Install dependencies with UV
├── Copy source code
└── Build application

Stage 2 (runtime): python:3.13-slim-bookworm
├── Create non-root user (app:app)
├── Copy built application from builder
├── Set up entrypoint script for runtime configuration
└── Expose default port
```

## 🚀 Quick Start

### 1. Build the Docker Image

```bash
# Using the build script
./build-mcp-docker.sh

# Or manually
docker build -t snapture-quickdata-mcp:latest .
```

### 2. Run the MCP Server

```bash
# With your .env file
docker run -p 3000:3000 --env-file .env --name snapture-quickdata-mcp-container snapture-quickdata-mcp

# Override environment variables for custom settings
docker run -p 8080:8080 \
  -e PORT=8080 \
  -e WORKERS=4 \
  --name snapture-quickdata-mcp-container-custom \
  snapture-quickdata-mcp
```

## ⚙️ Configuration

The server is configured via environment variables, which are handled by the `entrypoint.sh` script.

### Environment Variables

| Variable  | Description                  | Default   |
|-----------|------------------------------|-----------|
| `HOST`    | The host address to bind to. | `0.0.0.0` |
| `PORT`    | The port to run the server on. | `3000`    |
| `WORKERS` | The number of uvicorn workers. | `1`       |

## 🔧 Advanced Usage

### Custom Port and Workers
```bash
docker run -p 8080:8080 \
  -e PORT=8080 \
  -e WORKERS=4 \
  snapture-quickdata-mcp
```

### Volume Mounting for Data
```bash
# Mount local data directory
docker run -p 3000:3000 --env-file .env \
  -v ./data:/app/data \
  snapture-quickdata-mcp
```

## 🏥 Monitoring

Monitor the container and MCP server status:

```bash
# Check container status
docker ps

# Check if MCP server process is running
docker exec <container_id> pgrep -f "uvicorn"

# View container logs
docker logs <container_id>

# Follow logs in real-time
docker logs -f <container_id>
```

## 🔍 Troubleshooting

### Container Won't Start
```bash
# Check logs
docker logs <container_id>

# Run interactively for debugging
docker run -it --entrypoint bash snapture-quickdata-mcp
```

### Port Already in Use
```bash
# Use a different host port
docker run -p 3001:3000 --env-file .env snapture-quickdata-mcp
```

## 🚢 Deployment

### Docker Compose (Recommended)

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  mcp-server:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    env_file:
      - .env
    restart: unless-stopped
    volumes:
      - ./data:/app/data  # Optional: for persistent data
```

Run with:
```bash
docker-compose up -d
```

## 📊 Container Specifications

| Aspect            | Details                               |
|-------------------|---------------------------------------|
| Base Image        | `python:3.13-slim-bookworm`           |
| Package Manager   | UV (ultra-fast Python package manager)|
| User              | Non-root (`app:app`, UID/GID 1000)    |
| Working Directory | `/app`                                |
| Default Port      | `3000` (exposed)                      |
| Entry Point       | `/app/entrypoint.sh`                  |
| Default Command   | Uvicorn server on HTTP transport      |

This Docker setup provides a robust, secure, and efficient way to deploy the QuickDataMCP server in any containerized environment.
