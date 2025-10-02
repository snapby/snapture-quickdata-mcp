"""Server configuration resource implementation."""

from ..config.settings import settings
from typing import Dict, Any, Optional


async def get_server_config() -> dict:
    """Get server configuration."""
    return settings.model_dump()
