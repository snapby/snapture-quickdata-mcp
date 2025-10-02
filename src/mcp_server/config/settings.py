from pydantic_settings import BaseSettings, SettingsConfigDict

from typing import List, Dict, Any

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

    server_name: str = "QuickDataMCP"
    version: str = "0.1.0"
    HOST: str = "0.0.0.0"
    PORT: int = 3000
    WORKERS: int = 1
    author: str = "QuickData AI"
    log_level: str = "INFO"
    description: str = "An MCP server for advanced data analytics."
    analytics_features: List[str] = [
        "dataset_loading",
        "schema_discovery",
        "correlation_analysis",
        "segmentation",
        "data_quality_assessment",
        "visualization",
        "outlier_detection",
        "time_series_analysis"
    ]
    supported_formats: List[str] = ["CSV", "JSON"]
    memory_storage: str = "in_memory_dataframes"

settings = Settings()
