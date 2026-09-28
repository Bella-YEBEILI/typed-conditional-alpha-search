from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]
    debug: bool = False

    project_root: Path = Path(__file__).resolve().parents[2]
    static_results_dir: str = "data/results"

    model_config = {"env_prefix": "QUANTAALPHA_API_"}


@lru_cache
def get_settings() -> APISettings:
    return APISettings()
