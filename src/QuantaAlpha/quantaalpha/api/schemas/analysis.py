from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AnalysisRunRequest(BaseModel):
    library_suffix: str | None = None
    quality_filter: str = "high+medium"  # high, medium, high+medium, all
    factor_ids: list[str] | None = None
    config_path: str | None = None


class AnalysisFactorResult(BaseModel):
    factor_id: str
    factor_name: str
    quality: str = "low"
    performance: dict[str, Any] = {}
    group_returns: dict[str, float] = {}
    style_exposures: dict[str, Any] = {}
    plot_url: str | None = None


class AnalysisTaskStatus(BaseModel):
    task_id: str
    status: str
    progress: dict[str, Any] = {}
    errors: list[dict[str, str]] = []


class AnalysisResults(BaseModel):
    task_id: str
    factors: list[AnalysisFactorResult] = []
