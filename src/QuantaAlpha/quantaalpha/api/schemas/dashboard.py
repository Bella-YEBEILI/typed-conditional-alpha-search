from __future__ import annotations

from pydantic import BaseModel


class LibraryInfo(BaseModel):
    suffix: str
    path: str
    factor_count: int
    last_updated: str | None = None


class DashboardSummary(BaseModel):
    total_factors: int = 0
    high_quality_count: int = 0
    medium_quality_count: int = 0
    low_quality_count: int = 0
    available_libraries: list[LibraryInfo] = []
    active_mining_tasks: int = 0
