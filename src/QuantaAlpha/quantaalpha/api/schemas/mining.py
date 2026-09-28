from __future__ import annotations

from pydantic import BaseModel


class DirectionPortfolio(BaseModel):
    portfolio_id: int
    description: str


class DirectionsResponse(BaseModel):
    mode: str
    source_file: str
    portfolios: list[DirectionPortfolio] = []


class MiningConfig(BaseModel):
    max_rounds: int = 11
    num_directions: int = 5
    factors_per_hypothesis: int = 3
    mutation_enabled: bool = True
    crossover_enabled: bool = True
    evolution_enabled: bool = True
    parallel_enabled: bool = True


class MiningStartRequest(BaseModel):
    mode: str  # pv, minutes, joint
    directions: list[str] | None = None
    direction_ids: list[int] | None = None  # 选中方向的 portfolio_id
    custom_direction: str | None = None
    config: MiningConfig = MiningConfig()


class MiningProgress(BaseModel):
    current_round: int = 0
    max_rounds: int = 11
    phase: str = "original"
    step: str = "propose"
    factors_mined: int = 0
    high_quality_count: int = 0
    elapsed_seconds: float = 0.0


class MiningTaskInfo(BaseModel):
    task_id: str
    status: str
    mode: str = ""
    started_at: str = ""
    progress: MiningProgress = MiningProgress()
    error: str | None = None
