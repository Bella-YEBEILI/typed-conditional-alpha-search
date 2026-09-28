from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ProfileInfo(BaseModel):
    id: str
    label: str


class BacktestConfigResponse(BaseModel):
    profiles: list[ProfileInfo] = []
    universes: list[str] = []
    default_cost: float = 0.0012
    default_trading_days: int = 243
    default_date_range: dict[str, str] = {"start": "2016-01-01", "end": "2025-12-31"}


class FactorSelection(BaseModel):
    source: str  # library, expression
    factor_id: str | None = None
    library_suffix: str | None = None
    expression: str | None = None
    factor_name: str | None = None


class BacktestParams(BaseModel):
    profile: str = "stables_o1_o2"
    universe: str = "standards"
    cost: float = 0.0012
    start_date: str = "2016-01-01"
    end_date: str = "2025-12-31"


class BacktestRunRequest(BaseModel):
    factors: list[FactorSelection]
    config: BacktestParams = BacktestParams()


class PerformanceMetrics(BaseModel):
    long_ret: float | None = None
    long_ir: float | None = None
    long_maxdd: float | None = None
    long_netret: float | None = None
    long_netir: float | None = None
    long_netmaxdd: float | None = None
    ls_ret: float | None = None
    ls_ir: float | None = None
    ls_maxdd: float | None = None
    ls_netret: float | None = None
    ls_netir: float | None = None
    ls_netmaxdd: float | None = None
    rankic: float | None = None
    rankicir: float | None = None
    long_turnover: float | None = None
    ls_turnover: float | None = None
    long_num: float | None = None
    coverage: float | None = None


class BacktestFactorResult(BaseModel):
    factor_name: str
    factor_expression: str = ""
    performance: PerformanceMetrics = PerformanceMetrics()
    group_returns: dict[str, float] = {}
    plot_url: str | None = None


class BacktestTaskStatus(BaseModel):
    task_id: str
    status: str
    progress: dict[str, Any] = {}


class BacktestResults(BaseModel):
    task_id: str
    factors: list[BacktestFactorResult] = []
