from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class QualityFilter(BaseModel):
    quality: str = "all"  # all, high, medium, low


class KeyMetrics(BaseModel):
    long_ret: float | None = None
    long_netret: float | None = None
    ls_ir: float | None = None
    ls_netret: float | None = None
    rankic: float | None = None
    rankicir: float | None = None
    long_turnover: float | None = None
    coverage: float | None = None


class CheckFlags(BaseModel):
    check_passed: bool = False
    raw_passed: bool = False
    zz1000s_passed: bool = False
    complete_passed: bool = False
    check_tag: str = ""


class FactorListItem(BaseModel):
    factor_id: str
    factor_name: str
    factor_expression: str = ""
    factor_level: str = "days"
    quality: str = "low"
    check_passed: bool = False
    train_passed: bool = False
    test_passed: bool = False
    key_metrics: KeyMetrics = KeyMetrics()
    added_at: str | None = None


class FactorDetail(BaseModel):
    factor_id: str
    factor_name: str
    factor_expression: str = ""
    factor_description: str = ""
    factor_formulation: str = ""
    factor_level: str = "days"
    quality: str = "low"
    check_passed: bool = False
    train_passed: bool = False
    test_passed: bool = False
    train_check_flags: CheckFlags = CheckFlags()
    test_check_flags: CheckFlags = CheckFlags()
    test_backtest_metrics: dict[str, Any] = {}
    style_exposures: dict[str, Any] | None = None
    metadata: dict[str, Any] = {}
    feedback: dict[str, Any] = {}
    run_context: dict[str, Any] = {}


class QualityCounts(BaseModel):
    high: int = 0
    medium: int = 0
    low: int = 0


class FactorListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    quality_counts: QualityCounts = QualityCounts()
    factors: list[FactorListItem] = []


class FactorUpdateRequest(BaseModel):
    factor_expression: str
    factor_description: str | None = None
    factor_formulation: str | None = None
