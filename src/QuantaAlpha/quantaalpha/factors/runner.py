from __future__ import annotations

import os
from pathlib import Path

from quantaalpha.factors.data_domains import normalize_factor_mode


def resolve_backtest_mode(backtest_mode_env: str | None = None) -> str:
    mode = (backtest_mode_env or os.getenv("FACTOR_BACKTEST_MODE", "single")).strip().lower()
    return "hypothesis_combined" if mode in ["combined", "hypothesis", "hypothesis_combined"] else "single"


def resolve_factor_data_source(data_source: Path, data_mode: str) -> Path:
    del data_source
    mode = normalize_factor_mode(data_mode) or "daily"
    raise RuntimeError(
        f"legacy H5 factor data source is deprecated for mode={mode}; "
        "use the TQ upstream runtime/provider path instead of daily_pv.h5 or minute_pv.h5"
    )
