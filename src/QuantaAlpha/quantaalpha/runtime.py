from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
_ENV_LOADED = False
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def load_project_env() -> Path:
    """Load the project .env once and return its path."""
    global _ENV_LOADED
    if not _ENV_LOADED and ENV_PATH.exists():
        load_dotenv(ENV_PATH)
        _ENV_LOADED = True
    return ENV_PATH


def project_root() -> Path:
    load_project_env()
    return PROJECT_ROOT


def env_str(name: str, default: str | None = None) -> str | None:
    """Read a string environment variable after loading project env."""
    load_project_env()
    value = os.getenv(name)
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def env_path(name: str, default: str | Path | None = None) -> Path:
    """Read a path environment variable and resolve relative paths under the project root."""
    load_project_env()
    raw = env_str(name)
    value = Path(raw) if raw is not None else Path(default) if default is not None else PROJECT_ROOT
    value = value.expanduser()
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve(strict=False)


def expand_env_placeholders(value: Any) -> Any:
    """Recursively expand ${ENV_NAME} placeholders inside config values."""
    load_project_env()
    if isinstance(value, dict):
        return {key: expand_env_placeholders(val) for key, val in value.items()}
    if isinstance(value, list):
        return [expand_env_placeholders(item) for item in value]
    if not isinstance(value, str):
        return value

    def _replace(match: re.Match[str]) -> str:
        return os.getenv(match.group(1), "")

    return _ENV_VAR_PATTERN.sub(_replace, value)


def experiment_config_path() -> Path:
    return env_path("QUANTAALPHA_EXPERIMENT_CONFIG", Path("configs") / "experiment.yaml")


def tq_upstream_config_path() -> Path:
    return env_path("QUANTAALPHA_TQ_CONFIG", Path("configs") / "tq_upstream_alignment.yaml")


def results_root() -> Path:
    return env_path("DATA_RESULTS_DIR", Path("data") / "results")


def runtime_root() -> Path:
    return env_path("QUANTAALPHA_RUNTIME_DIR", Path("data") / "runtime")


def experiment_id(default: str | None = None) -> str | None:
    return env_str("EXPERIMENT_ID", default)


def experiment_results_dir() -> Path:
    current_experiment_id = str(experiment_id("") or "").strip()
    root = results_root()
    if current_experiment_id and current_experiment_id != "shared":
        return root / current_experiment_id
    return root


def experiment_runtime_dir() -> Path:
    current_experiment_id = str(experiment_id("") or "").strip()
    root = runtime_root()
    if current_experiment_id and current_experiment_id != "shared":
        return root / current_experiment_id
    return root


def market_universe() -> str:
    return str(env_str("QUANTAALPHA_MARKET", "standards") or "standards").strip() or "standards"


def data_root() -> Path:
    legacy = env_str("QUANTAALPHA_SOURCE_DATA_DIR")
    if legacy:
        legacy_path = Path(legacy).expanduser()
        if legacy_path.is_absolute():
            return legacy_path.resolve(strict=False)
        return (PROJECT_ROOT / legacy_path).resolve(strict=False)
    return env_path("QUANTAALPHA_DATA_ROOT", "/home/workspace/common/test_data")


def pv_dir() -> Path:
    return env_path("QUANTAALPHA_PV_DIR", data_root() / "pv")


def fundamental_dir() -> Path:
    return env_path("QUANTAALPHA_FUNDAMENTAL_DIR", data_root() / "fundamental")


def universe_dir() -> Path:
    return env_path("QUANTAALPHA_UNIVERSE_DIR", data_root() / "universe")


def status_dir() -> Path:
    return env_path("QUANTAALPHA_STATUS_DIR", data_root() / "status")


def limit_dir() -> Path:
    return env_path("QUANTAALPHA_LIMIT_DIR", data_root() / "limit")


def industry_dir() -> Path:
    return env_path("QUANTAALPHA_INDUSTRY_DIR", data_root() / "industry")


def style_dir() -> Path:
    return env_path("QUANTAALPHA_STYLE_DIR", data_root() / "styles")


def style_derived_dir() -> Path:
    return env_path("QUANTAALPHA_STYLE_DERIVED_DIR", style_dir() / "derived")


def minute_data_root() -> Path:
    return env_path("QUANTAALPHA_MINUTE_DATA_ROOT", data_root())


def minute_vwap_dir() -> Path:
    return env_path("QUANTAALPHA_MINUTE_VWAP_DIR", minute_data_root() / "minute_vwaps")


def minute_h5_path() -> Path:
    return env_path("QUANTAALPHA_MINUTE_H5_PATH", minute_data_root() / "all_minute_data.h5")


def factor_library_dir() -> Path:
    return env_path("FACTOR_LIBRARY_DIR", Path("data") / "factorlib")


def factor_library_default_suffix() -> str | None:
    explicit_suffix = env_str("FACTOR_LIBRARY_SUFFIX")
    if explicit_suffix:
        return explicit_suffix
    experiment_id = env_str("EXPERIMENT_ID")
    if experiment_id and experiment_id != "shared":
        return experiment_id
    return None


def factor_library_path(suffix: str | None = None) -> Path:
    clean_suffix = (suffix or factor_library_default_suffix() or "").strip()
    filename = f"all_factors_library_{clean_suffix}.json" if clean_suffix else "all_factors_library.json"
    return factor_library_dir() / filename


def tq_candidate_dir() -> Path:
    return env_path("TQ_CANDIDATE_DIR", experiment_runtime_dir() / "tq_candidates")


def tq_factor_base_dir() -> Path:
    return env_path("TQ_FACTOR_BASE_DIR", experiment_runtime_dir() / "factor_base")
