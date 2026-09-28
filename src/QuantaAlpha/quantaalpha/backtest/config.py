from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from quantaalpha.runtime import (
    data_root,
    env_str,
    expand_env_placeholders,
    tq_candidate_dir,
    tq_factor_base_dir,
    tq_upstream_config_path,
)

TQ_RUNNER_CLASS_PATH = "quantaalpha.factors.tq_runner.TQFactorRunner"

DEFAULT_CONFIG_PATH = tq_upstream_config_path()

DEFAULT_EXECUTION_CONSTRAINTS: dict[str, bool] = {
    "use_tradables": True,
    "use_limit_masks": True,
}

DEFAULT_BACKTEST_CONFIG: dict[str, Any] = {
    "profile_id": "stables_o1_o2",
    "profile_mode": "o1_o2",
    "params": {
        "start": None,
        "end": None,
        "cost": 0.0012,
        "trading_days": 243,
    },
    "factor_value_window": {
        "start": "2016-01-01",
        "end": "2025-12-31",
    },
    "spec_defaults": {
        "author": "quantaalpha",
        "level": "days",
        "tag": "qa_upstream",
        "category": "unknown",
        "universe": "standards",
        "pasteurization": True,
    },
    "transform_spec": {
        "subuniverse": [],
        "neutralize": [],
    },
    "metric_mode": "long_only",
    "use_net_metrics": True,
    "execution_constraints": dict(DEFAULT_EXECUTION_CONSTRAINTS),
    "submit_after_check": False,
    "mining": {
        "execution_constraints": dict(DEFAULT_EXECUTION_CONSTRAINTS),
        "discovery_split": "train",
        "train_splits": ["train"],
        "primary_split": "train",
        "test_split": "test",
        "periods": {
            "train": {
                "start": "2016-01-01",
                "end": "2021-12-31",
            },
            "test": {
                "start": "2022-01-01",
                "end": "2025-12-31",
            },
        },
    },
    "standalone": {
        "execution_constraints": dict(DEFAULT_EXECUTION_CONSTRAINTS),
        "params": {
            "start": "2016-01-01",
            "end": "2025-12-31",
        },
    },
    "tq": {
        "factor_base_dir": str(tq_factor_base_dir()),
        "data_dir": str(data_root()),
        "data_start_date": env_str("QUANTAALPHA_DATA_START_DATE", "20150101"),
    },
    "output": {
        "candidate_dir": str(tq_candidate_dir()),
    },
}

REQUIRED_BACKTEST_FIELDS = frozenset(
    {
        "opens",
        "closes",
        "adj_factors",
        "standards",
    }
)

EXECUTION_CONSTRAINT_FIELD_REQUIREMENTS = {
    "use_tradables": {"tradables"},
    "use_limit_masks": {"limit_up_cto", "limit_down_cto", "limit_up_ctc", "limit_down_ctc"},
}

TRANSFORM_FIELD_REQUIREMENTS = {
    "subuniverse": {
        "hs300s": {"hs300s"},
        "zz1000s": {"zz1000s"},
    },
    "neutralize": {
        "industry": {"industrys"},
        "size": {"Size", "Nlsize"},
        "styles": {"Beta", "Liq", "Mom", "Nlsize", "Rev", "Size", "Vol"},
        "complete": {"industrys", "Beta", "Liq", "Mom", "Nlsize", "Rev", "Size", "Vol"},
    },
}

PROFILE_MODE_TO_ID = {
    "o1_o2": "stables_o1_o2",
    "c1_c2": "stables_c1_c2",
    "stables_o1_o2": "stables_o1_o2",
    "stables_c1_c2": "stables_c1_c2",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_config_path(config_path: str | Path | None) -> Path:
    if config_path is None:
        env_default = env_str("TQ_UPSTREAM_CONFIG_PATH")
        if env_default:
            env_path = Path(env_default)
            if env_path.is_absolute():
                return env_path
            return Path(__file__).resolve().parents[2] / env_path
        return DEFAULT_CONFIG_PATH
    path = Path(config_path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    tq_cfg = config.setdefault("tq", {})
    output_cfg = config.setdefault("output", {})
    spec_defaults = config.setdefault("spec_defaults", {})

    tq_cfg["factor_base_dir"] = str(tq_factor_base_dir())
    tq_cfg["data_dir"] = str(data_root())
    if not str(spec_defaults.get("universe") or "").strip():
        spec_defaults["universe"] = DEFAULT_BACKTEST_CONFIG["spec_defaults"]["universe"]
    tq_cfg["data_start_date"] = env_str(
        "QUANTAALPHA_DATA_START_DATE",
        str(tq_cfg.get("data_start_date") or "20150101"),
    )
    output_cfg["candidate_dir"] = str(tq_candidate_dir())
    return config


def normalize_profile_id(profile_id: str | None = None, profile_mode: str | None = None) -> tuple[str, str]:
    raw_profile_id = str(profile_id or "").strip().lower()
    raw_profile_mode = str(profile_mode or "").strip().lower()

    if raw_profile_id in PROFILE_MODE_TO_ID:
        normalized_id = PROFILE_MODE_TO_ID[raw_profile_id]
    elif raw_profile_id:
        normalized_id = raw_profile_id
    else:
        normalized_id = PROFILE_MODE_TO_ID.get(raw_profile_mode, DEFAULT_BACKTEST_CONFIG["profile_id"])

    normalized_mode = "c1_c2" if normalized_id == "stables_c1_c2" else "o1_o2"
    return normalized_id, normalized_mode


def validate_backtest_config(config: dict[str, Any]) -> None:
    profile_id = str(config.get("profile_id") or "").strip()
    if not profile_id:
        raise ValueError("tq upstream config requires a non-empty profile_id")

    params = config.get("params") or {}
    for key in ("cost", "trading_days"):
        if key not in params:
            raise ValueError(f"tq upstream config params missing required key: {key}")

    spec_defaults = config.get("spec_defaults") or {}
    if not str(spec_defaults.get("universe") or "").strip():
        raise ValueError("tq upstream config requires spec_defaults.universe")

    metric_mode = str(config.get("metric_mode") or "long_only").strip().lower()
    if metric_mode not in {"long_only", "long_short"}:
        raise ValueError("metric_mode must be one of: long_only, long_short")

    output_cfg = config.get("output") or {}
    if not str(output_cfg.get("candidate_dir") or "").strip():
        raise ValueError("tq upstream config requires output.candidate_dir")

    tq_cfg = config.get("tq") or {}
    data_dir = tq_cfg.get("data_dir")
    if data_dir in (None, ""):
        return
    resolved = Path(data_dir)
    if not resolved.is_absolute():
        resolved = Path(__file__).resolve().parents[2] / resolved
    if not resolved.exists():
        raise ValueError(f"tq upstream config data_dir does not exist: {resolved}")

    mining_cfg = config.get("mining") or {}
    periods = mining_cfg.get("periods") or {}
    if not periods:
        raise ValueError("tq upstream config requires mining.periods")
    primary_split = str(mining_cfg.get("primary_split") or "train").strip() or "train"
    if primary_split not in periods:
        raise ValueError(f"tq upstream config primary_split not found in mining.periods: {primary_split}")
    discovery_split = str(mining_cfg.get("discovery_split") or "").strip()
    if discovery_split and discovery_split not in periods:
        raise ValueError(f"tq upstream config discovery_split not found in mining.periods: {discovery_split}")
    train_splits = mining_cfg.get("train_splits") or []
    if train_splits is not None and not isinstance(train_splits, list):
        raise ValueError("tq upstream config mining.train_splits must be a list")
    for split_name in train_splits:
        split_text = str(split_name or "").strip()
        if not split_text:
            raise ValueError("tq upstream config mining.train_splits contains an empty split name")
        if split_text not in periods:
            raise ValueError(f"tq upstream config train split not found in mining.periods: {split_text}")
    test_split = str(mining_cfg.get("test_split") or "").strip()
    if test_split and test_split not in periods:
        raise ValueError(f"tq upstream config test_split not found in mining.periods: {test_split}")


def load_backtest_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = _resolve_config_path(config_path)
    config = copy.deepcopy(DEFAULT_BACKTEST_CONFIG)
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            loaded = expand_env_placeholders(yaml.safe_load(fh) or {})
        if not isinstance(loaded, dict):
            raise ValueError("tq upstream config must be a mapping")
        config = _deep_merge(config, loaded)
    config = _apply_env_overrides(config)
    profile_id, profile_mode = normalize_profile_id(
        profile_id=config.get("profile_id"),
        profile_mode=config.get("profile_mode"),
    )
    config["profile_id"] = profile_id
    config["profile_mode"] = profile_mode
    validate_backtest_config(config)
    config["_config_path"] = str(path)
    return config


DEFAULT_TQ_UPSTREAM_CONFIG = DEFAULT_BACKTEST_CONFIG
validate_tq_upstream_config = validate_backtest_config
load_tq_upstream_config = load_backtest_config


def resolve_backtest_engine(backtest_cfg: dict[str, Any] | None, current_env: dict[str, str] | None = None) -> str:
    del backtest_cfg, current_env
    return "tq"


def configure_backtest_engine(engine: str | None = None, alignment_config_path: str | None = None) -> None:
    del engine
    from quantaalpha.pipeline.settings import ALPHA_AGENT_FACTOR_PROP_SETTING, FACTOR_BACK_TEST_PROP_SETTING

    if alignment_config_path:
        os.environ["TQ_UPSTREAM_CONFIG_PATH"] = alignment_config_path
        os.environ["QUANTAALPHA_BACKTEST_CONFIG_PATH"] = alignment_config_path
    ALPHA_AGENT_FACTOR_PROP_SETTING.runner = TQ_RUNNER_CLASS_PATH
    FACTOR_BACK_TEST_PROP_SETTING.runner = TQ_RUNNER_CLASS_PATH
    os.environ["QUANTAALPHA_BACKTEST_ENGINE"] = "tq"


