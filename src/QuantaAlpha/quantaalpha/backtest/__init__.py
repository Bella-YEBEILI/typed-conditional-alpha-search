"""QuantaAlpha 内置的独立 TQ 风格回测工具集。"""

from __future__ import annotations

__all__ = [
    "FactorPlotter",
    "FactorQualityAnalyzer",
    "FilesystemDataProvider",
    "REQUIRED_BACKTEST_FIELDS",
    "StaticFieldDataProvider",
    "TQUpstreamBridge",
    "TQ_RUNNER_CLASS_PATH",
    "TRANSFORM_FIELD_REQUIREMENTS",
    "configure_backtest_engine",
    "load_backtest_config",
    "load_tq_upstream_config",
    "normalize_profile_id",
    "resolve_backtest_engine",
    "summarize_tq_check_result",
]


def __getattr__(name: str):
    if name in {"TQUpstreamBridge", "summarize_tq_check_result"}:
        from .bridge import TQUpstreamBridge, summarize_tq_check_result

        return {
            "TQUpstreamBridge": TQUpstreamBridge,
            "summarize_tq_check_result": summarize_tq_check_result,
        }[name]

    if name in {
        "REQUIRED_BACKTEST_FIELDS",
        "TRANSFORM_FIELD_REQUIREMENTS",
        "TQ_RUNNER_CLASS_PATH",
        "configure_backtest_engine",
        "load_backtest_config",
        "load_tq_upstream_config",
        "normalize_profile_id",
        "resolve_backtest_engine",
    }:
        from .config import (
            REQUIRED_BACKTEST_FIELDS,
            TRANSFORM_FIELD_REQUIREMENTS,
            TQ_RUNNER_CLASS_PATH,
            configure_backtest_engine,
            load_backtest_config,
            load_tq_upstream_config,
            normalize_profile_id,
            resolve_backtest_engine,
        )

        return {
            "REQUIRED_BACKTEST_FIELDS": REQUIRED_BACKTEST_FIELDS,
            "TRANSFORM_FIELD_REQUIREMENTS": TRANSFORM_FIELD_REQUIREMENTS,
            "TQ_RUNNER_CLASS_PATH": TQ_RUNNER_CLASS_PATH,
            "configure_backtest_engine": configure_backtest_engine,
            "load_backtest_config": load_backtest_config,
            "load_tq_upstream_config": load_tq_upstream_config,
            "normalize_profile_id": normalize_profile_id,
            "resolve_backtest_engine": resolve_backtest_engine,
        }[name]

    if name in {"FilesystemDataProvider", "StaticFieldDataProvider"}:
        from .provider import FilesystemDataProvider, StaticFieldDataProvider

        return {
            "FilesystemDataProvider": FilesystemDataProvider,
            "StaticFieldDataProvider": StaticFieldDataProvider,
        }[name]

    if name in {"FactorPlotter", "FactorQualityAnalyzer"}:
        from .quality import FactorPlotter, FactorQualityAnalyzer

        return {
            "FactorPlotter": FactorPlotter,
            "FactorQualityAnalyzer": FactorQualityAnalyzer,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
