"""Manual quality-analysis helpers for saved TQ backtest outputs."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import fire
import pandas as pd

from quantaalpha.backtest import FactorPlotter, FactorQualityAnalyzer, TQUpstreamBridge


def _load_factor_result(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        raise ValueError("factor_result_file must be the raw *_factor_result.pkl artifact, not the JSON export.")
    with path.open("rb") as fh:
        return pickle.load(fh)


def main(
    factor_value_file: str,
    factor_name: str | None = None,
    factor_result_file: str | None = None,
    config_path: str | None = None,
    output_dir: str | None = None,
    universe_name: str = "standards",
    plot: bool = False,
    show: bool = False,
):
    """Build a manual quality report from saved backtest artifacts."""
    factor_value_path = Path(factor_value_file)
    if not factor_value_path.exists():
        raise FileNotFoundError(f"factor_value_file does not exist: {factor_value_path}")

    factor_name = factor_name or factor_value_path.stem.replace("_factor_value", "")
    bridge = TQUpstreamBridge(config_path)
    out_dir = Path(output_dir) if output_dir else bridge.candidate_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    factor_value = pd.read_pickle(factor_value_path)
    factor_result = {}
    if factor_result_file:
        result_path = Path(factor_result_file)
        if not result_path.exists():
            raise FileNotFoundError(f"factor_result_file does not exist: {result_path}")
        factor_result = _load_factor_result(result_path)

    analyzer = FactorQualityAnalyzer(bridge._get_runtime_data_provider(require_data=True))
    report = analyzer.build_report(
        factor_name=factor_name,
        factor_value=factor_value,
        factor_result=factor_result,
        universe_name=universe_name,
    )
    quality_path = out_dir / f"{factor_name}_quality.json"
    quality_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_path = None
    if plot:
        if not factor_result:
            raise ValueError("plot=true requires factor_result_file with raw pickle output.")
        plot_path = out_dir / f"{factor_name}_diagnostics.png"
        plotter = FactorPlotter()
        plotter.plot_result(
            factor_name,
            factor_result,
            params=dict(bridge.config.get("params") or {}),
            output_path=plot_path,
            show=show,
        )

    payload = {
        "factor_name": factor_name,
        "quality_path": str(quality_path),
    }
    if plot_path is not None:
        payload["plot_path"] = str(plot_path)
    if str(os.environ.get("QUANTAALPHA_QUIET_CONSOLE", "")).strip().lower() not in {"1", "true", "yes", "y", "on"}:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    fire.Fire(main)
