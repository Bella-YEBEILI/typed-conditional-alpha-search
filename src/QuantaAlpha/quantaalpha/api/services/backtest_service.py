from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from quantaalpha.runtime import (
    project_root,
    results_root,
    factor_library_path,
    tq_upstream_config_path,
)

logger = logging.getLogger(__name__)


class BacktestService:
    def get_config(self) -> dict[str, Any]:
        return {
            "profiles": [
                {"id": "stables_o1_o2", "label": "开盘价 O1-O2"},
                {"id": "stables_c1_c2", "label": "收盘价 C1-C2"},
            ],
            "universes": ["standards", "hs300s", "zz1000s"],
            "default_cost": 0.0012,
            "default_trading_days": 243,
            "default_date_range": {
                "start": "2016-01-01",
                "end": "2025-12-31",
            },
        }

    def run_backtest_task(
        self,
        factors: list[dict[str, Any]],
        config: dict[str, Any],
        task_id: str,
        progress_callback: Any = None,
    ) -> list[dict[str, Any]]:
        output_dir = results_root() / f"backtest_{task_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        total = len(factors)

        for i, factor_spec in enumerate(factors):
            factor_name = factor_spec.get("factor_name") or f"factor_{i}"
            source = factor_spec.get("source", "expression")

            try:
                if source == "expression":
                    result = self._backtest_expression(
                        expression=factor_spec["expression"],
                        factor_name=factor_name,
                        config=config,
                        output_dir=output_dir,
                    )
                elif source == "library":
                    result = self._backtest_from_library(
                        factor_id=factor_spec["factor_id"],
                        library_suffix=factor_spec.get("library_suffix"),
                        factor_name=factor_name,
                        config=config,
                        output_dir=output_dir,
                    )
                else:
                    result = {"factor_name": factor_name, "error": f"未知来源: {source}"}

                results.append(result)
                logger.info(f"回测完成 [{i+1}/{total}]: {factor_name}")

            except Exception as e:
                logger.error(f"回测失败 [{i+1}/{total}] {factor_name}: {e}")
                results.append({
                    "factor_name": factor_name,
                    "factor_expression": factor_spec.get("expression", ""),
                    "performance": {},
                    "group_returns": {},
                    "plot_url": None,
                    "error": str(e),
                })

        return results

    def _backtest_expression(
        self,
        expression: str,
        factor_name: str,
        config: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        from quantaalpha.backtest.run_backtest import run_single_factor_backtest

        factor_dir = output_dir / factor_name
        factor_dir.mkdir(parents=True, exist_ok=True)

        result = run_single_factor_backtest(
            expression=expression,
            factor_name=factor_name,
            config_path=str(tq_upstream_config_path()),
            output_dir=str(factor_dir),
            plot=True,
            show=False,
            quality_report=True,
            notebook_report=False,
        )

        return self._parse_backtest_result(
            factor_name=factor_name,
            factor_expression=expression,
            result_paths=result,
            factor_dir=factor_dir,
        )

    def _backtest_from_library(
        self,
        factor_id: str,
        library_suffix: str | None,
        factor_name: str,
        config: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        lib_path = factor_library_path(library_suffix or "")
        if not lib_path.exists():
            raise FileNotFoundError(f"因子库不存在: {lib_path}")

        lib_data = json.loads(lib_path.read_text(encoding="utf-8"))
        factor_data = lib_data.get("factors", {}).get(factor_id)
        if not factor_data:
            raise KeyError(f"因子不存在: {factor_id}")

        expression = factor_data.get("factor_expression", "")
        if not expression:
            raise ValueError(f"因子 {factor_id} 没有表达式")

        return self._backtest_expression(
            expression=expression,
            factor_name=factor_data.get("factor_name", factor_name),
            config=config,
            output_dir=output_dir,
        )

    def _parse_backtest_result(
        self,
        factor_name: str,
        factor_expression: str,
        result_paths: dict[str, str | None],
        factor_dir: Path,
    ) -> dict[str, Any]:
        performance = {}
        group_returns = {}
        plot_url = None

        quality_path = result_paths.get("quality_report_path")
        if quality_path and Path(quality_path).exists():
            try:
                report = json.loads(Path(quality_path).read_text(encoding="utf-8"))
                performance = report.get("result_summary") or report.get("performance") or {}
                group_summary = report.get("group_summary") or {}
                for gname, gdata in group_summary.items():
                    if isinstance(gdata, dict):
                        group_returns[gname] = gdata.get("final_cum_return", 0)
                    else:
                        group_returns[gname] = gdata
            except Exception as e:
                logger.warning(f"解析质量报告失败: {e}")

        plot_path = result_paths.get("plot_path")
        if plot_path and Path(plot_path).exists():
            rel = Path(plot_path).relative_to(results_root())
            plot_url = f"/static/results/{rel.as_posix()}"

        return {
            "factor_name": factor_name,
            "factor_expression": factor_expression,
            "performance": performance,
            "group_returns": group_returns,
            "plot_url": plot_url,
        }
