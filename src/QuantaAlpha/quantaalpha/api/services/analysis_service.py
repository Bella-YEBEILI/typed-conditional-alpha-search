from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from quantaalpha.runtime import (
    results_root,
    factor_library_path,
    tq_upstream_config_path,
)
from quantaalpha.api.services.library_service import classify_quality

logger = logging.getLogger(__name__)


class AnalysisService:
    def run_analysis_task(
        self,
        library_suffix: str | None,
        quality_filter: str,
        factor_ids: list[str] | None,
        task_id: str,
    ) -> list[dict[str, Any]]:
        lib_path = factor_library_path(library_suffix or "")
        if not lib_path.exists():
            raise FileNotFoundError(f"因子库不存在: {lib_path}")

        lib_data = json.loads(lib_path.read_text(encoding="utf-8"))
        factors_dict = lib_data.get("factors", {})

        candidates = []
        for fid, fdata in factors_dict.items():
            if factor_ids and fid not in factor_ids:
                continue
            q = classify_quality(fdata)
            if quality_filter == "high" and q != "high":
                continue
            if quality_filter == "medium" and q != "medium":
                continue
            if quality_filter == "high+medium" and q not in ("high", "medium"):
                continue
            candidates.append((fid, fdata, q))

        output_dir = results_root() / f"analysis_{task_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        total = len(candidates)
        logger.info(f"分析任务开始: {total} 个因子待回测")

        for i, (fid, fdata, quality) in enumerate(candidates):
            factor_name = fdata.get("factor_name", fid)
            expression = fdata.get("factor_expression", "")

            if not expression:
                logger.warning(f"跳过无表达式因子: {factor_name}")
                continue

            try:
                result = self._analyze_single_factor(
                    factor_id=fid,
                    factor_name=factor_name,
                    expression=expression,
                    quality=quality,
                    output_dir=output_dir,
                )
                results.append(result)
                logger.info(f"分析完成 [{i+1}/{total}]: {factor_name}")

            except Exception as e:
                logger.error(f"分析失败 [{i+1}/{total}] {factor_name}: {e}")
                results.append({
                    "factor_id": fid,
                    "factor_name": factor_name,
                    "quality": quality,
                    "performance": {},
                    "group_returns": {},
                    "style_exposures": {},
                    "plot_url": None,
                    "error": str(e),
                })

        return results

    def _analyze_single_factor(
        self,
        factor_id: str,
        factor_name: str,
        expression: str,
        quality: str,
        output_dir: Path,
    ) -> dict[str, Any]:
        from quantaalpha.backtest.run_backtest import run_single_factor_backtest

        factor_dir = output_dir / factor_id
        factor_dir.mkdir(parents=True, exist_ok=True)

        result_paths = run_single_factor_backtest(
            expression=expression,
            factor_name=factor_name,
            config_path=str(tq_upstream_config_path()),
            output_dir=str(factor_dir),
            plot=True,
            show=False,
            quality_report=True,
        )

        performance = {}
        group_returns = {}
        style_exposures = {}
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
                style_exposures = report.get("style_exposures") or {}
            except Exception as e:
                logger.warning(f"解析质量报告失败: {e}")

        plot_path = result_paths.get("plot_path")
        if plot_path and Path(plot_path).exists():
            rel = Path(plot_path).relative_to(results_root())
            plot_url = f"/static/results/{rel.as_posix()}"

        return {
            "factor_id": factor_id,
            "factor_name": factor_name,
            "quality": quality,
            "performance": performance,
            "group_returns": group_returns,
            "style_exposures": style_exposures,
            "plot_url": plot_url,
        }
