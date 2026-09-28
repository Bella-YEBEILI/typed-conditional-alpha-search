from __future__ import annotations

import json
import glob as glob_mod
import logging
import pickle
from pathlib import Path
from typing import Any

from quantaalpha.runtime import (
    factor_library_dir,
    factor_library_path,
    results_root,
    tq_upstream_config_path,
)

logger = logging.getLogger(__name__)


def classify_quality(factor: dict) -> str:
    train_flags = factor.get("train_check_flags") or {}
    test_flags = factor.get("test_check_flags") or {}
    train_passed = bool(train_flags.get("check_passed", False))
    test_passed = bool(test_flags.get("check_passed", False))
    if train_passed and test_passed:
        return "high"
    if train_passed or test_passed:
        return "medium"
    return "low"


# 前端挖掘产出的因子库前缀白名单
_FRONTEND_LIBRARY_PREFIXES = ("pv_", "minutes_", "joint_", "fundamental_")


class LibraryService:
    def list_libraries(self, frontend_only: bool = True) -> list[dict[str, Any]]:
        """
        列出���子库。
        frontend_only=True 时只返回前端挖掘产生的因子库（后缀以 pv_/minutes_/joint_ 开头）
        """
        lib_dir = factor_library_dir()
        if not lib_dir.exists():
            return []

        results = []
        pattern = str(lib_dir / "all_factors_library*.json")
        for fp in sorted(glob_mod.glob(pattern)):
            path = Path(fp)
            suffix = ""
            name = path.stem
            prefix = "all_factors_library"
            if name.startswith(prefix + "_"):
                suffix = name[len(prefix) + 1:]
            elif name != prefix:
                continue

            # 过滤：只展示前端挖掘产出的因子库
            if frontend_only:
                if not suffix:
                    continue  # 跳过默认���子库
                if not suffix.startswith(_FRONTEND_LIBRARY_PREFIXES):
                    continue  # 跳过非前端产出的因子库

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                factor_count = len(data.get("factors", {}))
                last_updated = (data.get("metadata") or {}).get("last_updated", "")
            except Exception:
                factor_count = 0
                last_updated = ""

            results.append({
                "suffix": suffix,
                "path": str(path),
                "factor_count": factor_count,
                "last_updated": last_updated,
            })
        return results

    def load_library(self, suffix: str | None = None) -> dict:
        path = factor_library_path(suffix or "")
        if not path.exists():
            return {"metadata": {}, "factors": {}}
        return json.loads(path.read_text(encoding="utf-8"))

    def save_library(self, data: dict, suffix: str | None = None) -> None:
        path = factor_library_path(suffix or "")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_factors(
        self,
        suffix: str | None = None,
        quality: str = "all",
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "added_at",
        sort_order: str = "desc",
        search: str = "",
    ) -> dict[str, Any]:
        data = self.load_library(suffix)
        factors_dict = data.get("factors", {})

        factors_list = []
        quality_counts = {"high": 0, "medium": 0, "low": 0}

        for fid, fdata in factors_dict.items():
            q = classify_quality(fdata)
            quality_counts[q] = quality_counts.get(q, 0) + 1

            if quality != "all" and q != quality:
                continue

            if search:
                name = (fdata.get("factor_name") or "").lower()
                desc = (fdata.get("factor_description") or "").lower()
                expr = (fdata.get("factor_expression") or "").lower()
                needle = search.lower()
                if needle not in name and needle not in desc and needle not in expr:
                    continue

            metrics = fdata.get("test_backtest_metrics") or {}
            train_flags = fdata.get("train_check_flags") or {}
            test_flags = fdata.get("test_check_flags") or {}

            factors_list.append({
                "factor_id": fid,
                "factor_name": fdata.get("factor_name", ""),
                "factor_expression": fdata.get("factor_expression", ""),
                "factor_level": fdata.get("factor_level", "days"),
                "quality": q,
                "check_passed": bool(train_flags.get("check_passed")) and bool(test_flags.get("check_passed")),
                "train_passed": bool(train_flags.get("check_passed")),
                "test_passed": bool(test_flags.get("check_passed")),
                "key_metrics": {
                    "long_ret": metrics.get("long_ret"),
                    "long_netret": metrics.get("long_netret"),
                    "ls_ir": metrics.get("ls_ir"),
                    "ls_netret": metrics.get("ls_netret"),
                    "rankic": metrics.get("rankic"),
                    "rankicir": metrics.get("rankicir"),
                    "long_turnover": metrics.get("long_turnover"),
                    "coverage": metrics.get("coverage"),
                },
                "added_at": fdata.get("added_at", ""),
            })

        reverse = sort_order == "desc"
        factors_list.sort(key=lambda x: x.get(sort_by) or "", reverse=reverse)

        total = len(factors_list)
        start = (page - 1) * page_size
        end = start + page_size
        page_items = factors_list[start:end]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "quality_counts": quality_counts,
            "factors": page_items,
        }

    def get_factor_detail(self, factor_id: str, suffix: str | None = None) -> dict | None:
        data = self.load_library(suffix)
        fdata = data.get("factors", {}).get(factor_id)
        if not fdata:
            return None

        q = classify_quality(fdata)
        train_flags = fdata.get("train_check_flags") or {}
        test_flags = fdata.get("test_check_flags") or {}

        return {
            "factor_id": factor_id,
            "factor_name": fdata.get("factor_name", ""),
            "factor_expression": fdata.get("factor_expression", ""),
            "factor_description": fdata.get("factor_description", ""),
            "factor_formulation": fdata.get("factor_formulation", ""),
            "factor_level": fdata.get("factor_level", "days"),
            "quality": q,
            "check_passed": bool(train_flags.get("check_passed")) and bool(test_flags.get("check_passed")),
            "train_passed": bool(train_flags.get("check_passed")),
            "test_passed": bool(test_flags.get("check_passed")),
            "train_check_flags": {
                "check_passed": bool(train_flags.get("check_passed", False)),
                "raw_passed": bool(train_flags.get("raw_passed", False)),
                "zz1000s_passed": bool(train_flags.get("zz1000s_passed", False)),
                "complete_passed": bool(train_flags.get("complete_passed", False)),
                "check_tag": train_flags.get("check_tag", ""),
            },
            "test_check_flags": {
                "check_passed": bool(test_flags.get("check_passed", False)),
                "raw_passed": bool(test_flags.get("raw_passed", False)),
                "zz1000s_passed": bool(test_flags.get("zz1000s_passed", False)),
                "complete_passed": bool(test_flags.get("complete_passed", False)),
                "check_tag": test_flags.get("check_tag", ""),
            },
            "test_backtest_metrics": fdata.get("test_backtest_metrics") or {},
            "style_exposures": fdata.get("style_exposures"),
            "metadata": {
                "experiment_id": fdata.get("experiment_id", ""),
                "round_number": fdata.get("round_number", 0),
                "evolution_phase": fdata.get("evolution_phase", ""),
                "trajectory_id": fdata.get("trajectory_id", ""),
                "parent_trajectory_ids": fdata.get("parent_trajectory_ids", []),
                "hypothesis": fdata.get("hypothesis", ""),
                "initial_direction": fdata.get("initial_direction", ""),
                "user_initial_direction": fdata.get("user_initial_direction", ""),
                "planning_direction": fdata.get("planning_direction", ""),
            },
            "feedback": fdata.get("feedback") or {},
            "run_context": fdata.get("run_context") or {},
        }

    def update_factor_expression(
        self, factor_id: str, new_expression: str, suffix: str | None = None,
        re_evaluate: bool = True,
        description: str | None = None,
        formulation: str | None = None,
    ) -> dict | None:
        data = self.load_library(suffix)
        factors = data.get("factors", {})
        if factor_id not in factors:
            return None

        fdata = factors[factor_id]
        expression_changed = fdata.get("factor_expression") != new_expression
        fdata["factor_expression"] = new_expression
        if description is not None:
            fdata["factor_description"] = description
        if formulation is not None:
            fdata["factor_formulation"] = formulation
        re_evaluate = re_evaluate and expression_changed

        if re_evaluate:
            try:
                eval_result = self._re_evaluate_factor(
                    factor_id=factor_id,
                    factor_name=fdata.get("factor_name", factor_id),
                    expression=new_expression,
                )
                if eval_result.get("test_backtest_metrics"):
                    fdata["test_backtest_metrics"] = eval_result["test_backtest_metrics"]
                if eval_result.get("train_check_flags"):
                    fdata["train_check_flags"] = eval_result["train_check_flags"]
                if eval_result.get("test_check_flags"):
                    fdata["test_check_flags"] = eval_result["test_check_flags"]
                if eval_result.get("style_exposures") is not None:
                    fdata["style_exposures"] = eval_result["style_exposures"]
            except Exception as e:
                logger.error(f"因子重新评估失败 {factor_id}: {e}", exc_info=True)

        self.save_library(data, suffix)
        return self.get_factor_detail(factor_id, suffix)

    def _re_evaluate_factor(
        self,
        factor_id: str,
        factor_name: str,
        expression: str,
    ) -> dict[str, Any]:
        from quantaalpha.backtest.run_backtest import run_single_factor_backtest
        from quantaalpha.backtest.bridge import TQUpstreamBridge

        output_dir = results_root() / "re_evaluate" / factor_id
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = str(tq_upstream_config_path())

        result_paths = run_single_factor_backtest(
            expression=expression,
            factor_name=factor_name,
            config_path=config_path,
            output_dir=str(output_dir),
            plot=False,
            show=False,
            quality_report=True,
        )

        test_metrics = {}
        style_exposures = None
        quality_path = result_paths.get("quality_path")
        if quality_path and Path(quality_path).exists():
            report = json.loads(Path(quality_path).read_text(encoding="utf-8"))
            test_metrics = report.get("result_summary") or {}
            raw_styles = report.get("style_exposures")
            if raw_styles:
                style_exposures = raw_styles

        train_flags = {}
        test_flags = {}
        fv_path = output_dir / f"{factor_name}_factor_value.pkl"
        if fv_path.exists():
            try:
                import pandas as pd
                factor_value = pd.read_pickle(fv_path)
                bridge = TQUpstreamBridge(config_path)
                checks = bridge.evaluate_submission_checks_for_value(
                    factor_name=factor_name,
                    factor_value=factor_value,
                )
                criteria = checks.get("criteria", {})
                train_flags = {
                    "check_passed": criteria.get("check_passed", False),
                    "raw_passed": criteria.get("raw_passed", False),
                    "zz1000s_passed": criteria.get("zz1000s_passed", False),
                    "complete_passed": criteria.get("complete_passed", False),
                    "check_tag": criteria.get("tag_suggestion", ""),
                }
                test_flags = dict(train_flags)
            except Exception as e:
                logger.warning(f"Submission check 失败: {e}")

        return {
            "test_backtest_metrics": test_metrics,
            "train_check_flags": train_flags,
            "test_check_flags": test_flags,
            "style_exposures": style_exposures,
        }

    def delete_factor(self, factor_id: str, suffix: str | None = None) -> bool:
        data = self.load_library(suffix)
        factors = data.get("factors", {})
        if factor_id not in factors:
            return False
        del factors[factor_id]
        meta = data.get("metadata", {})
        meta["total_factors"] = len(factors)
        self.save_library(data, suffix)
        return True

    def get_summary(self) -> dict[str, Any]:
        libs = self.list_libraries(frontend_only=True)
        total = 0
        high = 0
        medium = 0
        low = 0
        for lib in libs:
            path = Path(lib["path"])
            if not path.exists():
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                for fdata in raw.get("factors", {}).values():
                    q = classify_quality(fdata)
                    total += 1
                    if q == "high":
                        high += 1
                    elif q == "medium":
                        medium += 1
                    else:
                        low += 1
            except Exception:
                continue
        return {
            "total_factors": total,
            "high_quality_count": high,
            "medium_quality_count": medium,
            "low_quality_count": low,
            "available_libraries": libs,
        }
