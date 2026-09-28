from __future__ import annotations
 
import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantaalpha.backtest import TQUpstreamBridge
from quantaalpha.backtest.submission_check import CHECK_STAGE_ORDER
from quantaalpha.components.runner import CachedRunner
from quantaalpha.core.exception import FactorEmptyError
from quantaalpha.core.utils import cache_with_pickle
from quantaalpha.factors.alignment.preflight_guard import validate_expression_against_registry
from quantaalpha.factors.combined_domain_contract import validate_joint_pv_minutes_contract
from quantaalpha.factors.data_domains import resolve_factor_domains
from quantaalpha.factors.data_domains import get_domain_field_map
from quantaalpha.factors.experiment import FactorMiningExperiment
from quantaalpha.log import logger


def _to_native_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, float):
        return float(value)
    return value


def _series_to_native_dict(series: pd.Series) -> dict[str, Any]:
    return {str(key): _to_native_value(val) for key, val in series.items()}


def _format_metric_snapshot(metrics: dict[str, Any], keys: tuple[str, ...]) -> str:
    parts: list[str] = []
    for key in keys:
        value = metrics.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


CHECK_FEEDBACK_METRIC_KEYS = (
    "long_ret",
    "long_ir",
    "long_netret",
    "ls_ir",
    "ls_netret",
    "rankic",
    "rankicir",
    "long_turnover",
    "coverage",
)


def _format_percent(value: float | None) -> str | None:
    if value is None or not np.isfinite(value):
        return None
    return f"{value * 100:.1f}%"


def _format_float(value: float | None, digits: int = 4) -> str | None:
    if value is None or not np.isfinite(value):
        return None
    return f"{value:.{digits}f}"


def _join_values(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _set_module_meta_tag(module_text: str, tag_value: str) -> str:
    text = str(module_text or "")
    if not text.strip():
        return text

    lines = text.splitlines(keepends=True)
    updated_lines: list[str] = []
    inside_meta = False
    tag_written = False
    inserted_tag = False
    line_ending = "\n"

    for line in lines:
        if line.endswith("\r\n"):
            line_ending = "\r\n"
        elif line.endswith("\n"):
            line_ending = "\n"

        if not inside_meta and re.match(r"^\s*META\s*=\s*\{\s*$", line):
            inside_meta = True
            tag_written = False
            inserted_tag = False
            updated_lines.append(line)
            continue

        if inside_meta and re.match(r'^\s*"tag"\s*:', line):
            indent = re.match(r"^(\s*)", line).group(1)
            updated_lines.append(f'{indent}"tag":"{tag_value}",{line_ending}')
            tag_written = True
            continue

        if inside_meta and re.match(r"^\s*\}\s*$", line):
            if not tag_written and not inserted_tag:
                updated_lines.append(f'    "tag":"{tag_value}",{line_ending}')
                inserted_tag = True
            inside_meta = False
            updated_lines.append(line)
            continue

        updated_lines.append(line)

    return "".join(updated_lines)


def _build_submission_check_context(check_result: dict[str, Any]) -> str:
    if not isinstance(check_result, dict) or not check_result:
        return ""

    stages = check_result.get("stages") or {}
    parts = [
        "Submission gating uses raw + zz1000s as the pass condition; complete only determines whether the factor should be tagged as smart_styles."
    ]
    feedback_text = str(check_result.get("feedback_text") or "").strip()
    if feedback_text:
        parts.append(feedback_text)

    for stage_name in CHECK_STAGE_ORDER:
        stage_payload = stages.get(stage_name) or {}
        metrics = stage_payload.get("metrics") or {}
        snapshot = _format_metric_snapshot(metrics, CHECK_FEEDBACK_METRIC_KEYS)
        if snapshot:
            parts.append(f"{stage_name} metrics: {snapshot}.")
    return " ".join(parts)


def _build_check_status_summary(check_result: dict[str, Any]) -> str:
    if not isinstance(check_result, dict) or not check_result:
        return ""
    return (
        f"check_passed={bool(check_result.get('check_passed'))}, "
        f"raw={bool(check_result.get('raw_passed'))}, "
        f"zz1000s={bool(check_result.get('zz1000s_passed'))}, "
        f"complete={bool(check_result.get('complete_passed'))}, "
        f"tag={str(check_result.get('tag_suggestion') or 'none')}"
    )


def _resolve_train_splits(
    configured_periods: dict[str, dict[str, Any]],
    configured_train_splits: list[str],
    primary_split: str,
) -> list[str]:
    normalized = [
        str(split_name).strip()
        for split_name in configured_train_splits
        if str(split_name).strip() and str(split_name).strip() in configured_periods
    ]
    if normalized:
        return normalized
    return [primary_split] if primary_split in configured_periods else []


def _flatten_failure_matrix(train_checks_by_split: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for split_name, check_payload in train_checks_by_split.items():
        criteria = dict(check_payload.get("criteria") or {})
        stages = dict(criteria.get("stages") or {})
        for stage_name in CHECK_STAGE_ORDER:
            stage_payload = dict(stages.get(stage_name) or {})
            details = dict(stage_payload.get("details") or {})
            for metric_name, metric_detail in details.items():
                detail = dict(metric_detail or {})
                matrix.append(
                    {
                        "split": str(split_name),
                        "stage": str(stage_name),
                        "metric": str(metric_name),
                        "value": _to_native_value(detail.get("value")),
                        "min": _to_native_value(detail.get("min")),
                        "max": _to_native_value(detail.get("max")),
                        "passed": bool(detail.get("passed")),
                    }
                )
    return matrix


def _build_train_failed_metrics(failure_matrix: list[dict[str, Any]]) -> list[str]:
    failed: list[str] = []
    for row in failure_matrix:
        if bool(row.get("passed")):
            continue
        split_name = str(row.get("split") or "").strip()
        stage_name = str(row.get("stage") or "").strip()
        metric_name = str(row.get("metric") or "").strip()
        if split_name and stage_name and metric_name:
            failed.append(f"{split_name}.{stage_name}.{metric_name}")
    return failed


def _build_train_matrix_text(
    train_checks_by_split: dict[str, dict[str, Any]],
    failure_matrix: list[dict[str, Any]],
) -> str:
    if not train_checks_by_split:
        return ""

    train_scope = "the configured train stage" if len(train_checks_by_split) == 1 else "all configured train stages"
    parts = [f"Train feedback is based on {train_scope}; optimize against all failing train items together."]
    failed_rows = [row for row in failure_matrix if not bool(row.get("passed"))]
    if not failed_rows:
        parts.append("All configured train stages passed the raw + zz1000s submission gate.")
    for split_name, payload in train_checks_by_split.items():
        criteria = dict(payload.get("criteria") or {})
        parts.append(f"{split_name}: {_build_check_status_summary(criteria)}.")
        split_failures = [row for row in failed_rows if str(row.get("split")) == str(split_name)]
        if not split_failures:
            parts.append(f"{split_name} has no failed train metrics.")
            continue
        failure_parts: list[str] = []
        for row in split_failures:
            value = row.get("value")
            min_value = row.get("min")
            max_value = row.get("max")
            detail_parts = [f"{row['stage']}.{row['metric']}"]
            if isinstance(value, float):
                detail_parts.append(f"value={value:.4f}")
            elif value is not None:
                detail_parts.append(f"value={value}")
            if isinstance(min_value, float):
                detail_parts.append(f"min={min_value:.4f}")
            elif min_value is not None:
                detail_parts.append(f"min={min_value}")
            if isinstance(max_value, float):
                detail_parts.append(f"max={max_value:.4f}")
            elif max_value is not None:
                detail_parts.append(f"max={max_value}")
            failure_parts.append(", ".join(detail_parts))
        parts.append(f"{split_name} failures: " + "; ".join(failure_parts) + ".")
    return " ".join(parts)


def _build_train_alpha_feedback(
    train_checks_by_split: dict[str, dict[str, Any]],
    runtime_analysis_context: str = "",
    selection_diagnostics_context: str = "",
) -> str:
    parts = ["Alpha evaluation uses train-stage results only; test metrics are recorded separately and are not part of the optimization loop."]
    train_text = _build_train_matrix_text(train_checks_by_split, _flatten_failure_matrix(train_checks_by_split))
    if train_text:
        parts.append(train_text)
    if runtime_analysis_context:
        parts.append(runtime_analysis_context)
    if selection_diagnostics_context:
        parts.append(selection_diagnostics_context)
    return " ".join(parts)


def _sort_group_columns(columns: list[str]) -> list[str]:
    def _key(name: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", str(name))
        return (int(match.group(1)) if match else 10**9, str(name))

    return sorted(columns, key=_key)


def _extract_expression_fields(expression: str | None) -> list[str]:
    text = str(expression or "").strip()
    if not text:
        return []
    try:
        check = validate_expression_against_registry(text)
    except Exception:
        return []
    return sorted(str(field) for field in check.get("used_fields", []) if str(field).strip())


def _build_runtime_analysis_payload(
    active_domains: tuple[str, ...],
    module_globals: dict[str, Any],
    module_text: str,
    task_expression: str | None = None,
) -> dict[str, Any]:
    setting = module_globals.get("SETTING") or {}
    meta = module_globals.get("META") or {}
    real_daily_fields = sorted(str(item).strip() for item in (setting.get("data_needed") or []) if str(item).strip())
    real_minute_inputs = sorted(_extract_minute_inputs(module_text))
    expression_fields = _extract_expression_fields(task_expression)
    module_level = str(meta.get("level") or "").strip().lower() or _infer_module_level(module_text)
    domain_field_map = get_domain_field_map(active_domains)

    domain_evidence: dict[str, list[str]] = {}
    for domain in active_domains:
        allowed_fields = set(domain_field_map.get(domain) or ())
        if domain == "minutes":
            observed = list(dict.fromkeys([*real_minute_inputs, *expression_fields]))
        else:
            observed = list(dict.fromkeys([*real_daily_fields, *expression_fields]))
        domain_evidence[domain] = [field for field in observed if field in allowed_fields]

    return {
        "active_domains": list(active_domains),
        "module_level": module_level,
        "real_daily_fields": real_daily_fields,
        "real_minute_inputs": real_minute_inputs,
        "expression_fields": expression_fields,
        "domain_evidence": domain_evidence,
    }


def _build_runtime_analysis_context(runtime_payload: dict[str, Any]) -> str:
    active_domains = [str(domain) for domain in (runtime_payload.get("active_domains") or []) if str(domain).strip()]
    if not active_domains:
        return ""

    parts = [f"Active domains in the current run: {', '.join(active_domains)}."]
    for domain in active_domains:
        evidence = [str(item) for item in (runtime_payload.get("domain_evidence") or {}).get(domain, []) if str(item).strip()]
        if domain == "minutes":
            parts.append(
                f"Minutes-domain evidence from real module inputs: level={runtime_payload.get('module_level', 'unknown')}, "
                f"minute_inputs={_join_values(evidence)}."
            )
        else:
            parts.append(f"{domain}-domain evidence from real fields: {_join_values(evidence)}.")

    real_daily_fields = [str(item) for item in (runtime_payload.get("real_daily_fields") or []) if str(item).strip()]
    real_minute_inputs = [str(item) for item in (runtime_payload.get("real_minute_inputs") or []) if str(item).strip()]
    if real_daily_fields:
        parts.append(f"Observed daily fields declared by the module: {_join_values(real_daily_fields)}.")
    if real_minute_inputs:
        parts.append(f"Observed minute inputs declared by the module: {_join_values(real_minute_inputs)}.")
    return " ".join(parts)


def _summarize_rankic_temporal_concentration(rankics: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(rankics, errors="coerce").dropna()
    if clean.empty:
        return {}

    abs_clean = clean.abs()
    total_abs = float(abs_clean.sum())
    if total_abs <= 0:
        return {}

    top1_share = float(abs_clean.nlargest(min(1, len(abs_clean))).sum() / total_abs)
    top5_share = float(abs_clean.nlargest(min(5, len(abs_clean))).sum() / total_abs)
    quantiles = clean.quantile([0.05, 0.95]) if len(clean) >= 5 else pd.Series([clean.min(), clean.max()], index=[0.05, 0.95])
    winsorized = clean.clip(lower=float(quantiles.loc[0.05]), upper=float(quantiles.loc[0.95]))

    if top1_share >= 0.20 or top5_share >= 0.50:
        assessment = "high"
    elif top1_share >= 0.12 or top5_share >= 0.35:
        assessment = "moderate"
    else:
        assessment = "low"

    return {
        "sample_size": int(len(clean)),
        "mean_rankic": float(clean.mean()),
        "median_rankic": float(clean.median()),
        "winsorized_mean_rankic_5_95": float(winsorized.mean()),
        "top1_abs_share": top1_share,
        "top5_abs_share": top5_share,
        "assessment": assessment,
    }


def _summarize_group_monotonicity(group_rets: pd.DataFrame) -> dict[str, Any]:
    if not isinstance(group_rets, pd.DataFrame) or group_rets.empty:
        return {}

    ordered_cols = _sort_group_columns([str(col) for col in group_rets.columns])
    ordered = group_rets.loc[:, ordered_cols].apply(pd.to_numeric, errors="coerce")
    mean_returns = ordered.mean(axis=0).dropna()
    if len(mean_returns) < 2:
        return {}

    values = mean_returns.to_numpy(dtype=float)
    diffs = np.diff(values)
    increasing_violations = int((diffs < 0).sum())
    decreasing_violations = int((diffs > 0).sum())
    direction = "increasing" if increasing_violations <= decreasing_violations else "decreasing"
    violations = increasing_violations if direction == "increasing" else decreasing_violations
    monotonic = violations == 0
    ranks = pd.Series(np.arange(len(mean_returns), dtype=float))
    spearman = pd.Series(values).corr(ranks, method="spearman")

    return {
        "direction": direction,
        "monotonic": monotonic,
        "violations": violations,
        "step_count": int(len(diffs)),
        "spearman": None if pd.isna(spearman) else float(spearman),
        "low_high_mean_spread": float(values[-1] - values[0]),
    }


def _build_selection_diagnostics_payload(factor_result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(factor_result, dict) or not factor_result:
        return {}
    payload: dict[str, Any] = {}
    rankic_summary = _summarize_rankic_temporal_concentration(factor_result.get("rankics", pd.Series(dtype=float)))
    if rankic_summary:
        payload["rankic_temporal_concentration"] = rankic_summary
    group_summary = _summarize_group_monotonicity(factor_result.get("group_rets", pd.DataFrame()))
    if group_summary:
        payload["group_monotonicity"] = group_summary
    return payload


def _build_selection_diagnostics_context(diagnostics_payload: dict[str, Any]) -> str:
    if not diagnostics_payload:
        return ""

    parts: list[str] = []
    rankic_summary = diagnostics_payload.get("rankic_temporal_concentration") or {}
    if rankic_summary:
        rankic_parts = []
        for label, raw_value in (
            ("mean_rankic", _format_float(rankic_summary.get("mean_rankic"))),
            ("median_rankic", _format_float(rankic_summary.get("median_rankic"))),
            ("winsorized_mean_5_95", _format_float(rankic_summary.get("winsorized_mean_rankic_5_95"))),
            ("top1_abs_share", _format_percent(rankic_summary.get("top1_abs_share"))),
            ("top5_abs_share", _format_percent(rankic_summary.get("top5_abs_share"))),
        ):
            if raw_value is not None:
                rankic_parts.append(f"{label}={raw_value}")
        assessment = str(rankic_summary.get("assessment") or "").strip()
        if assessment:
            rankic_parts.append(f"temporal_concentration={assessment}")
        sample_size = rankic_summary.get("sample_size")
        if sample_size:
            rankic_parts.append(f"n={sample_size}")
        if rankic_parts:
            parts.append(
                "RankIC concentration check based on the time series (this judges concentration across dates, not cross-sectional stock outliers): "
                + ", ".join(rankic_parts)
                + "."
            )

    group_summary = diagnostics_payload.get("group_monotonicity") or {}
    if group_summary:
        group_parts = []
        direction = str(group_summary.get("direction") or "").strip()
        if direction:
            group_parts.append(f"direction={direction}")
        group_parts.append(f"monotonic={'yes' if group_summary.get('monotonic') else 'no'}")
        group_parts.append(f"violations={int(group_summary.get('violations', 0))}/{int(group_summary.get('step_count', 0))}")
        spearman = _format_float(group_summary.get("spearman"))
        if spearman is not None:
            group_parts.append(f"spearman={spearman}")
        spread = _format_float(group_summary.get("low_high_mean_spread"))
        if spread is not None:
            group_parts.append(f"low_high_mean_spread={spread}")
        parts.append(
            "Group-return monotonicity check based on mean daily decile returns over the evaluation window: "
            + ", ".join(group_parts)
            + "."
        )

    return " ".join(parts)


def _build_alpha_evaluation_feedback(
    selection_metrics: dict[str, Any],
    split_name: str,
    check_result: dict[str, Any] | None = None,
    runtime_analysis_context: str = "",
    selection_diagnostics_context: str = "",
) -> str:
    metric_keys = CHECK_FEEDBACK_METRIC_KEYS + (
        "long_maxdd",
        "long_netir",
        "long_netmaxdd",
        "ls_ret",
        "ls_maxdd",
        "ls_netir",
        "ls_netmaxdd",
        "ls_turnover",
        "long_num",
    )
    selection_snapshot = _format_metric_snapshot(selection_metrics, metric_keys)

    parts = [
        f"Primary mining snapshot uses raw metrics on split={split_name}.",
    ]
    if selection_snapshot:
        parts.append(f"Raw selection metrics: {selection_snapshot}.")
    check_context = _build_submission_check_context(check_result or {})
    if check_context:
        parts.append(check_context)
    if runtime_analysis_context:
        parts.append(runtime_analysis_context)
    if selection_diagnostics_context:
        parts.append(selection_diagnostics_context)
    return " ".join(parts)


def _extract_workspace_factor_module(code_dict: dict[str, str] | None) -> tuple[str, str]:
    if not isinstance(code_dict, dict):
        return "", "days"
    for preferred in ("factor.py", "minute.py", "regular.py", "super.py"):
        module_text = str(code_dict.get(preferred) or "").strip()
        if module_text:
            return module_text, _infer_module_level(module_text)
    for module_text in code_dict.values():
        text = str(module_text or "").strip()
        if text:
            return text, _infer_module_level(text)
    return "", "days"


def _infer_module_level(module_text: str) -> str:
    text = str(module_text or "").strip()
    if not text:
        return "days"
    try:
        tree = ast.parse(text)
    except Exception:
        return "minutes" if "prepare_minute_datas" in text else "days"
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "META":
            try:
                meta = ast.literal_eval(node.value)
            except Exception:
                meta = None
            if isinstance(meta, dict):
                level = str(meta.get("level") or "").strip().lower()
                if level in {"days", "minutes"}:
                    return level
    return "minutes" if any(
        isinstance(node, ast.FunctionDef) and node.name == "prepare_minute_datas"
        for node in tree.body
    ) else "days"


def _extract_minute_inputs(module_text: str) -> set[str]:
    text = str(module_text or "").strip()
    if not text:
        return set()
    try:
        tree = ast.parse(text)
    except Exception:
        return set()

    minute_fields: set[str] = set()

    def _literal_text(node: ast.AST) -> str | None:
        try:
            parsed = ast.literal_eval(node)
        except Exception:
            return None
        if isinstance(parsed, str):
            text_item = parsed.strip()
            return text_item or None
        return None

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "prepare_minute_datas":
            continue
        for subnode in ast.walk(node):
            if not isinstance(subnode, ast.Return) or not isinstance(subnode.value, ast.Dict):
                continue
            for key_node in subnode.value.keys:
                text_item = _literal_text(key_node)
                if text_item:
                    minute_fields.add(text_item)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_single_minute":
            candidate_values: list[Any] = []
            if len(node.args) >= 2:
                candidate_values.append(node.args[1])
            for kw in node.keywords:
                if kw.arg == "fld":
                    candidate_values.append(kw.value)
            for value_node in candidate_values:
                text_item = _literal_text(value_node)
                if text_item:
                    minute_fields.add(text_item)
            continue

        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "handle":
                text_item = _literal_text(node.slice)
                if text_item and text_item.startswith("data/"):
                    field_name = text_item.split("/", 1)[1].strip()
                    if field_name:
                        minute_fields.add(field_name)
            continue
        if node.func.attr in {"run", "cached_run"}:
            candidate_values: list[Any] = []
            for kw in node.keywords:
                if kw.arg == "inputs":
                    candidate_values.append(kw.value)
            if node.args:
                candidate_values.append(node.args[0])

            for value_node in candidate_values:
                try:
                    parsed = ast.literal_eval(value_node)
                except Exception:
                    parsed = None
                if isinstance(parsed, str):
                    text_item = parsed.strip()
                    if text_item:
                        minute_fields.add(text_item)
                elif isinstance(parsed, (list, tuple)):
                    for item in parsed:
                        text_item = str(item).strip()
                        if text_item:
                            minute_fields.add(text_item)
    return minute_fields


def _validate_module_domain_coverage(
    module_globals: dict[str, Any],
    active_domains: tuple[str, ...],
    module_text: str = "",
) -> None:
    joint_contract_errors = validate_joint_pv_minutes_contract(module_text, active_domains=active_domains)
    if joint_contract_errors:
        raise ValueError("combined pv/minutes contract validation failed: " + "; ".join(joint_contract_errors))

    setting = module_globals.get("SETTING") or {}
    meta = module_globals.get("META") or {}
    data_needed = {str(item).strip() for item in (setting.get("data_needed") or []) if str(item).strip()}
    level = str(meta.get("level") or "").strip().lower()
    domain_field_map = get_domain_field_map(active_domains)
    minute_inputs = _extract_minute_inputs(module_text)
    missing_domains: list[str] = []

    for domain in active_domains:
        if domain == "minutes":
            allowed_fields = set(domain_field_map.get(domain) or ())
            if (
                level != "minutes"
                or not callable(module_globals.get("prepare_minute_datas"))
                or not (minute_inputs & allowed_fields)
            ):
                missing_domains.append(domain)
            continue
        allowed_fields = set(domain_field_map.get(domain) or ())
        if allowed_fields and not (data_needed & allowed_fields):
            missing_domains.append(domain)

    if missing_domains:
        raise ValueError(
            f"factor module does not cover all active domains={active_domains}; missing={missing_domains}, "
            f"data_needed={sorted(data_needed)}, minute_inputs={sorted(minute_inputs)}, level={level}"
        )


def _resolve_stage_splits(
    configured_periods: dict[str, dict[str, Any]],
    discovery_split: str,
    train_splits: list[str],
    test_split: str,
) -> dict[str, str]:
    stages: dict[str, str] = {}
    if discovery_split in configured_periods:
        stages["discovery"] = discovery_split
    for split_name in train_splits:
        split_text = str(split_name).strip()
        if split_text in configured_periods:
            stages[split_text] = split_text
    if test_split in configured_periods:
        stages["test"] = test_split
    return stages


class TQFactorRunner(CachedRunner[FactorMiningExperiment]):
    """使用统一的本地 TQ 回测引擎执行因子挖掘阶段评估。"""

    def __init__(self, scen, *args, **kwargs):
        super().__init__(scen, *args, **kwargs)
        self.bridge = TQUpstreamBridge(os.getenv("TQ_UPSTREAM_CONFIG_PATH"))

    def _resolve_output_dir(self, exp: FactorMiningExperiment) -> Path:
        workspace = getattr(getattr(exp, "experiment_workspace", None), "workspace_path", None)
        if workspace is not None:
            return Path(workspace) / "tq_candidates"
        return self.bridge.candidate_dir

    def get_cache_key(self, exp: FactorMiningExperiment, **kwargs) -> str:
        base_key = super().get_cache_key(exp, **kwargs)
        mining_cfg = self.bridge.config.get("mining") or {}
        cache_context = {
            "config_path": self.bridge.config.get("_config_path"),
            "profile_id": self.bridge.config.get("profile_id"),
            "profile_mode": self.bridge.config.get("profile_mode"),
            "metric_mode": self.bridge.config.get("metric_mode"),
            "use_net_metrics": bool(self.bridge.config.get("use_net_metrics", False)),
            "transform_spec": self.bridge.config.get("transform_spec") or {},
            "mining_params": self.bridge.get_eval_params("mining"),
            "execution_constraints": self.bridge.get_execution_constraints("mining"),
            "discovery_split": str(mining_cfg.get("discovery_split") or "train").strip() or "train",
            "train_splits": self.bridge.get_train_mining_splits(),
            "primary_split": self.bridge.get_primary_mining_split(),
            "test_split": str(mining_cfg.get("test_split") or "").strip(),
            "mining_periods": self.bridge.get_mining_period_params(),
        }
        payload = json.dumps(
            {"task_key": base_key, "cache_context": cache_context},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    @cache_with_pickle(lambda self, exp, **kwargs: self.get_cache_key(exp, **kwargs), CachedRunner.assign_cached_result)
    def develop(self, exp: FactorMiningExperiment, use_local: bool = True) -> FactorMiningExperiment:
        if exp.based_experiments:
            last_based_exp = exp.based_experiments[-1]
            last_based_tasks = list(getattr(last_based_exp, "sub_tasks", []) or [])
            if not last_based_tasks:
                last_based_tasks = list(getattr(last_based_exp, "tasks", []) or [])
            if last_based_tasks and last_based_exp.result is None:
                exp.based_experiments[-1] = self.develop(last_based_exp, use_local=use_local)

        runtime_check = self.bridge.check_runtime_fields(context="mining")
        if not runtime_check["ok"]:
            raise FactorEmptyError(
                "TQ runtime data is not ready: "
                f"{runtime_check['reason']} -> {runtime_check['missing_required']}"
            )

        backtest_mode = str(os.getenv("FACTOR_BACKTEST_MODE", "single")).strip().lower()
        if backtest_mode != "single":
            logger.warning(
                f"TQFactorRunner only supports per-factor staged backtest. FACTOR_BACKTEST_MODE={backtest_mode} will be treated as single."
            )

        output_dir = self._resolve_output_dir(exp)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, pd.Series] = {}
        structured_results: dict[str, dict[str, Any]] = {}
        export_manifest: list[dict[str, Any]] = []
        profile_id = self.bridge.config["profile_id"]
        metric_mode = str(self.bridge.config.get("metric_mode", "long_only"))
        use_net_metrics = bool(self.bridge.config.get("use_net_metrics", False))
        submit_after_check = bool(self.bridge.config.get("submit_after_check", False))
        mining_params = self.bridge.get_eval_params("mining")
        mining_periods = self.bridge.get_mining_period_params()
        mining_cfg = self.bridge.config.get("mining") or {}
        discovery_split = str(mining_cfg.get("discovery_split") or "train").strip() or "train"
        primary_split = self.bridge.get_primary_mining_split()
        train_splits = _resolve_train_splits(
            mining_periods,
            self.bridge.get_train_mining_splits(),
            primary_split,
        )
        test_split = str(mining_cfg.get("test_split") or "").strip()
        if not mining_periods:
            mining_periods = {primary_split: mining_params}
        stage_splits = _resolve_stage_splits(mining_periods, discovery_split, train_splits, test_split)
        active_domains = resolve_factor_domains()

        exp.tq_stage_backtest_results = {}
        exp.tq_stage_backtest_payloads = {}
        exp.tq_submission_checks = {}
        exp.tq_library_backtest_metrics = {}
        exp.tq_library_check_flags = {}
        implementation_selection_applied = bool(getattr(exp, "implementation_selection_applied", False))
        raw_candidate_tasks = list(getattr(exp, "sub_tasks", []) or [])
        raw_candidate_workspaces = list(getattr(exp, "sub_workspace_list", []) or [])
        requested_count = getattr(exp, "factor_generation_requested_count", None)
        proposed_count = getattr(exp, "factor_generation_proposed_count", None)
        unique_count = getattr(exp, "factor_generation_unique_count", None)
        logger.info(
            "TQ runner received coder candidates: "
            f"tasks={len(raw_candidate_tasks)}, workspaces={len(raw_candidate_workspaces)}, "
            f"selection_applied={implementation_selection_applied}, "
            f"requested_per_direction={requested_count}, proposed={proposed_count}, unique={unique_count}"
        )
        candidate_tasks = []
        candidate_workspaces = []
        skipped_failed_implementation: list[str] = []
        skipped_failed_details: list[str] = []
        for task_idx, task in enumerate(raw_candidate_tasks):
            implementation_passed = bool(getattr(task, "factor_implementation", False))
            if not implementation_passed:
                factor_name = getattr(task, "factor_name", getattr(task, "name", f"factor_{task_idx}"))
                factor_name = str(factor_name)
                skipped_failed_implementation.append(factor_name)
                feedback = str(getattr(task, "alpha_evaluation_feedback", "") or "").strip()
                if feedback:
                    first_feedback_line = next((line.strip() for line in feedback.splitlines() if line.strip()), "")
                    if first_feedback_line:
                        skipped_failed_details.append(f"{factor_name}: {first_feedback_line[:300]}")
                continue
            candidate_tasks.append(task)
            candidate_workspaces.append(raw_candidate_workspaces[task_idx] if task_idx < len(raw_candidate_workspaces) else None)
        if skipped_failed_implementation:
            logger.warning(
                "Skip factors with final_decision=false before TQ runner: "
                + ", ".join(skipped_failed_implementation)
            )
            if skipped_failed_details:
                logger.warning(
                    "Coder failure feedback for skipped factors: "
                    + " | ".join(skipped_failed_details)
                )
        if not candidate_tasks and not raw_candidate_tasks and not implementation_selection_applied:
            candidate_tasks = list(getattr(exp, "tasks", []) or [])
            candidate_workspaces = [None] * len(candidate_tasks)
            if candidate_tasks:
                logger.warning("exp.sub_tasks is empty, fallback to exp.tasks for staged mining backtest.")

        skipped_contract_violations: list[str] = []
        for task_idx, task in enumerate(candidate_tasks):
            factor_name = getattr(task, "factor_name", getattr(task, "name", "unknown_factor"))
            logger.info(f"Starting staged mining backtest: {factor_name}")
            workspace = candidate_workspaces[task_idx] if task_idx < len(candidate_workspaces) else None
            module_text, module_level = _extract_workspace_factor_module(getattr(workspace, "code_dict", {}))
            exported_text = ""
            try:
                if module_text:
                    factor_path = output_dir / f"{factor_name}.py"
                    factor_path.write_text(module_text, encoding="utf-8")
                    self.bridge.validate_factor_file(factor_path)
                    module_globals = self.bridge._load_local_factor_module(factor_path)
                    _validate_module_domain_coverage(module_globals, active_domains, module_text=module_text)
                    exported = {
                        "factor_name": factor_name,
                        "factor_path": str(factor_path),
                        "spec": {
                            "factor_name": factor_name,
                            "level": module_level,
                        },
                    }
                else:
                    if "minutes" in active_domains:
                        raise ValueError(
                            f"minute-domain factor {factor_name} did not produce generated module code; "
                            "minutes runs require a concrete factor module with META.level='minutes' "
                            "and prepare_minute_datas(), and cannot fall back to expression export."
                        )
                    exported = self.bridge.export_task(task, output_dir=output_dir)
                    module_globals = self.bridge._load_local_factor_module(exported["factor_path"])
                    exported_text = Path(exported["factor_path"]).read_text(encoding="utf-8")
                    _validate_module_domain_coverage(module_globals, active_domains, module_text=exported_text)
            except ValueError as contract_exc:
                logger.warning(
                    f"Skip factor {factor_name} due to domain/contract validation failure: {contract_exc}"
                )
                try:
                    task.factor_implementation = False
                    task.alpha_evaluation_feedback = (
                        (getattr(task, "alpha_evaluation_feedback", "") or "")
                        + f"\n[runner-domain-contract] {contract_exc}"
                    ).strip()
                except Exception:
                    pass
                skipped_contract_violations.append(str(factor_name))
                continue
            factor_value = self.bridge.evaluate_factor_file_value(exported["factor_path"], context="mining")
            runtime_payload = _build_runtime_analysis_payload(
                active_domains=active_domains,
                module_globals=module_globals,
                module_text=module_text or exported_text,
                task_expression=getattr(task, "factor_expression", ""),
            )
            runtime_analysis_context = _build_runtime_analysis_context(runtime_payload)

            staged_metrics: dict[str, dict[str, Any]] = {}
            selection_diagnostics_payload: dict[str, Any] = {}
            train_checks_by_split: dict[str, dict[str, Any]] = {}
            train_stage_payloads: dict[str, dict[str, Any]] = {}
            test_check: dict[str, Any] | None = None
            test_stage_payloads: dict[str, Any] | None = None
            primary_train_stage_payload: dict[str, Any] | None = None

            for split_name in train_splits:
                split_params = mining_periods.get(split_name) or mining_params
                split_metrics: dict[str, Any] = {}
                current_raw_stage_payload: dict[str, Any] | None = None
                if split_name == primary_split:
                    evaluation = self.bridge.evaluate_factor_value(
                        factor_name=factor_name,
                        factor_value=factor_value,
                        params=split_params,
                        context="mining",
                    )
                    current_raw_stage_payload = {
                        "factor_performance": dict(evaluation["factor_performance"]),
                        "factor_result": evaluation["factor_result"],
                        "summary": evaluation["summary"].copy(),
                    }
                    primary_train_stage_payload = current_raw_stage_payload
                    split_metrics = _series_to_native_dict(evaluation["summary"][factor_name])
                    selection_diagnostics_payload = _build_selection_diagnostics_payload(evaluation.get("factor_result", {}))
                else:
                    split_series = self.bridge.summarize_factor_value(
                        factor_name=factor_name,
                        factor_value=factor_value,
                        params=split_params,
                        context="mining",
                    )
                    split_metrics = _series_to_native_dict(split_series)

                split_check = self.bridge.evaluate_submission_checks_for_value(
                    factor_name=factor_name,
                    factor_value=factor_value,
                    profile_id=profile_id,
                    params=split_params,
                    context="mining",
                    execution_constraints=self.bridge.get_execution_constraints("mining"),
                    precomputed_stage_payloads={"raw": current_raw_stage_payload} if current_raw_stage_payload else None,
                )
                train_checks_by_split[split_name] = split_check
                train_stage_payloads[split_name] = split_check.get("stage_payloads") or {}
                compact_metrics_by_stage = {
                    str(stage_name): {
                        str(metric_name): metric_value
                        for metric_name, metric_value in dict(stage_metrics_payload or {}).items()
                    }
                    for stage_name, stage_metrics_payload in dict(split_check.get("metrics_by_stage") or {}).items()
                }
                split_criteria = dict(split_check.get("criteria") or {})
                staged_metrics[split_name] = {
                    "split": split_name,
                    "metrics": split_metrics,
                    "metric_source": "raw",
                    "transform_context": {
                        "active": False,
                        "base_universe": self.bridge.get_backtest_universe(),
                        "subuniverse": [],
                        "neutralize": [],
                        "metric_source": "raw",
                        "applied_steps": [],
                        "description": "raw",
                    },
                    "params": dict(split_params),
                    "submission_check": {
                        "check_passed": bool(split_criteria.get("check_passed")),
                        "raw_passed": bool(split_criteria.get("raw_passed")),
                        "zz1000s_passed": bool(split_criteria.get("zz1000s_passed")),
                        "complete_passed": bool(split_criteria.get("complete_passed")),
                        "tag_suggestion": str(split_criteria.get("tag_suggestion") or ""),
                        "metrics_by_stage": compact_metrics_by_stage,
                    },
                }

            primary_params = mining_periods.get(primary_split) or mining_params
            primary_metrics = dict((staged_metrics.get(primary_split) or {}).get("metrics") or {})
            if not primary_metrics:
                primary_series = self.bridge.summarize_factor_value(
                    factor_name=factor_name,
                    factor_value=factor_value,
                    params=primary_params,
                    context="mining",
                )
                primary_metrics = _series_to_native_dict(primary_series)

            if test_split in mining_periods:
                test_params = mining_periods.get(test_split) or mining_params
                test_check = self.bridge.evaluate_submission_checks_for_value(
                    factor_name=factor_name,
                    factor_value=factor_value,
                    profile_id=profile_id,
                    params=test_params,
                    context="mining",
                    execution_constraints=self.bridge.get_execution_constraints("mining"),
                )
                test_stage_payloads = test_check.get("stage_payloads") or {}
                compact_test_metrics_by_stage = {
                    str(stage_name): {
                        str(metric_name): metric_value
                        for metric_name, metric_value in dict(stage_metrics_payload or {}).items()
                    }
                    for stage_name, stage_metrics_payload in dict(test_check.get("metrics_by_stage") or {}).items()
                }
                test_criteria = dict(test_check.get("criteria") or {})
                test_raw_metrics = dict(compact_test_metrics_by_stage.get("raw") or {})
                staged_metrics["test"] = {
                    "split": test_split,
                    "metrics": test_raw_metrics,
                    "metric_source": "raw",
                    "transform_context": {
                        "active": False,
                        "base_universe": self.bridge.get_backtest_universe(),
                        "subuniverse": [],
                        "neutralize": [],
                        "metric_source": "raw",
                        "applied_steps": [],
                        "description": "raw",
                    },
                    "params": dict(test_params),
                    "submission_check": {
                        "check_passed": bool(test_criteria.get("check_passed")),
                        "raw_passed": bool(test_criteria.get("raw_passed")),
                        "zz1000s_passed": bool(test_criteria.get("zz1000s_passed")),
                        "complete_passed": bool(test_criteria.get("complete_passed")),
                        "tag_suggestion": str(test_criteria.get("tag_suggestion") or ""),
                        "metrics_by_stage": compact_test_metrics_by_stage,
                    },
                }
            else:
                compact_test_metrics_by_stage = {}
                test_criteria = {}
                test_raw_metrics = {}

            train_failure_matrix = _flatten_failure_matrix(train_checks_by_split)
            train_failed_metrics = _build_train_failed_metrics(train_failure_matrix)
            train_feedback_text = _build_train_matrix_text(
                train_checks_by_split,
                train_failure_matrix,
            )
            train_passed = bool(train_checks_by_split) and all(
                bool(dict(payload.get("criteria") or {}).get("check_passed"))
                for payload in train_checks_by_split.values()
            )
            train_raw_passed = bool(train_checks_by_split) and all(
                bool(dict(payload.get("criteria") or {}).get("raw_passed"))
                for payload in train_checks_by_split.values()
            )
            train_zz1000s_passed = bool(train_checks_by_split) and all(
                bool(dict(payload.get("criteria") or {}).get("zz1000s_passed"))
                for payload in train_checks_by_split.values()
            )
            train_complete_passed = bool(train_checks_by_split) and all(
                bool(dict(payload.get("criteria") or {}).get("complete_passed"))
                for payload in train_checks_by_split.values()
            )
            test_passed = bool(test_criteria.get("check_passed")) if test_check is not None else True
            test_raw_passed = bool(test_criteria.get("raw_passed")) if test_check is not None else True
            test_zz1000s_passed = bool(test_criteria.get("zz1000s_passed")) if test_check is not None else True
            test_complete_passed = bool(test_criteria.get("complete_passed")) if test_check is not None else True
            test_tag_suggestion = str(test_criteria.get("tag_suggestion") or "")
            train_check_flags = {
                "check_passed": bool(train_passed),
                "raw_passed": bool(train_raw_passed),
                "zz1000s_passed": bool(train_zz1000s_passed),
                "complete_passed": bool(train_complete_passed),
                "check_tag": "",
            }
            train_tag_suggestion = (
                "llm_X"
                if (train_passed and train_complete_passed)
                else ("smart_styles" if train_passed else "")
            )
            train_check_flags["check_tag"] = str(train_tag_suggestion)
            test_check_flags = {
                "check_passed": bool(test_passed),
                "raw_passed": bool(test_raw_passed),
                "zz1000s_passed": bool(test_zz1000s_passed),
                "complete_passed": bool(test_complete_passed),
                "check_tag": str(test_tag_suggestion),
            }
            library_check_flags = {
                "check_passed": bool(train_passed and test_passed),
                "train_passed": bool(train_passed),
                "test_passed": bool(test_passed),
                "train_check_flags": dict(train_check_flags),
                "test_check_flags": dict(test_check_flags),
            }
            optimization_submission_check = {
                "check_passed": bool(train_passed),
                "train_passed": bool(train_passed),
                "raw_passed": bool(train_raw_passed),
                "zz1000s_passed": bool(train_zz1000s_passed),
                "complete_passed": bool(train_complete_passed),
                "tag_suggestion": str(train_tag_suggestion),
                "train_splits": list(train_splits),
                "failed_metrics": list(train_failed_metrics),
                "feedback_text": str(train_feedback_text),
                "train_matrix": train_failure_matrix,
                "train_status_by_split": {
                    str(split_name): {
                        "check_passed": bool(dict(payload.get("criteria") or {}).get("check_passed")),
                        "raw_passed": bool(dict(payload.get("criteria") or {}).get("raw_passed")),
                        "zz1000s_passed": bool(dict(payload.get("criteria") or {}).get("zz1000s_passed")),
                        "complete_passed": bool(dict(payload.get("criteria") or {}).get("complete_passed")),
                    }
                    for split_name, payload in train_checks_by_split.items()
                },
                "metrics_by_stage": {},
            }
            flat_result = dict(primary_metrics)
            flat_result["tq_profile_id"] = profile_id
            flat_result["tq_metric_mode"] = metric_mode
            flat_result["tq_use_net_metrics"] = use_net_metrics
            flat_result["tq_primary_split"] = primary_split
            flat_result["tq_eval_splits"] = tuple(train_splits)
            flat_result["tq_metric_source"] = "raw"
            flat_result["check_passed"] = bool(optimization_submission_check.get("check_passed"))
            flat_result["train_passed"] = bool(optimization_submission_check.get("train_passed"))
            flat_result["raw_passed"] = bool(optimization_submission_check.get("raw_passed"))
            flat_result["zz1000s_passed"] = bool(optimization_submission_check.get("zz1000s_passed"))
            flat_result["complete_passed"] = bool(optimization_submission_check.get("complete_passed"))
            results[factor_name] = pd.Series(flat_result, name=factor_name)

            structured_results[factor_name] = {
                **dict(primary_metrics),
                "tq_primary_split": primary_split,
                "train_splits": list(train_splits),
                "test_split": test_split,
                "check_passed": bool(optimization_submission_check.get("check_passed")),
                "train_passed": bool(optimization_submission_check.get("train_passed")),
                "raw_passed": bool(optimization_submission_check.get("raw_passed")),
                "zz1000s_passed": bool(optimization_submission_check.get("zz1000s_passed")),
                "complete_passed": bool(optimization_submission_check.get("complete_passed")),
            }
            exp.tq_stage_backtest_results[factor_name] = staged_metrics
            exp.tq_stage_backtest_payloads[factor_name] = {
                "primary_train": primary_train_stage_payload or {},
                "train": train_stage_payloads,
                "test": test_stage_payloads or {},
            }
            if not hasattr(exp, "tq_submission_checks") or not isinstance(getattr(exp, "tq_submission_checks"), dict):
                exp.tq_submission_checks = {}
            exp.tq_submission_checks[factor_name] = {
                "check_passed": bool(optimization_submission_check.get("check_passed")),
                "train_passed": bool(optimization_submission_check.get("train_passed")),
                "raw_passed": bool(optimization_submission_check.get("raw_passed")),
                "zz1000s_passed": bool(optimization_submission_check.get("zz1000s_passed")),
                "complete_passed": bool(optimization_submission_check.get("complete_passed")),
                "tag_suggestion": str(optimization_submission_check.get("tag_suggestion") or ""),
                "metrics_by_stage": {},
                "train_splits": list(train_splits),
                "train_matrix": train_failure_matrix,
                "failed_metrics": list(optimization_submission_check.get("failed_metrics") or []),
                "feedback_text": str(optimization_submission_check.get("feedback_text") or ""),
            }
            exp.tq_library_backtest_metrics[factor_name] = {
                **dict(test_raw_metrics),
                "tq_metric_split": test_split or primary_split,
            }
            exp.tq_library_check_flags[factor_name] = dict(library_check_flags)
            task.runtime_analysis_payload = runtime_payload
            task.runtime_analysis_context = runtime_analysis_context
            task.selection_diagnostics_payload = selection_diagnostics_payload
            task.selection_diagnostics_context = _build_selection_diagnostics_context(selection_diagnostics_payload)
            task.submission_check = dict(optimization_submission_check)
            task.submission_check["metrics_by_stage"] = {}
            task.alpha_evaluation_feedback = _build_train_alpha_feedback(
                train_checks_by_split=train_checks_by_split,
                runtime_analysis_context=task.runtime_analysis_context,
                selection_diagnostics_context=task.selection_diagnostics_context,
            )
            desired_tag = str(train_tag_suggestion or "")
            if module_text:
                updated_module_text = _set_module_meta_tag(module_text, desired_tag)
                module_text = updated_module_text
                if workspace is not None and isinstance(getattr(workspace, "code_dict", None), dict):
                    for preferred in ("factor.py", "minute.py", "regular.py", "super.py"):
                        current_text = workspace.code_dict.get(preferred)
                        if isinstance(current_text, str) and current_text.strip():
                            workspace.code_dict[preferred] = updated_module_text
                            break
                factor_path = Path(exported["factor_path"])
                factor_path.write_text(updated_module_text, encoding="utf-8")
            elif exported.get("factor_path"):
                factor_path = Path(exported["factor_path"])
                updated_exported_text = _set_module_meta_tag(factor_path.read_text(encoding="utf-8"), desired_tag)
                factor_path.write_text(updated_exported_text, encoding="utf-8")
                exported_text = updated_exported_text

            exported["check_summary"] = {
                "stages": staged_metrics,
                "split": primary_split,
                "metrics": primary_metrics,
                "train_splits": list(train_splits),
                "test_split": test_split,
                "transform_context": {
                    "active": False,
                    "base_universe": self.bridge.get_backtest_universe(),
                    "subuniverse": [],
                    "neutralize": [],
                    "metric_source": "raw",
                    "applied_steps": [],
                    "description": "raw",
                },
                "submission_check": {
                    "check_passed": bool(optimization_submission_check.get("check_passed")),
                    "train_passed": bool(optimization_submission_check.get("train_passed")),
                    "raw_passed": bool(optimization_submission_check.get("raw_passed")),
                    "zz1000s_passed": bool(optimization_submission_check.get("zz1000s_passed")),
                    "complete_passed": bool(optimization_submission_check.get("complete_passed")),
                    "tag_suggestion": str(optimization_submission_check.get("tag_suggestion") or ""),
                    "metrics_by_stage": {},
                    "train_matrix": train_failure_matrix,
                    "failed_metrics": list(optimization_submission_check.get("failed_metrics") or []),
                    "feedback_text": str(optimization_submission_check.get("feedback_text") or ""),
                },
                "runtime_analysis": runtime_payload,
                "selection_diagnostics": selection_diagnostics_payload,
            }
            if submit_after_check:
                exported["submit_result"] = self.bridge.submit(exported["factor_path"])
            export_manifest.append(exported)

        if skipped_contract_violations:
            logger.warning(
                "Skipped factors due to runner-side domain/contract validation: "
                + ", ".join(skipped_contract_violations)
            )

        if not results:
            raise FactorEmptyError("No valid factors were exported to TQ for train checks.")

        exp.result = results
        exp.tq_structured_backtest_results = structured_results
        exp.tq_upstream_manifest = export_manifest
        exp.tq_runtime_context = {
            "backtest_engine": "tq",
            "backtest_mode": "single",
            "profile_id": profile_id,
            "profile_mode": self.bridge.config.get("profile_mode", "o1_o2"),
            "backtest_universe": self.bridge.get_backtest_universe(),
            "metric_mode": metric_mode,
            "use_net_metrics": use_net_metrics,
            "transform_spec": self.bridge.get_transform_spec(),
            "params": mining_params,
            "discovery_split": discovery_split,
            "train_splits": list(train_splits),
            "primary_split": primary_split,
            "test_split": test_split,
            "stage_splits": stage_splits,
            "mining_periods": mining_periods,
            "data_mode": str(os.getenv("FACTOR_DATA_MODE", "daily")).strip().lower(),
            "prompt_mode": str(os.getenv("FACTOR_PROMPT_MODE", "daily")).strip().lower(),
            "data_domains": list(resolve_factor_domains()),
        }
        return exp
