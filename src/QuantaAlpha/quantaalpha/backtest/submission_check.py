from __future__ import annotations

from typing import Any

import math

CHECK_STAGE_ORDER = ("raw", "zz1000s", "complete")

DEFAULT_SUBMISSION_CRITERIA: dict[str, dict[str, dict[str, float]]] = {
    "raw": {
        "long_ret": {"min": 0.08},
        "long_netret": {"min": 0.0},
        "ls_ir": {"min": 1.5},
        "ls_netret": {"min": 0.0},
        "rankic": {"min": 0.01},
        "rankicir": {"min": 3.0},
        "long_turnover": {"min": 0.05, "max": 1.0},
        "coverage": {"min": 0.7},
    },
    "zz1000s": {
        "ls_ir": {"min": 1.0},
    },
    "complete": {
        "ls_ir": {"min": 0.0},
        "rankicir": {"min": 0.0},
    },
}


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _evaluate_metric(value: Any, bounds: dict[str, float]) -> dict[str, Any]:
    numeric_value = _to_float(value)
    passed = numeric_value is not None
    failed_checks: list[str] = []

    min_value = bounds.get("min")
    if min_value is not None and (numeric_value is None or numeric_value < float(min_value)):
        passed = False
        failed_checks.append("min")

    max_value = bounds.get("max")
    if max_value is not None and (numeric_value is None or numeric_value > float(max_value)):
        passed = False
        failed_checks.append("max")

    return {
        "value": numeric_value,
        "min": _to_float(min_value),
        "max": _to_float(max_value),
        "passed": passed,
        "failed_checks": failed_checks,
    }


def _format_metric_failure(stage_name: str, metric_name: str, metric_result: dict[str, Any]) -> str:
    value = metric_result.get("value")
    min_value = metric_result.get("min")
    max_value = metric_result.get("max")

    if value is None:
        threshold_parts: list[str] = []
        if min_value is not None:
            threshold_parts.append(f">= {min_value:.4f}")
        if max_value is not None:
            threshold_parts.append(f"<= {max_value:.4f}")
        threshold_text = " and ".join(threshold_parts) if threshold_parts else "available"
        return f"{stage_name}.{metric_name} is missing, expected {threshold_text}"

    failed_checks = metric_result.get("failed_checks") or []
    if failed_checks == ["min"] and min_value is not None:
        return f"{stage_name}.{metric_name}={value:.4f} < {min_value:.4f}"
    if failed_checks == ["max"] and max_value is not None:
        return f"{stage_name}.{metric_name}={value:.4f} > {max_value:.4f}"

    parts = [f"{stage_name}.{metric_name}={value:.4f}"]
    if min_value is not None:
        parts.append(f"min {min_value:.4f}")
    if max_value is not None:
        parts.append(f"max {max_value:.4f}")
    return ", ".join(parts)


def evaluate_submission_metrics(
    metrics_by_stage: dict[str, dict[str, Any]],
    criteria: dict[str, dict[str, dict[str, float]]] | None = None,
) -> dict[str, Any]:
    effective_criteria = criteria or DEFAULT_SUBMISSION_CRITERIA
    stages: dict[str, dict[str, Any]] = {}
    failed_metrics: list[str] = []

    for stage_name in CHECK_STAGE_ORDER:
        stage_metrics = dict(metrics_by_stage.get(stage_name) or {})
        stage_criteria = dict(effective_criteria.get(stage_name) or {})
        metric_details: dict[str, dict[str, Any]] = {}
        stage_failed_metrics: list[str] = []

        for metric_name, bounds in stage_criteria.items():
            metric_result = _evaluate_metric(stage_metrics.get(metric_name), bounds)
            metric_details[metric_name] = metric_result
            if not metric_result["passed"]:
                stage_failed_metrics.append(metric_name)
                failed_metrics.append(_format_metric_failure(stage_name, metric_name, metric_result))

        stages[stage_name] = {
            "passed": len(stage_failed_metrics) == 0,
            "metrics": stage_metrics,
            "details": metric_details,
            "failed_metrics": stage_failed_metrics,
        }

    raw_passed = bool(stages.get("raw", {}).get("passed"))
    zz1000s_passed = bool(stages.get("zz1000s", {}).get("passed"))
    complete_passed = bool(stages.get("complete", {}).get("passed"))
    check_passed = raw_passed and zz1000s_passed
    tag_suggestion = "llm_X" if (check_passed and complete_passed) else ("smart_styles" if check_passed else "")

    feedback_parts = [
        "Submission check status: "
        f"raw={'pass' if raw_passed else 'fail'}, "
        f"zz1000s={'pass' if zz1000s_passed else 'fail'}, "
        f"complete={'pass' if complete_passed else 'fail'}, "
        f"check_passed={'true' if check_passed else 'false'}, "
        f"tag={tag_suggestion if tag_suggestion else 'none'}."
    ]
    if failed_metrics:
        feedback_parts.append("Optimize these metrics: " + "; ".join(failed_metrics) + ".")
    else:
        feedback_parts.append("All submission thresholds passed.")

    return {
        "check_passed": check_passed,
        "raw_passed": raw_passed,
        "zz1000s_passed": zz1000s_passed,
        "complete_passed": complete_passed,
        "tag_suggestion": tag_suggestion,
        "stages": stages,
        "failed_metrics": failed_metrics,
        "feedback_text": " ".join(feedback_parts),
    }
