import json
import os
import re
from pathlib import Path
from typing import List, Tuple

from jinja2 import Environment, StrictUndefined

from quantaalpha.factors.coder.factor import FactorExperiment, FactorTask
from quantaalpha.components.proposal import FactorHypothesis2Experiment, FactorHypothesisGen
from quantaalpha.core.proposal import Hypothesis, Scenario, Trace
from quantaalpha.core.experiment import Experiment
from quantaalpha.factors.experiment import FactorMiningExperiment
from quantaalpha.llm.client import APIBackend, robust_json_parse
import pandas as pd
from quantaalpha.log import logger
from quantaalpha.factors.alignment.preflight_guard import validate_expression_against_registry
from quantaalpha.factors.alignment.prompt_proxy import DomainPromptProxy
from quantaalpha.factors.alignment.registry import (
    get_allowed_field_names,
    get_allowed_operator_names,
    get_operator_arity_constraints,
    get_prompt_operator_names,
    get_prompt_operator_signature_hints,
    get_operator_semantic_notes,
)
from quantaalpha.factors import data_domains as factor_data_domains
from quantaalpha.factors.coder.minute_op_registry import get_minute_op_spec
from quantaalpha.factors.coder.minute_spec_compiler import MAX_INTRADAY_LAG, compile_expression_to_minute_spec
from quantaalpha.factors.regulator.factor_regulator import FactorRegulator

DEFAULT_HISTORY_LIMIT = 6
MIN_HISTORY_LIMIT = 1


def _build_function_lib_description(base_description: str) -> str:
    domains = factor_data_domains.resolve_factor_domains()
    allowed_operators = sorted(get_prompt_operator_names(domains))
    allowed_fields = sorted(get_allowed_field_names(domains))
    arity_constraints = get_operator_arity_constraints(domains)
    signature_hints = get_prompt_operator_signature_hints(domains)
    operator_allowlist = ", ".join(allowed_operators) if allowed_operators else "none"
    field_allowlist = ", ".join(allowed_fields) if allowed_fields else "none"
    required_signature_ops = [
        signature_hints[name]
        for name in allowed_operators
        if name in signature_hints and arity_constraints.get(name, (0, None))[0] > 1
    ]
    signature_note = ""
    if required_signature_ops:
        signature_note = (
            "- Do not omit required operator arguments. Runtime signatures that require explicit extra args include: "
            + ", ".join(required_signature_ops)
            + "\n"
        )
    semantic_notes = get_operator_semantic_notes(domains)
    semantic_note = ""
    if semantic_notes:
        semantic_note = "Operator semantic notes:\n- " + "\n- ".join(semantic_notes) + "\n"
    joint_domain_note = ""
    if "pv" in domains and "minutes" in domains:
        joint_domain_note = (
            "\n"
            "- Joint pv/minutes expressions must contain at least one explicit daily pv field and at least one explicit minute field.\n"
            "- Do not rely on shared names like `volumes` alone to satisfy both domains.\n"
            f"{_describe_joint_pv_minutes_semantics()}\n"
        )
    minute_runtime_note = ""
    if "minutes" in domains:
        minute_runtime_note = (
            "- In minute-domain or joint-domain runs, the expression must be compilable into the structured minute runtime used by the coder.\n"
            "- Inside raw minute reducers such as `ts_corr/ts_regression/mean/...`, use only raw minute fields or registered minute tensor operators.\n"
            "- Apply daily-frame operators such as `winsorize1`, `winsorize2`, `bucket`, `ts_delay`, and `cs_*` only after the minute subtree has already been reduced to a daily 2D frame.\n"
            "- `winsorize1(df, n)` is an n-sigma clip, not percentile clipping. If you need percentile/quantile clipping, use `winsorize2(df, lower_bound, upper_bound)` on the post-reduction daily frame.\n"
        )
    return (
        "Use only the current TQ/QA runtime registry.\n\n"
        "Hard constraints:\n"
        "- Use lowercase operator names only.\n"
        "- Do not use legacy style variables like $open, $close, HIGH, LOW, VOLUME.\n"
        "- Do not invent free variables such as n or w_1.\n"
        "- If multiple data domains are active, every factor expression must use at least one field from each active domain.\n"
        "- Ignore stale operator names or examples elsewhere; the runtime allowlists below are the source of truth.\n\n"
        "Runtime registry constraints for this run:\n"
        f"- {factor_data_domains.describe_factor_domains(domains)}\n"
        f"- Only use operators from this allowlist: {operator_allowlist}\n"
        f"- Only use fields from this allowlist: {field_allowlist}\n"
        f"{signature_note}"
        f"{semantic_note}"
        f"{minute_runtime_note}"
        f"{joint_domain_note}\n"
        f"{base_description}"
    )


def _resolve_target_factor_count() -> int:
    try:
        return max(1, int(os.getenv("FACTOR_FACTORS_PER_HYPOTHESIS", "1")))
    except (TypeError, ValueError):
        return 1


def _resolve_max_refill_attempts(target_factor_count: int) -> int:
    """Give refill enough tries to top up partial batches after filtering."""
    try:
        target = max(1, int(target_factor_count))
    except (TypeError, ValueError):
        target = 1
    return max(6, target * 3)


def _resolve_refill_request_count(target_factor_count: int, accepted_count: int) -> int:
    try:
        target = max(1, int(target_factor_count))
    except (TypeError, ValueError):
        target = 1
    try:
        accepted = max(0, int(accepted_count))
    except (TypeError, ValueError):
        accepted = 0
    return max(0, target - accepted)


def _trim_factor_payload(response_dict: dict, target_factor_count: int) -> dict:
    trimmed: dict = {}
    for factor_name, factor_data in response_dict.items():
        if len(trimmed) >= target_factor_count:
            break
        if isinstance(factor_data, dict):
            trimmed[factor_name] = factor_data
    return trimmed


def _candidate_runtime_rejection_reason(factor_name: str, factor_data: dict, expr: str) -> str | None:
    try:
        domains = tuple(factor_data_domains.resolve_factor_domains())
    except Exception:
        domains = ()
    if "minutes" not in domains:
        return None

    if "pv" in domains:
        try:
            check = validate_expression_against_registry(expr)
            used_fields = {str(field) for field in (check.get("used_fields") or [])}
        except Exception:
            used_fields = set()
        # Use classifier-based validation when expression is available
        joint_check = _validate_joint_pv_minutes_expression_fields(used_fields, expression=expr)
        if not joint_check["ok"]:
            return "joint_domain_missing"

    task = FactorTask(
        factor_name,
        str(factor_data.get("description", "") or ""),
        str(factor_data.get("formulation", "") or ""),
        factor_expression=expr,
    )
    try:
        if compile_expression_to_minute_spec(task, active_domains=domains) is None:
            return "minute_compile_failed"
    except Exception:
        return "minute_compile_failed"
    return None


def _filter_similar_factor_payload(
    response_dict: dict,
    factor_regulator,
    accepted_payload: dict | None = None,
    reserved_factor_names: set[str] | None = None,
) -> tuple[dict, list[str]]:
    if not isinstance(response_dict, dict) or not response_dict:
        return {}, []

    filtered: dict = {}
    skipped: list[str] = []
    seen_exprs: set[str] = set()
    accepted_payload = accepted_payload or {}
    reserved_factor_names = set(reserved_factor_names or set())

    temp_regulator = FactorRegulator(
        factor_zoo_path=None,
        duplication_threshold=factor_regulator.duplication_threshold,
        symbol_length_threshold=factor_regulator.symbol_length_threshold,
        base_features_threshold=factor_regulator.base_features_threshold,
        allowed_functions=factor_regulator.allowed_functions,
        allowed_fields=factor_regulator.allowed_fields,
        domains=factor_regulator.domains,
    )
    temp_regulator.alphazoo = factor_regulator.alphazoo.copy()
    for accepted_name, accepted_data in accepted_payload.items():
        if not isinstance(accepted_data, dict):
            continue
        accepted_expr = str(accepted_data.get("expression", "") or "").strip()
        if not accepted_expr:
            continue
        seen_exprs.add(accepted_expr)
        temp_regulator.add_factor([str(accepted_name)], [accepted_expr])

    for factor_name, factor_data in response_dict.items():
        if not isinstance(factor_data, dict):
            continue
        if str(factor_name) in reserved_factor_names or str(factor_name) in accepted_payload:
            skipped.append(f"{factor_name}: factor_name_already_used")
            continue
        expr = str(factor_data.get("expression", "") or "").strip()
        if not expr:
            skipped.append(f"{factor_name}: empty_expression")
            continue
        if expr in seen_exprs:
            skipped.append(f"{factor_name}: exact_duplicate_expression")
            continue
        runtime_rejection = _candidate_runtime_rejection_reason(str(factor_name), factor_data, expr)
        if runtime_rejection:
            skipped.append(f"{factor_name}: {runtime_rejection}")
            continue
        success, eval_dict = temp_regulator.evaluate(expr)
        if not success or not isinstance(eval_dict, dict):
            skipped.append(f"{factor_name}: evaluate_failed")
            continue
        duplicated_subtree_size = int(eval_dict.get("duplicated_subtree_size", 0) or 0)
        if duplicated_subtree_size > temp_regulator.duplication_threshold:
            matched_alpha = str(eval_dict.get("matched_alpha") or "").strip() or "batch_or_zoo_factor"
            skipped.append(
                f"{factor_name}: duplicated_subtree_size={duplicated_subtree_size}>{temp_regulator.duplication_threshold} "
                f"matched={matched_alpha}"
            )
            continue
        filtered[factor_name] = factor_data
        seen_exprs.add(expr)
        temp_regulator.add_factor([factor_name], [expr])

    return filtered, skipped


def _build_factor_refill_feedback(
    *,
    target_factor_count: int,
    accepted_count: int,
    skipped_similar: list[str],
    accepted_payload: dict | None = None,
) -> str:
    missing_count = max(0, int(target_factor_count) - int(accepted_count))
    skipped_text = "; ".join(skipped_similar) if skipped_similar else "none"
    replacement_word = "replacement" if missing_count == 1 else "replacements"
    accepted_payload = accepted_payload or {}
    accepted_lines = []
    for factor_name, factor_data in accepted_payload.items():
        if not isinstance(factor_data, dict):
            continue
        expr = str(factor_data.get("expression", "") or "").strip()
        accepted_lines.append(f"{factor_name}: {expr}")
    accepted_text = "; ".join(accepted_lines) if accepted_lines else "none"
    return (
        "The previous batch did not provide enough distinct usable factors after validation and "
        f"same-batch similarity filtering: accepted {accepted_count}/{target_factor_count}. "
        f"Already accepted factors that must be kept and must not be regenerated: {accepted_text}. "
        f"Rejected candidates: {skipped_text}. "
        f"Generate exactly {missing_count} new {replacement_word} to fill the missing slots. "
        "Do not replace, rename, or repeat the already accepted factors. The replacements must use "
        "distinct operator families, different primary fields, different lookback windows, and "
        "materially different expression structures from the accepted/rejected candidates. Do not "
        "merely rename factors or only change constants."
    )


def render_hypothesis_and_feedback(prompt_dict, trace: Trace, history_limit: int = DEFAULT_HISTORY_LIMIT) -> str:
    """Render hypothesis_and_feedback with configurable history limit."""
    if len(trace.hist) > 0:
        limited_trace = Trace(scen=trace.scen)
        limited_trace.hist = trace.hist[-history_limit:] if history_limit > 0 else trace.hist
        return (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["hypothesis_and_feedback"])
            .render(trace=limited_trace)
        )
    else:
        return "No previous hypothesis and feedback available since it's the first round."


def _append_current_round_guidance(hypothesis_and_feedback: str, potential_direction: str | None) -> str:
    """Keep mutation/crossover guidance visible even when parent history exists."""
    guidance = str(potential_direction or "").strip()
    if not guidance:
        return hypothesis_and_feedback
    return (
        f"{hypothesis_and_feedback}\n\n"
        "## Current Round Guidance\n"
        f"{guidance}"
    )


def is_input_length_error(error_msg: str) -> bool:
    """Check if error is due to input length limit."""
    error_indicators = [
        "input length",
        "context length", 
        "maximum context",
        "token limit",
        "InvalidParameter",
        "Range of input length",
        "max_tokens",
        "too long"
    ]
    error_str = str(error_msg).lower()
    return any(indicator.lower() in error_str for indicator in error_indicators)


FactorHypothesis = Hypothesis

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _describe_joint_pv_minutes_semantics() -> str:
    describe = getattr(factor_data_domains, "describe_joint_pv_minutes_semantics", None)
    if callable(describe):
        return str(describe())
    return (
        "Joint pv/minutes semantics:\n"
        "- Do not rely on shared fields alone to prove both domains are present.\n"
        "- Actual pv storage fields use plural names `vwaps` and `turnovers`; do not generate singular `vwap` or `turnover`.\n"
        "- In joint runs, `volumes`, `vwaps`, and `turnovers` route by context: inside a minute subtree they are minute data, while top-level declared daily anchors use `data_ctx`.\n"
        "- After minute data has been reduced to a daily 2D frame (`dates x stocks`), regular daily pv operators may be used on that reduced result."
    )


def _build_minute_runtime_preflight_feedback(
    *,
    factor_name: str,
    expression: str,
    called_operators: list[str] | tuple[str, ...] | None,
    active_domains,
) -> str | None:
    domains = factor_data_domains.parse_factor_domains(active_domains)
    if "minutes" not in domains:
        return None
    expr = str(expression or "").strip()
    if not expr:
        return f"Factor `{factor_name}` has an empty expression and must be regenerated."

    minute_fields = set(getattr(factor_data_domains, "MINUTE_FIELDS", ()))
    oversized_lags: list[str] = []
    for op_name, field_name, lag_text in re.findall(
        r"\b(ts_return|pct)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)",
        expr,
    ):
        if field_name in minute_fields and int(lag_text) > MAX_INTRADAY_LAG:
            oversized_lags.append(f"{op_name}({field_name}, {lag_text})")
    if oversized_lags:
        return (
            f"Factor `{factor_name}` uses cross-day sized minute lags that cannot run in the structured minute runtime: "
            + ", ".join(oversized_lags)
            + ". The raw minute axis is an intraday axis, not a continuous multi-month minute series. "
            f"Use intraday lags <= {MAX_INTRADAY_LAG} for raw minute fields, or first reduce minute data to a daily 2D frame "
            "and then apply regular daily operators for multi-day/month horizons."
        )

    stage_mismatch_ops = sorted(
        {
            str(op)
            for op in (called_operators or [])
            if str(op) not in {"WHERE", "where"} and get_minute_op_spec(str(op)) is None
        }
    )
    mismatch_note = ""
    if stage_mismatch_ops:
        mismatch_note = (
            " Operators that appear to require post-reduction daily-frame semantics in this expression: "
            + ", ".join(stage_mismatch_ops)
            + "."
        )
    hard_stage_mismatch_ops = sorted(set(stage_mismatch_ops) & {"winsorize1", "winsorize2", "bucket"})
    if hard_stage_mismatch_ops:
        return (
            f"Factor `{factor_name}` cannot be compiled into the structured minute runtime used by this run. "
            "Do not put daily-frame operators inside raw minute reducers or tensor subtrees. "
            f"Move these operators after the minute feature has been reduced to a daily 2D frame: {', '.join(hard_stage_mismatch_ops)}."
            f"{mismatch_note}"
        )

    try:
        compiled_spec = compile_expression_to_minute_spec(
            FactorTask(factor_name or "candidate_factor", "", "", factor_expression=expr),
            active_domains=domains,
        )
    except Exception as exc:
        compiled_spec = None
        compile_error = str(exc).strip() or exc.__class__.__name__
    else:
        compile_error = None
    if compiled_spec is not None:
        return None

    compile_error_note = f" Compiler detail: {compile_error}." if compile_error else ""
    return (
        f"Factor `{factor_name}` cannot be compiled into the structured minute runtime used by this run. "
        "Regenerate it with a simpler minute shape: use one or two minute features, each with one raw field "
        "or one registered tensor operator inside a reducer (`ts_corr`, `ts_regression`, `mean`, `std`, etc.). "
        "Avoid nested `WHERE`/multi-stage reducer chains inside a single minute feature. Apply daily-frame "
        "operators such as `winsorize1`, `winsorize2`, `bucket`, `ts_delay`, or `cs_*` only after that "
        "minute subtree has been reduced to a daily 2D frame. "
        "`winsorize1(df, n)` is sigma-based; use `winsorize2(lower_bound, upper_bound)` for percentile "
        f"clipping.{mismatch_note}{compile_error_note}"
    )


def _build_expression_parse_feedback(factor_name: str, expr: str, parse_error: str) -> str:
    return (
        f"Factor `{factor_name}` has an invalid factor_expression and must be regenerated.\n"
        f"Failed expression: {expr}\n"
        f"Parse error: {parse_error}\n"
        "Regenerate the expression as complete, syntactically valid expression text. "
        "Do not leave unclosed parentheses or truncated function calls."
    )


def _validate_joint_pv_minutes_expression_fields(used_fields, expression=None):
    validate = getattr(factor_data_domains, "validate_joint_pv_minutes_expression_fields", None)
    if callable(validate):
        return validate(used_fields, expression=expression)
    used = {str(field).strip() for field in (used_fields or []) if str(field).strip()}
    pv_fields = tuple(getattr(factor_data_domains, "PV_FIELDS", ()))
    minute_fields = tuple(getattr(factor_data_domains, "MINUTE_FIELDS", ()))
    shared_fields = set(pv_fields) & set(minute_fields)
    raw_shared_ohlc_fields = {"opens", "highs", "lows", "closes"}
    pv_exclusive = sorted(used & (set(pv_fields) - shared_fields))
    minute_exclusive = sorted(used & (set(minute_fields) - shared_fields))
    raw_shared_ohlc_used = sorted(used & raw_shared_ohlc_fields)
    errors = []
    if not pv_exclusive:
        errors.append("joint pv/minutes expressions must include at least one explicit daily pv field")
    if not minute_exclusive:
        errors.append("joint pv/minutes expressions must include at least one explicit minute field")
    if raw_shared_ohlc_used:
        errors.append(
            "joint pv/minutes expressions must not use ambiguous raw OHLC field names "
            f"{raw_shared_ohlc_used}; use explicit daily hfq_* fields and minute-exclusive fields instead"
        )
    return {
        "ok": len(errors) == 0,
        "used_fields": sorted(used),
        "pv_exclusive_used": pv_exclusive,
        "minute_exclusive_used": minute_exclusive,
        "shared_used": sorted(used & shared_fields),
        "raw_shared_ohlc_used": raw_shared_ohlc_used,
        "errors": errors,
    }


def _format_feedback_items(items, *, limit: int = 80) -> str:
    normalized = sorted({str(item).strip() for item in (items or []) if str(item).strip()})
    if not normalized:
        return "none"
    if len(normalized) <= limit:
        return ", ".join(normalized)
    return ", ".join(normalized[:limit]) + f", ... ({len(normalized) - limit} more)"


def _raw_binary_division_present(expression: str) -> bool:
    expr = str(expression or "").strip()
    if not expr or "/" not in expr:
        return False
    try:
        from quantaalpha.factors.coder.factor_ast import BinaryOpNode, parse_expression

        root = parse_expression(expr)
    except Exception:
        return "/" in expr

    def _walk(node) -> bool:
        if isinstance(node, BinaryOpNode) and node.op == "/":
            return True
        for value in vars(node).values():
            if isinstance(value, list):
                if any(_walk(child) for child in value):
                    return True
            elif hasattr(value, "__dict__") and _walk(value):
                return True
        return False

    return _walk(root)


class DomainProposalPolicy:
    """Domain-specific proposal feedback without changing expression semantics."""

    name = "pv"

    def __init__(self, domains):
        self.domains = factor_data_domains.parse_factor_domains(domains)

    def build_registry_feedback(
        self,
        *,
        factor_name: str,
        expression: str,
        registry_check: dict,
        allowed_operators,
        allowed_fields,
    ) -> str:
        unsupported_operators = registry_check.get("unsupported_operators") or []
        unsupported_fields = registry_check.get("unsupported_fields") or []
        used_fields = registry_check.get("used_fields") or []
        parts = [
            f"Factor `{factor_name}` violates the {self.name} expression registry.",
            f"Failed expression: {expression}",
        ]
        if unsupported_fields:
            parts.append(f"Invalid field(s): {_format_feedback_items(unsupported_fields)}.")
        if unsupported_operators:
            parts.append(
                f"Unsupported operator/function(s): {_format_feedback_items(unsupported_operators)}."
            )
        if used_fields:
            parts.append(f"Fields detected in the expression: {_format_feedback_items(used_fields)}.")
        parts.extend(self._field_allowlist_feedback(allowed_fields))
        parts.append(
            "Allowed operators/functions: "
            + _format_feedback_items(allowed_operators, limit=120)
            + "."
        )
        if any(str(op).strip() == "sign" for op in unsupported_operators):
            parts.append(
                "Repair hint for sign(x): rewrite it with registered primitives, for example "
                "`if_else(gt(x, 0), 1, if_else(lt(x, 0), -1, 0))`."
            )
        parts.append(
            "Regenerate the expression using exact field/operator names from these allowlists. "
            "Do not singularize plural fields such as `volumes`, `vwaps`, or `turnovers`, and do not invent aliases."
        )
        return "\n".join(parts)

    def build_preflight_feedback(
        self,
        *,
        factor_name: str,
        expression: str,
        registry_check: dict | None = None,
    ) -> str | None:
        return None

    def _field_allowlist_feedback(self, allowed_fields) -> list[str]:
        return ["Allowed fields: " + _format_feedback_items(allowed_fields, limit=120) + "."]


class PVProposalPolicy(DomainProposalPolicy):
    name = "PV"


class MinuteProposalPolicy(DomainProposalPolicy):
    name = "minute"

    def build_preflight_feedback(
        self,
        *,
        factor_name: str,
        expression: str,
        registry_check: dict | None = None,
    ) -> str | None:
        safety_feedback = self._build_numerical_safety_feedback(factor_name, expression)
        if safety_feedback is not None:
            return safety_feedback
        called_operators = []
        if isinstance(registry_check, dict):
            called_operators = registry_check.get("called_operators", [])
        return _build_minute_runtime_preflight_feedback(
            factor_name=factor_name,
            expression=expression,
            called_operators=called_operators,
            active_domains=self.domains,
        )

    def _field_allowlist_feedback(self, allowed_fields) -> list[str]:
        minute_fields = getattr(factor_data_domains, "MINUTE_FIELDS", ())
        return ["Allowed minute fields: " + _format_feedback_items(minute_fields) + "."]

    def _build_numerical_safety_feedback(self, factor_name: str, expression: str) -> str | None:
        if not _raw_binary_division_present(expression):
            return None
        return (
            f"Factor `{factor_name}` has a minute-domain Numerical safety issue.\n"
            f"Failed expression: {expression}\n"
            "The expression uses raw `/` on data-derived arrays. In minute tensors this can create inf/nan values "
            "when the denominator is zero or all-null, which later shows up as runtime warnings or unstable factor data.\n"
            "Regenerate with `safe_div(numerator, denominator, 1e-8)` for the ratio, then apply the remaining "
            "registered transforms. Example shape: `mul(safe_div(turnovers, ts_mean(turnovers, 30), 1e-8), ts_std(returns, 30))`."
        )


class JointProposalPolicy(MinuteProposalPolicy):
    name = "joint PV/minute"

    def build_preflight_feedback(
        self,
        *,
        factor_name: str,
        expression: str,
        registry_check: dict | None = None,
    ) -> str | None:
        if isinstance(registry_check, dict):
            joint_expression_check = _validate_joint_pv_minutes_expression_fields(
                registry_check.get("used_fields", []),
                expression=expression,
            )
            if not joint_expression_check["ok"]:
                shared_info = joint_expression_check.get("shared_field_usage", {})
                shared_detail = (
                    f"\nShared field domain classification: {shared_info}"
                    if shared_info
                    else ""
                )
                return (
                    f"Factor `{factor_name}` violates the joint PV/minute field contract.\n"
                    f"Failed expression: {expression}\n"
                    + "; ".join(joint_expression_check["errors"])
                    + f"\nFields detected: {_format_feedback_items(joint_expression_check['used_fields'])}.\n"
                    + "Explicit daily PV fields currently present: "
                    + _format_feedback_items(joint_expression_check.get("pv_exclusive_used", []))
                    + ".\nExplicit minute fields currently present: "
                    + _format_feedback_items(joint_expression_check.get("minute_exclusive_used", []))
                    + ".\nShared-only fields: "
                    + _format_feedback_items(joint_expression_check.get("shared_used", []))
                    + shared_detail
                    + ".\nRegenerate with at least one explicit daily PV anchor such as `hfq_closes`, `hfq_opens`, "
                    "`hfq_highs`, `hfq_lows`, or shared fields used in daily context (e.g., `cs_zscore(turnovers)`), "
                    "and at least one minute-domain operation such as `ts_zscore(field, window)`, `ts_mean(field, window)`, "
                    "etc. Avoid ambiguous raw OHLC names for daily PV; use `hfq_*` there."
                )
        return super().build_preflight_feedback(
            factor_name=factor_name,
            expression=expression,
            registry_check=registry_check,
        )

    def _field_allowlist_feedback(self, allowed_fields) -> list[str]:
        pv_fields = getattr(factor_data_domains, "PV_FIELDS", ())
        minute_fields = getattr(factor_data_domains, "MINUTE_FIELDS", ())
        shared_fields = sorted(set(pv_fields) & set(minute_fields))
        return [
            "Allowed daily PV fields: " + _format_feedback_items(pv_fields) + ".",
            "Allowed minute fields: " + _format_feedback_items(minute_fields) + ".",
            "Shared names that must still be routed by context: "
            + _format_feedback_items(shared_fields)
            + ".",
        ]


def _get_domain_proposal_policy(domains) -> DomainProposalPolicy:
    parsed_domains = factor_data_domains.parse_factor_domains(domains)
    if "pv" in parsed_domains and "minutes" in parsed_domains:
        return JointProposalPolicy(parsed_domains)
    if "minutes" in parsed_domains:
        return MinuteProposalPolicy(parsed_domains)
    return PVProposalPolicy(parsed_domains)


class AlphaAgentHypothesis(Hypothesis):
    """
    AlphaAgentHypothesis extends the Hypothesis class to include a potential_direction,
    which represents the initial idea or starting point for the hypothesis.
    """

    def __init__(
        self,
        hypothesis: str,
        concise_observation: str,
        concise_justification: str,
        concise_knowledge: str,
        concise_specification: str
    ) -> None:
        super().__init__(
            hypothesis,
            "",
            "",
            concise_observation,
            concise_justification,
            concise_knowledge,
        )
        self.concise_specification = concise_specification
        
    def __str__(self) -> str:
        return f"""Hypothesis: {self.hypothesis}
                Concise Observation: {self.concise_observation}
                Concise Justification: {self.concise_justification}
                Concise Knowledge: {self.concise_knowledge}
                concise Specification: {self.concise_specification}
                """

base_prompt_dict = DomainPromptProxy(base_file="prompts.yaml", prompt_dir=_PROMPTS_DIR)

class FactorMiningHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)

    def prepare_context(self, trace: Trace) -> Tuple[dict, bool]:
        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(base_prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )
        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": None,
            "hypothesis_output_format": base_prompt_dict["hypothesis_output_format"],
            "hypothesis_specification": base_prompt_dict["factor_hypothesis_specification"],
        }
        return context_dict, True

    def convert_response(self, response: str) -> Hypothesis:
        response_dict = robust_json_parse(response)
        hypothesis = FactorHypothesis(
            hypothesis=response_dict.get("hypothesis", ""),
            reason=response_dict.get("reason", ""),
            concise_reason=response_dict.get("concise_reason", ""),
            concise_observation=response_dict.get("concise_observation", ""),
            concise_justification=response_dict.get("concise_justification", ""),
            concise_knowledge=response_dict.get("concise_knowledge", ""),
        )
        return hypothesis


class FactorMiningHypothesis2Experiment(FactorHypothesis2Experiment):
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace) -> Tuple[dict | bool]:
        scenario = trace.scen.get_scenario_all_desc()
        experiment_output_format = base_prompt_dict["factor_experiment_output_format"]

        hypothesis_and_feedback = (
            (
                Environment(undefined=StrictUndefined)
                .from_string(base_prompt_dict["hypothesis_and_feedback"])
                .render(trace=trace)
            )
            if len(trace.hist) > 0
            else "No previous hypothesis and feedback available since it's the first round."
        )

        experiment_list: List[FactorExperiment] = [t[1] for t in trace.hist]

        factor_list = []
        for experiment in experiment_list:
            factor_list.extend(experiment.sub_tasks)

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "experiment_output_format": experiment_output_format,
            "target_list": factor_list,
            "RAG": None,
        }, True

    def convert_response(self, response: str, trace: Trace) -> FactorExperiment:
        response_dict = robust_json_parse(response)
        tasks = []

        for factor_name in response_dict:
            factor_data = response_dict.get(factor_name, {})
            if not isinstance(factor_data, dict):
                continue
            description = factor_data.get("description", "")
            formulation = factor_data.get("formulation", "")
            # expression = factor_data.get("expression", "")
            variables = factor_data.get("variables", {})
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    # factor_expression=expression,
                    variables=variables,
                )
            )

        exp = FactorMiningExperiment(tasks)
        exp.based_experiments = [FactorMiningExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]

        unique_tasks = []

        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        exp.tasks = unique_tasks
        return exp



qa_prompt_dict = DomainPromptProxy(base_file="prompts.yaml", prompt_dir=_PROMPTS_DIR)

# prompt_dict not as attribute: class instance is pickled later, prompt_dict cannot be pickled
class AlphaAgentHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario, potential_direction: str=None) -> Tuple[dict, bool]:
        super().__init__(scen)
        self.potential_direction = potential_direction

    def prepare_context(self, trace: Trace, history_limit: int = DEFAULT_HISTORY_LIMIT) -> Tuple[dict, bool]:
        
        if len(trace.hist) > 0:
            hypothesis_and_feedback = render_hypothesis_and_feedback(
                qa_prompt_dict, trace, history_limit
            )
            hypothesis_and_feedback = _append_current_round_guidance(
                hypothesis_and_feedback,
                self.potential_direction,
            )
            
        elif self.potential_direction is not None: 
            hypothesis_and_feedback = (
                Environment(undefined=StrictUndefined)
                .from_string(qa_prompt_dict["potential_direction_transformation"])
                .render(potential_direction=self.potential_direction)
            ) # 
        else:
            hypothesis_and_feedback = "No previous hypothesis and feedback available since it's the first round. You are encouraged to propose an innovative hypothesis that diverges significantly from existing perspectives."
            
        context_dict = {
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "RAG": None,
            "hypothesis_output_format": qa_prompt_dict["hypothesis_output_format"],
            "hypothesis_specification": qa_prompt_dict["factor_hypothesis_specification"],
        }
        return context_dict, True

    def convert_response(self, response: str) -> AlphaAgentHypothesis:
        """
        Convert LLM JSON to AlphaAgentHypothesis; use default empty string for missing fields to avoid KeyError.
        """
        response_dict = robust_json_parse(response)
        # Use get to avoid KeyError on missing fields
        hypothesis = AlphaAgentHypothesis(
            hypothesis=response_dict.get("hypothesis", ""),
            concise_observation=response_dict.get("concise_observation", ""),
            concise_knowledge=response_dict.get("concise_knowledge", ""),
            concise_justification=response_dict.get("concise_justification", ""),
            concise_specification=response_dict.get("concise_specification", ""),
        )
        return hypothesis
    
    def gen(self, trace: Trace) -> AlphaAgentHypothesis:
        """Generate hypothesis; supports dynamic history limit for input length."""
        history_limit = DEFAULT_HISTORY_LIMIT
        
        while history_limit >= MIN_HISTORY_LIMIT:
            try:
                context_dict, json_flag = self.prepare_context(trace, history_limit)
                system_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(qa_prompt_dict["hypothesis_gen"]["system_prompt"])
                    .render(
                        targets=self.targets,
                        scenario=self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
                        hypothesis_output_format=context_dict["hypothesis_output_format"],
                        hypothesis_specification=context_dict["hypothesis_specification"],
                    )
                )
                user_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(qa_prompt_dict["hypothesis_gen"]["user_prompt"])
                    .render(
                        targets=self.targets,
                        hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
                        RAG=context_dict["RAG"],
                        round=len(trace.hist)
                    )
                )

                resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)
                hypothesis = self.convert_response(resp)
                return hypothesis
            
            except Exception as e:
                if is_input_length_error(str(e)) and history_limit > MIN_HISTORY_LIMIT:
                    history_limit -= 1
                    logger.warning(f"Input length exceeded, retrying with history_limit={history_limit}...")
                else:
                    raise
        
        # Last attempt with minimum history limit
        context_dict, json_flag = self.prepare_context(trace, MIN_HISTORY_LIMIT)
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_prompt_dict["hypothesis_gen"]["system_prompt"])
            .render(
                targets=self.targets,
                scenario=self.scen.get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
                hypothesis_output_format=context_dict["hypothesis_output_format"],
                hypothesis_specification=context_dict["hypothesis_specification"],
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_prompt_dict["hypothesis_gen"]["user_prompt"])
            .render(
                targets=self.targets,
                hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
                RAG=context_dict["RAG"],
                round=len(trace.hist)
            )
        )
        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)
        hypothesis = self.convert_response(resp)
        return hypothesis
    
    

class EmptyHypothesisGen(FactorHypothesisGen):
    def __init__(self, scen: Scenario) -> Tuple[dict, bool]:
        super().__init__(scen)
        
    def convert_response(self, *args, **kwargs) -> AlphaAgentHypothesis: 
        return super().convert_response(*args, **kwargs)  
    
    def prepare_context(self, *args, **kwargs) -> Tuple[dict | bool]:
        return super().prepare_context(*args, **kwargs)

    def gen(self, trace: Trace) -> AlphaAgentHypothesis:

        hypothesis = AlphaAgentHypothesis(
            hypothesis="",
            concise_observation="",
            concise_justification="",
            concise_knowledge="",
            concise_specification=""
        )

        return hypothesis




class AlphaAgentHypothesis2FactorExpression(FactorHypothesis2Experiment):
    def __init__(
        self,
        *args,
        consistency_enabled: bool = False,
        consistency_strict_mode: bool = False,
        max_correction_attempts: int = 3,
        complexity_enabled: bool = True,
        redundancy_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Initialize FactorRegulator with config settings
        from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
        active_domains = factor_data_domains.resolve_factor_domains()
        self.factor_regulator = FactorRegulator(
            factor_zoo_path=FACTOR_COSTEER_SETTINGS.factor_zoo_path,
            duplication_threshold=FACTOR_COSTEER_SETTINGS.duplication_threshold,
            allowed_functions=get_allowed_operator_names(active_domains),
            allowed_fields=get_allowed_field_names(active_domains),
            domains=active_domains,
        )
        self.target_factor_count = _resolve_target_factor_count()
        
        # Initialize consistency checker if enabled
        self.consistency_enabled = consistency_enabled
        self.consistency_strict_mode = bool(consistency_strict_mode)
        self.max_correction_attempts = max(1, int(max_correction_attempts))
        self.complexity_enabled = bool(complexity_enabled)
        self.redundancy_enabled = bool(redundancy_enabled)
        self._quality_gate = None
        
    @property
    def quality_gate(self):
        """Lazy-load FactorQualityGate."""
        if self._quality_gate is None and self.consistency_enabled:
            try:
                from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
                from quantaalpha.factors.regulator.consistency_checker import (
                    ComplexityChecker,
                    FactorConsistencyChecker,
                    FactorQualityGate,
                    RedundancyChecker,
                )

                self._quality_gate = FactorQualityGate(
                    consistency_checker=FactorConsistencyChecker(
                        enabled=self.consistency_enabled,
                        strict_mode=self.consistency_strict_mode,
                        max_correction_attempts=self.max_correction_attempts,
                    ),
                    complexity_checker=ComplexityChecker(
                        enabled=self.complexity_enabled,
                        symbol_length_threshold=FACTOR_COSTEER_SETTINGS.symbol_length_threshold,
                        base_features_threshold=FACTOR_COSTEER_SETTINGS.base_features_threshold,
                    ),
                    redundancy_checker=RedundancyChecker(
                        enabled=self.redundancy_enabled,
                        duplication_threshold=self.factor_regulator.duplication_threshold,
                        factor_zoo_path=FACTOR_COSTEER_SETTINGS.factor_zoo_path,
                    ),
                    consistency_enabled=self.consistency_enabled,
                    complexity_enabled=self.complexity_enabled,
                    redundancy_enabled=self.redundancy_enabled,
                )
            except ImportError as e:
                logger.warning(f"Could not load consistency checker: {e}")
                self._quality_gate = None
        return self._quality_gate
        
    def prepare_context(self, hypothesis: Hypothesis, trace: Trace, history_limit: int = DEFAULT_HISTORY_LIMIT) -> Tuple[dict | bool]:
        scenario = trace.scen.get_scenario_all_desc()
        experiment_output_format = qa_prompt_dict["factor_experiment_output_format"]
        function_lib_description = _build_function_lib_description(qa_prompt_dict['function_lib_description'])
        hypothesis_and_feedback = render_hypothesis_and_feedback(
            qa_prompt_dict, trace, history_limit
        )

        experiment_list: List[FactorExperiment] = [t[1] for t in trace.hist]

        factor_list = []
        for experiment in experiment_list:
            factor_list.extend(experiment.sub_tasks)

        return {
            "target_hypothesis": str(hypothesis),
            "scenario": scenario,
            "hypothesis_and_feedback": hypothesis_and_feedback,
            "function_lib_description": function_lib_description,
            "experiment_output_format": experiment_output_format,
            "target_list": factor_list,
            "target_factor_count": self.target_factor_count,
            "RAG": None,
        }, True
        
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> Experiment:
        """Convert hypothesis to factor expressions; supports dynamic history limit."""
        history_limit = DEFAULT_HISTORY_LIMIT
        
        while history_limit >= MIN_HISTORY_LIMIT:
            try:
                return self._convert_with_history_limit(hypothesis, trace, history_limit)
            except Exception as e:
                if is_input_length_error(str(e)) and history_limit > MIN_HISTORY_LIMIT:
                    history_limit -= 1
                    logger.warning(f"Input length exceeded, retrying with history_limit={history_limit}...")
                else:
                    raise
        
        # Last attempt with minimum history limit
        return self._convert_with_history_limit(hypothesis, trace, MIN_HISTORY_LIMIT)
    
    def _convert_with_history_limit(self, hypothesis: Hypothesis, trace: Trace, history_limit: int) -> Experiment:
        """Convert with given history limit."""
        context, json_flag = self.prepare_context(hypothesis, trace, history_limit)
        requested_factor_count = self.target_factor_count
        reserved_factor_names = {
            str(task.factor_name)
            for task in context.get("target_list", [])
            if getattr(task, "factor_name", None)
        }
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_prompt_dict["hypothesis2experiment"]["system_prompt"])
            .render(
                targets=self.targets,
                scenario=trace.scen.background, # get_scenario_all_desc(filtered_tag="hypothesis_and_experiment"),
                experiment_output_format=context["experiment_output_format"],
                target_factor_count=requested_factor_count,
            )
        )
        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
            .render(
                targets=self.targets,
                target_hypothesis=context["target_hypothesis"],
                hypothesis_and_feedback=context["hypothesis_and_feedback"],
                function_lib_description=context["function_lib_description"],
                target_list=context["target_list"],
                RAG=context["RAG"], 
                target_factor_count=requested_factor_count,
                expression_duplication=None
            )
        )
        
        # Detect duplicated sub-expressions
        flag = False
        expression_duplication_prompt = None
        quality_gate_prompt = None
        accepted_response_dict: dict = {}
        refill_attempts = 0
        max_refill_attempts = _resolve_max_refill_attempts(self.target_factor_count)
        domain_policy = _get_domain_proposal_policy(self.factor_regulator.domains)
        while True:
            if flag:
                break

            resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=json_flag)
            try:
                response_dict = robust_json_parse(resp)
                response_dict = _trim_factor_payload(response_dict, requested_factor_count)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed: {e}, retrying...")
                continue
            proposed_names = []
            proposed_exprs = []
            
            for i, factor_name in enumerate(response_dict):
                factor_data = response_dict.get(factor_name, {})
                if not isinstance(factor_data, dict):
                    continue
                expr = factor_data.get("expression", "")
                description = factor_data.get("description", "")
                formulation = factor_data.get("formulation", "")
                variables = factor_data.get("variables", {})

                registry_check = validate_expression_against_registry(
                    expr,
                    allowed_operators=self.factor_regulator.allowed_functions,
                    allowed_fields=self.factor_regulator.allowed_fields,
                )
                domain_coverage = factor_data_domains.validate_multi_domain_field_coverage(
                    registry_check["used_fields"],
                    self.factor_regulator.domains,
                )
                if not registry_check["ok"]:
                    whitelist_feedback = domain_policy.build_registry_feedback(
                        factor_name=factor_name,
                        expression=expr,
                        registry_check=registry_check,
                        allowed_operators=self.factor_regulator.allowed_functions,
                        allowed_fields=self.factor_regulator.allowed_fields,
                    )
                    if expression_duplication_prompt is not None:
                        expression_duplication_prompt = "\n\n".join([expression_duplication_prompt, whitelist_feedback])
                    else:
                        expression_duplication_prompt = whitelist_feedback

                    user_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                        .render(
                            targets=self.targets,
                            target_hypothesis=context["target_hypothesis"],
                            hypothesis_and_feedback=context["hypothesis_and_feedback"],
                            function_lib_description=context["function_lib_description"],
                            target_list=context["target_list"],
                            RAG=context["RAG"],
                            target_factor_count=requested_factor_count,
                            expression_duplication=expression_duplication_prompt,
                        )
                    )
                    break

                if not domain_coverage["ok"]:
                    domain_field_map = factor_data_domains.get_domain_field_map(self.factor_regulator.domains)
                    coverage_feedback = (
                        f"Factor `{factor_name}` must use at least one field from every active domain. "
                        f"Missing domains: {', '.join(domain_coverage['missing_domains'])}. "
                        f"Current used fields: {', '.join(domain_coverage['used_fields']) or 'none'}. "
                        "Active domain field requirements: "
                        + "; ".join(
                            f"{domain} -> {', '.join(fields)}"
                            for domain, fields in domain_field_map.items()
                        )
                        + ". Regenerate a mixed-domain expression that explicitly includes all active domains."
                    )
                    if expression_duplication_prompt is not None:
                        expression_duplication_prompt = "\n\n".join([expression_duplication_prompt, coverage_feedback])
                    else:
                        expression_duplication_prompt = coverage_feedback

                    user_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                        .render(
                            targets=self.targets,
                            target_hypothesis=context["target_hypothesis"],
                            hypothesis_and_feedback=context["hypothesis_and_feedback"],
                            function_lib_description=context["function_lib_description"],
                            target_list=context["target_list"],
                            RAG=context["RAG"],
                            target_factor_count=requested_factor_count,
                            expression_duplication=expression_duplication_prompt,
                        )
                    )
                    break

                domain_policy_feedback = domain_policy.build_preflight_feedback(
                    factor_name=factor_name,
                    expression=expr,
                    registry_check=registry_check,
                )
                if domain_policy_feedback is not None:
                    if expression_duplication_prompt is not None:
                        expression_duplication_prompt = "\n\n".join(
                            [expression_duplication_prompt, domain_policy_feedback]
                        )
                    else:
                        expression_duplication_prompt = domain_policy_feedback

                    user_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                        .render(
                            targets=self.targets,
                            target_hypothesis=context["target_hypothesis"],
                            hypothesis_and_feedback=context["hypothesis_and_feedback"],
                            function_lib_description=context["function_lib_description"],
                            target_list=context["target_list"],
                            RAG=context["RAG"],
                            target_factor_count=requested_factor_count,
                            expression_duplication=expression_duplication_prompt,
                        )
                    )
                    break

                if "pv" in self.factor_regulator.domains and "minutes" in self.factor_regulator.domains:
                    joint_expression_check = _validate_joint_pv_minutes_expression_fields(
                        registry_check["used_fields"]
                    )
                    if not joint_expression_check["ok"]:
                        semantics_feedback = (
                            f"Factor `{factor_name}` violates the joint pv/minutes field contract. "
                            + "; ".join(joint_expression_check["errors"])
                            + f". Current used fields: {', '.join(joint_expression_check['used_fields']) or 'none'}. "
                            + f"Explicit daily pv fields currently present: {', '.join(joint_expression_check['pv_exclusive_used']) or 'none'}. "
                            + f"Explicit minute fields currently present: {', '.join(joint_expression_check['minute_exclusive_used']) or 'none'}. "
                            + f"Shared-only fields: {', '.join(joint_expression_check['shared_used']) or 'none'}. "
                            + "Regenerate the expression with explicit routing semantics, using actual fields such as daily `hfq_*` or declared shared `vwaps/turnovers`, and minute `vwaps/turnovers/returns/...` inside minute subtrees."
                        )
                        if expression_duplication_prompt is not None:
                            expression_duplication_prompt = "\n\n".join([expression_duplication_prompt, semantics_feedback])
                        else:
                            expression_duplication_prompt = semantics_feedback

                        user_prompt = (
                            Environment(undefined=StrictUndefined)
                            .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                            .render(
                                targets=self.targets,
                                target_hypothesis=context["target_hypothesis"],
                                hypothesis_and_feedback=context["hypothesis_and_feedback"],
                                function_lib_description=context["function_lib_description"],
                                target_list=context["target_list"],
                                RAG=context["RAG"],
                                target_factor_count=requested_factor_count,
                                expression_duplication=expression_duplication_prompt,
                            )
                        )
                        break

                minute_runtime_feedback = _build_minute_runtime_preflight_feedback(
                    factor_name=factor_name,
                    expression=expr,
                    called_operators=registry_check.get("called_operators", []),
                    active_domains=self.factor_regulator.domains,
                )
                if minute_runtime_feedback is not None:
                    if expression_duplication_prompt is not None:
                        expression_duplication_prompt = "\n\n".join(
                            [expression_duplication_prompt, minute_runtime_feedback]
                        )
                    else:
                        expression_duplication_prompt = minute_runtime_feedback

                    user_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                        .render(
                            targets=self.targets,
                            target_hypothesis=context["target_hypothesis"],
                            hypothesis_and_feedback=context["hypothesis_and_feedback"],
                            function_lib_description=context["function_lib_description"],
                            target_list=context["target_list"],
                            RAG=context["RAG"],
                            target_factor_count=requested_factor_count,
                            expression_duplication=expression_duplication_prompt,
                        )
                    )
                    break

                # Deterministic operator whitelist check: unsupported operators must be regenerated.
                operators_allowed, unsupported_functions = self.factor_regulator.validate_allowed_functions(expr)
                if not operators_allowed:
                    parse_errors = [
                        str(item).removeprefix("<parse_error:").removesuffix(">")
                        for item in unsupported_functions
                        if str(item).startswith("<parse_error:")
                    ]
                    if parse_errors:
                        whitelist_feedback = _build_expression_parse_feedback(
                            factor_name,
                            expr,
                            "; ".join(parse_errors),
                        )
                    else:
                        unsupported_text = ", ".join(unsupported_functions)
                        whitelist_feedback = (
                            f"Factor `{factor_name}` uses unsupported operators/functions: {unsupported_text}. "
                            "Regenerate expression using ONLY operators listed in function_lib_description."
                        )
                    if expression_duplication_prompt is not None:
                        expression_duplication_prompt = "\n\n".join([expression_duplication_prompt, whitelist_feedback])
                    else:
                        expression_duplication_prompt = whitelist_feedback

                    user_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                        .render(
                            targets=self.targets,
                            target_hypothesis=context["target_hypothesis"],
                            hypothesis_and_feedback=context["hypothesis_and_feedback"],
                            function_lib_description=context["function_lib_description"],
                            target_list=context["target_list"],
                            RAG=context["RAG"],
                            target_factor_count=requested_factor_count,
                            expression_duplication=expression_duplication_prompt,
                        )
                    )
                    break
                
                # Check if expression is parsable
                if not self.factor_regulator.is_parsable(expr):
                    parse_error = self.factor_regulator.get_parse_error(expr) or "unknown parse error"
                    logger.info(f"Failed to parse expr: {expr}, retrying with feedback: {parse_error}")
                    parse_feedback = _build_expression_parse_feedback(factor_name, expr, parse_error)
                    if expression_duplication_prompt is not None:
                        expression_duplication_prompt = "\n\n".join([expression_duplication_prompt, parse_feedback])
                    else:
                        expression_duplication_prompt = parse_feedback

                    user_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                        .render(
                            targets=self.targets,
                            target_hypothesis=context["target_hypothesis"],
                            hypothesis_and_feedback=context["hypothesis_and_feedback"],
                            function_lib_description=context["function_lib_description"],
                            target_list=context["target_list"],
                            RAG=context["RAG"],
                            target_factor_count=requested_factor_count,
                            expression_duplication=expression_duplication_prompt,
                        )
                    )
                    break
                
                success, eval_dict = self.factor_regulator.evaluate(expr)
                if not success:
                    break
                
                # Consistency check (if enabled)
                if self.consistency_enabled and self.quality_gate is not None:
                    try:
                        passed, feedback, results = self.quality_gate.evaluate(
                            hypothesis=str(hypothesis),
                            factor_name=factor_name,
                            factor_description=description,
                            factor_formulation=formulation,
                            factor_expression=expr,
                            variables=variables
                        )
                        
                        # Apply corrections suggested by the consistency gate.
                        if results.get("corrected_expression") and results["corrected_expression"] != expr:
                            logger.info(f"Consistency check corrected expression: {expr} -> {results['corrected_expression']}")
                            expr = results["corrected_expression"]
                            factor_data["expression"] = expr
                            operators_allowed, unsupported_functions = self.factor_regulator.validate_allowed_functions(expr)
                            if not operators_allowed:
                                parse_errors = [
                                    str(item).removeprefix("<parse_error:").removesuffix(">")
                                    for item in unsupported_functions
                                    if str(item).startswith("<parse_error:")
                                ]
                                if parse_errors:
                                    whitelist_feedback = _build_expression_parse_feedback(
                                        factor_name,
                                        expr,
                                        "; ".join(parse_errors),
                                    )
                                else:
                                    unsupported_text = ", ".join(unsupported_functions)
                                    whitelist_feedback = (
                                        f"Factor `{factor_name}` uses unsupported operators/functions after correction: {unsupported_text}. "
                                        "Regenerate expression using ONLY operators listed in function_lib_description."
                                    )
                                if expression_duplication_prompt is not None:
                                    expression_duplication_prompt = "\n\n".join([expression_duplication_prompt, whitelist_feedback])
                                else:
                                    expression_duplication_prompt = whitelist_feedback

                                user_prompt = (
                                    Environment(undefined=StrictUndefined)
                                    .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                                    .render(
                                        targets=self.targets,
                                        target_hypothesis=context["target_hypothesis"],
                                        hypothesis_and_feedback=context["hypothesis_and_feedback"],
                                        function_lib_description=context["function_lib_description"],
                                        target_list=context["target_list"],
                                        RAG=context["RAG"],
                                        target_factor_count=requested_factor_count,
                                        expression_duplication=expression_duplication_prompt,
                                    )
                                )
                                break
                            
                            # Re-check corrected expression
                            if not self.factor_regulator.is_parsable(expr):
                                parse_error = self.factor_regulator.get_parse_error(expr) or "unknown parse error"
                                logger.warning(f"Corrected expression could not be parsed: {expr}; {parse_error}")
                                parse_feedback = _build_expression_parse_feedback(factor_name, expr, parse_error)
                                if expression_duplication_prompt is not None:
                                    expression_duplication_prompt = "\n\n".join([expression_duplication_prompt, parse_feedback])
                                else:
                                    expression_duplication_prompt = parse_feedback

                                user_prompt = (
                                    Environment(undefined=StrictUndefined)
                                    .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                                    .render(
                                        targets=self.targets,
                                        target_hypothesis=context["target_hypothesis"],
                                        hypothesis_and_feedback=context["hypothesis_and_feedback"],
                                        function_lib_description=context["function_lib_description"],
                                        target_list=context["target_list"],
                                        RAG=context["RAG"],
                                        target_factor_count=requested_factor_count,
                                        expression_duplication=expression_duplication_prompt,
                                    )
                                )
                                break
                            success, eval_dict = self.factor_regulator.evaluate(expr)
                            if not success:
                                break
                        if results.get("corrected_description") and results["corrected_description"] != description:
                            logger.info(f"Consistency check corrected description for {factor_name}")
                            description = results["corrected_description"]
                            factor_data["description"] = description
                        if results.get("corrected_formulation") and results["corrected_formulation"] != formulation:
                            logger.info(f"Consistency check corrected formulation for {factor_name}")
                            formulation = results["corrected_formulation"]
                            factor_data["formulation"] = formulation
                        response_dict[factor_name] = factor_data
                        
                        if not passed:
                            logger.warning(f"Quality gate failed: {factor_name}, feedback: {feedback}")
                            quality_feedback_item = (
                                f"Factor `{factor_name}` failed quality gate and must be regenerated.\n"
                                f"Failed expression: {expr}\n"
                                f"Failure reason: {feedback}"
                            )
                            if quality_gate_prompt is not None:
                                quality_gate_prompt = "\n\n".join([quality_gate_prompt, quality_feedback_item])
                            else:
                                quality_gate_prompt = quality_feedback_item

                            regen_feedback = quality_gate_prompt
                            if expression_duplication_prompt:
                                regen_feedback = "\n\n".join([expression_duplication_prompt, quality_gate_prompt])

                            user_prompt = (
                                Environment(undefined=StrictUndefined)
                                .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                                .render(
                                    targets=self.targets,
                                    target_hypothesis=context["target_hypothesis"],
                                    hypothesis_and_feedback=context["hypothesis_and_feedback"],
                                    function_lib_description=context["function_lib_description"],
                                    target_list=context["target_list"],
                                    RAG=context["RAG"],
                                    target_factor_count=requested_factor_count,
                                    expression_duplication=regen_feedback,
                                )
                            )
                            break
                    except Exception as e:
                        logger.warning(f"Consistency check error: {e}")
                
                # If expression has problems, regenerate with feedback
                if not self.factor_regulator.is_expression_acceptable(eval_dict):
                    # Calculate ratios for feedback
                    num_all_nodes = eval_dict['num_all_nodes']
                    free_args_ratio = float(eval_dict['num_free_args']) / float(num_all_nodes) if num_all_nodes > 0 else 0.0
                    unique_vars_ratio = float(eval_dict['num_unique_vars']) / float(num_all_nodes) if num_all_nodes > 0 else 0.0
                    
                    # Get symbol length and base features count for complexity feedback
                    symbol_length = eval_dict.get('symbol_length', 0)
                    num_base_features = eval_dict.get('num_base_features', 0)
                    symbol_length_threshold = self.factor_regulator.symbol_length_threshold
                    base_features_threshold = self.factor_regulator.base_features_threshold
                    
                    feedback_item = (
                            Environment(undefined=StrictUndefined)
                            .from_string(qa_prompt_dict["expression_duplication"])
                            .render(
                                prev_expression=expr,
                                duplicated_subtree_size=eval_dict['duplicated_subtree_size'],
                            duplication_threshold=self.factor_regulator.duplication_threshold,
                            duplicated_subtree=eval_dict.get('duplicated_subtree', ''),
                            matched_alpha=eval_dict.get('matched_alpha', ''),
                            free_args_ratio=free_args_ratio,
                            num_free_args=eval_dict['num_free_args'],
                            unique_vars_ratio=unique_vars_ratio,
                            num_unique_vars=eval_dict['num_unique_vars'],
                            num_all_nodes=num_all_nodes,
                            symbol_length=symbol_length,
                            symbol_length_threshold=symbol_length_threshold,
                            num_base_features=num_base_features,
                            base_features_threshold=base_features_threshold
                            )
                        )
                    
                    if expression_duplication_prompt is not None:
                        expression_duplication_prompt = '\n\n'.join([expression_duplication_prompt, feedback_item])
                    else:
                        expression_duplication_prompt = feedback_item
                    
                    user_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                        .render(
                            targets=self.targets,
                            target_hypothesis=context["target_hypothesis"],
                            hypothesis_and_feedback=context["hypothesis_and_feedback"],
                            function_lib_description=context["function_lib_description"],
                            target_list=context["target_list"],
                            RAG=context["RAG"], 
                            target_factor_count=requested_factor_count,
                            expression_duplication=expression_duplication_prompt
                        )
                    )
                    break
                else:
                    proposed_names.append(factor_name)
                    proposed_exprs.append(expr)
                    if i == len(response_dict) - 1:
                        flag = True
                    else:
                        continue
        

            filtered_response_dict, skipped_similar = _filter_similar_factor_payload(
                response_dict,
                self.factor_regulator,
                accepted_payload=accepted_response_dict,
                reserved_factor_names=reserved_factor_names,
            )
            if skipped_similar:
                logger.info(
                    "Filtered similar factor candidates within the same batch: "
                    + "; ".join(skipped_similar)
                )

            for factor_name, factor_data in filtered_response_dict.items():
                if len(accepted_response_dict) >= self.target_factor_count:
                    break
                if factor_name in accepted_response_dict:
                    continue
                accepted_response_dict[factor_name] = factor_data

            if len(accepted_response_dict) < self.target_factor_count and refill_attempts < max_refill_attempts:
                refill_attempts += 1
                requested_factor_count = _resolve_refill_request_count(
                    self.target_factor_count,
                    len(accepted_response_dict),
                )
                refill_feedback = _build_factor_refill_feedback(
                    target_factor_count=self.target_factor_count,
                    accepted_count=len(accepted_response_dict),
                    skipped_similar=skipped_similar,
                    accepted_payload=accepted_response_dict,
                )
                if expression_duplication_prompt is not None:
                    expression_duplication_prompt = "\n\n".join([expression_duplication_prompt, refill_feedback])
                else:
                    expression_duplication_prompt = refill_feedback
                logger.info(
                    "Requesting factor refill for filtered batch: "
                    f"{len(accepted_response_dict)}/{self.target_factor_count} accepted "
                    f"requesting {requested_factor_count} new "
                    f"(attempt {refill_attempts}/{max_refill_attempts})"
                )
                system_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(qa_prompt_dict["hypothesis2experiment"]["system_prompt"])
                    .render(
                        targets=self.targets,
                        scenario=trace.scen.background,
                        experiment_output_format=context["experiment_output_format"],
                        target_factor_count=requested_factor_count,
                    )
                )
                user_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(qa_prompt_dict["hypothesis2experiment"]["user_prompt"])
                    .render(
                        targets=self.targets,
                        target_hypothesis=context["target_hypothesis"],
                        hypothesis_and_feedback=context["hypothesis_and_feedback"],
                        function_lib_description=context["function_lib_description"],
                        target_list=context["target_list"],
                        RAG=context["RAG"],
                        target_factor_count=requested_factor_count,
                        expression_duplication=expression_duplication_prompt,
                    )
                )
                flag = False
                continue

            if len(accepted_response_dict) < self.target_factor_count:
                message = (
                    "Factor refill attempts exhausted without enough distinct usable factors: "
                    f"{len(accepted_response_dict)}/{self.target_factor_count}"
                )
                logger.warning(message)
                raise ValueError(message)

            filtered_response_dict = accepted_response_dict

            if filtered_response_dict:
                accepted_names = []
                accepted_exprs = []
                for factor_name, factor_data in filtered_response_dict.items():
                    if not isinstance(factor_data, dict):
                        continue
                    expr = str(factor_data.get("expression", "") or "").strip()
                    if not expr:
                        continue
                    accepted_names.append(factor_name)
                    accepted_exprs.append(expr)
                if accepted_names:
                    self.factor_regulator.add_factor(accepted_names, accepted_exprs)

            return self.convert_response(json.dumps(filtered_response_dict, ensure_ascii=False), trace)
    

    def convert_response(self, response: str, trace: Trace) -> FactorExperiment:
        response_dict = robust_json_parse(response)
        response_dict = _trim_factor_payload(response_dict, self.target_factor_count)
        tasks = []

        for factor_name in response_dict:
            factor_data = response_dict.get(factor_name, {})
            if not isinstance(factor_data, dict):
                continue
            description = factor_data.get("description", "")
            formulation = factor_data.get("formulation", "")
            expression = factor_data.get("expression", "")
            variables = factor_data.get("variables", {})
            tasks.append(
                FactorTask(
                    factor_name=factor_name,
                    factor_description=description,
                    factor_formulation=formulation,
                    factor_expression=expression,
                    variables=variables,
                )
            )
            
        exp = FactorMiningExperiment(tasks)
        exp.based_experiments = [FactorMiningExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]

        unique_tasks = []

        for task in tasks:
            duplicate = False
            for based_exp in exp.based_experiments:
                for sub_task in based_exp.sub_tasks:
                    if task.factor_name == sub_task.factor_name:
                        duplicate = True
                        break
                if duplicate:
                    break
            if not duplicate:
                unique_tasks.append(task)

        exp.sub_tasks = unique_tasks
        exp.sub_workspace_list = [None] * len(unique_tasks)
        exp.tasks = unique_tasks
        return exp



class BacktestHypothesis2FactorExpression(FactorHypothesis2Experiment):
    def __init__(self, factor_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor_path = factor_path
        
    def convert_response(self, *args, **kwargs) -> FactorExperiment:
        return super().convert_response(*args, **kwargs)
        
    def prepare_context(self, *args, **kwargs) -> Tuple[dict | bool]:
        return super().prepare_context(*args, **kwargs)
        
    def convert(self, hypothesis: Hypothesis, trace: Trace) -> FactorExperiment:
        if os.path.exists(self.factor_path):
            tasks = []
            factor_df = pd.read_csv(self.factor_path, usecols=["factor_name", "factor_expression"], index_col=None)
            for index, row in factor_df.iterrows():
                tasks.append(
                    FactorTask(
                        factor_name=row["factor_name"],
                        factor_description="",
                        factor_formulation="",
                        factor_expression=row["factor_expression"],
                        variables="",
                    )
                )
            
            exp = FactorMiningExperiment(tasks)
            exp.based_experiments = [FactorMiningExperiment(sub_tasks=[])] + [t[1] for t in trace.hist if t[2]]

            unique_tasks = []

            for task in tasks:
                duplicate = False
                for based_exp in exp.based_experiments:
                    for sub_task in based_exp.sub_tasks:
                        if task.factor_name == sub_task.factor_name:
                            duplicate = True
                            break
                    if duplicate:
                        break
                if not duplicate:
                    unique_tasks.append(task)

            exp.tasks = unique_tasks
            return exp
            
        else:
            raise ValueError(f"File {self.factor_csv_path} does not exist. ")


FactorHypothesisGen = FactorMiningHypothesisGen
FactorHypothesis2Experiment = FactorMiningHypothesis2Experiment
