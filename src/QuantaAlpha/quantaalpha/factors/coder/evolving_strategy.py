from __future__ import annotations

import copy
import json
from pathlib import Path
import re
from jinja2 import Environment, StrictUndefined

from quantaalpha.coder.costeer.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from quantaalpha.coder.costeer.knowledge_management import (
    CoSTEERQueriedKnowledge,
    CoSTEERQueriedKnowledgeV2,
)
from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
from quantaalpha.factors.coder.factor import FactorFBWorkspace, FactorTask
from quantaalpha.factors.coder.minute_op_registry import get_minute_op_registry, get_minute_op_spec
from quantaalpha.factors.coder.minute_spec_compiler import compile_expression_to_minute_spec
from quantaalpha.factors.coder.factor_ast import (
    BinaryOpNode,
    ConditionalNode,
    FunctionNode,
    NumberNode,
    UnaryOpNode,
    VarNode,
    parse_expression as parse_ast_expression,
)
from quantaalpha.core.template import CodeTemplate
from quantaalpha.llm.config import LLM_SETTINGS
from quantaalpha.llm.client import APIBackend
from quantaalpha.core.utils import multiprocessing_wrapper
from quantaalpha.factors.alignment.preflight_guard import validate_expression_against_registry
from quantaalpha.factors.combined_domain_contract import is_joint_pv_minutes_run
from quantaalpha.factors import data_domains as factor_data_domains
from quantaalpha.factors.coder.prompt_utils import DomainPromptProxy, build_runtime_field_constraints
from quantaalpha.core.conf import RD_AGENT_SETTINGS
from quantaalpha.log import logger

code_template = CodeTemplate(template_path=Path(__file__).parent / "template.jinjia2")
minute_code_template = CodeTemplate(template_path=Path(__file__).parent / "minute_template.jinjia2")
joint_minute_code_template = CodeTemplate(template_path=Path(__file__).parent / "joint_minute_template.jinjia2")
implement_prompts = DomainPromptProxy(base_file="prompts.yaml", prompt_dir=Path(__file__).parent)


_ALIGNED_ARITHMETIC_FUNCTIONS = {
    "+": "ADD",
    "-": "SUBTRACT",
    "*": "MULTIPLY",
    "/": "DIVIDE",
}
_ALIGNED_COMPARISON_FUNCTIONS = {
    ">": "GT",
    "<": "LT",
    ">=": "GE",
    "<=": "LE",
    "==": "EQ",
    "!=": "NE",
}
_ALIGNED_LOGICAL_FUNCTIONS = {
    "&": "AND",
    "&&": "AND",
    "|": "OR",
    "||": "OR",
}


def _format_expression_number(value: float) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return repr(numeric)


def _render_daily_aligned_expression_node(node) -> str:
    if isinstance(node, VarNode):
        return node.name
    if isinstance(node, NumberNode):
        return _format_expression_number(node.value)
    if isinstance(node, FunctionNode):
        return f"{node.name}({', '.join(_render_daily_aligned_expression_node(arg) for arg in node.args)})"
    if isinstance(node, UnaryOpNode):
        return f"({node.op}{_render_daily_aligned_expression_node(node.operand)})"
    if isinstance(node, BinaryOpNode):
        left = _render_daily_aligned_expression_node(node.left)
        right = _render_daily_aligned_expression_node(node.right)
        func_name = (
            _ALIGNED_ARITHMETIC_FUNCTIONS.get(node.op)
            or _ALIGNED_COMPARISON_FUNCTIONS.get(node.op)
            or _ALIGNED_LOGICAL_FUNCTIONS.get(node.op)
        )
        if func_name is not None:
            return f"{func_name}({left}, {right})"
        return f"({left} {node.op} {right})"
    if isinstance(node, ConditionalNode):
        condition = _render_daily_aligned_expression_node(node.condition)
        true_expr = _render_daily_aligned_expression_node(node.true_expr)
        false_expr = _render_daily_aligned_expression_node(node.false_expr)
        return f"WHERE({condition}, {true_expr}, {false_expr})"
    return str(node)


def _rewrite_daily_infix_expression(expression: str) -> str:
    return _render_daily_aligned_expression_node(parse_ast_expression(expression))


def _render_daily_expression_module(expression: str, factor_name: str, data_needed: list[str], domain_type: str) -> str:
    return code_template.render(
        expression=_rewrite_daily_infix_expression(expression),
        factor_name=factor_name,
        data_needed=data_needed,
        domain_type=domain_type,
    )


def _render_daily_expression_failure_module(target_task: FactorTask, failure_reason: str | None) -> str:
    message = str(failure_reason or "Daily expression render failed.").strip()
    domains = factor_data_domains.resolve_factor_domains()
    module = code_template.render(
        expression="raise_expression_render_failure()",
        factor_name=target_task.factor_name,
        data_needed=[],
        domain_type=_daily_template_domain_type(domains),
    )
    return (
        f"{module}\n\n"
        f"DAILY_RENDER_FAILURE_MESSAGE = {_python_literal(message)}\n\n"
        "def raise_expression_render_failure():\n"
        "    raise RuntimeError(DAILY_RENDER_FAILURE_MESSAGE)\n"
    )


def _format_expression_registry_error(check: dict) -> str:
    parts = []
    parse_error = check.get("parse_error")
    if parse_error:
        parts.append(f"parse_error={parse_error}")
    unsupported_operators = check.get("unsupported_operators") or []
    if unsupported_operators:
        parts.append(f"unsupported operators: {', '.join(str(item) for item in unsupported_operators)}")
    unsupported_fields = check.get("unsupported_fields") or []
    if unsupported_fields:
        parts.append(f"unsupported fields: {', '.join(str(item) for item in unsupported_fields)}")
        if any(str(item) == "None" for item in unsupported_fields):
            parts.append("remove optional keyword arguments or None literals from the expression")
    semantic_errors = check.get("semantic_errors") or []
    if semantic_errors:
        parts.append(f"semantic errors: {'; '.join(str(item) for item in semantic_errors)}")
    return "; ".join(parts) or "expression failed registry validation"


def _infer_expression_data_needed(expression: str) -> list[str]:
    check = validate_expression_against_registry(expression)
    if not check["ok"]:
        raise ValueError(f"invalid expression: {_format_expression_registry_error(check)}")
    return sorted(check["used_fields"])


_ALLOWLIST_FEEDBACK_MARKERS = (
    "allowlist",
    "not in the current runtime allowlist",
    "not in the allowed operators list",
    "unsupported operator",
    "unsupported operators",
    "registry constraint",
    "registry constraints",
    "violating registry constraints",
)


def _mentions_allowlist_violation(text: str | None) -> bool:
    if not text:
        return False
    normalized = " ".join(str(text).lower().split())
    return any(marker in normalized for marker in _ALLOWLIST_FEEDBACK_MARKERS)


def _sanitize_feedback_text_for_prompt(feedback) -> str:
    text = "" if feedback is None else str(feedback)
    if not text:
        return text
    kept_lines = [line for line in text.splitlines() if not _mentions_allowlist_violation(line)]
    sanitized = "\n".join(line for line in kept_lines if line.strip()).strip()
    if sanitized:
        return sanitized
    if _mentions_allowlist_violation(text):
        return (
            "Ignore earlier allowlist or registry complaints from historical runs. "
            "Operator and field legality is checked deterministically for the current runtime."
        )
    return text


def _format_implementation_and_feedback_for_prompt(knowledge) -> str:
    return (
        "------------------implementation code:------------------\n"
        f"{knowledge.implementation.code}\n"
        "------------------implementation feedback:------------------\n"
        f"{_sanitize_feedback_text_for_prompt(getattr(knowledge, 'feedback', None))}\n"
    )


def _clone_knowledge_with_sanitized_feedback(knowledge):
    if knowledge is None:
        return None
    cloned = copy.copy(knowledge)
    cloned.feedback = _sanitize_feedback_text_for_prompt(getattr(knowledge, "feedback", None))
    return cloned


def _build_deterministic_minute_code(target_task: FactorTask) -> str | None:
    expression = str(getattr(target_task, "factor_expression", "") or "").strip()
    minute_fields = {
        "opens",
        "highs",
        "lows",
        "closes",
        "volumes",
        "amounts",
        "ratios",
        "vwaps",
        "turnovers",
        "returns",
        "lreturns",
    }
    daily_pv_fields = {
        "hfq_opens",
        "hfq_closes",
        "hfq_highs",
        "hfq_lows",
        "vwaps",
        "turnovers",
    }
    match = re.fullmatch(
        r"safe_div\(\s*ts_return\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        expression,
    )
    if match:
        minute_field, lag_text, daily_field = match.groups()
        if minute_field in minute_fields and daily_field in daily_pv_fields:
            lag = int(lag_text)
            return f'''import h5py
import numpy as np
import pandas as pd
from vendors.quant_lib.minute_tools import MinuteFactorEngine, load_single_minute
from quant_union.common.quantEnum import DomainType, CategoryType

TYPE = "regular"
META = {{
    "factor_name": "{target_task.factor_name}",
    "author": "quantaalpha",
    "level": "minutes",
    "domain": DomainType.pv,
    "tag": "",
    "category": "unknown",
}}
SETTING = {{
    "universe": "standards",
    "data_needed": ["{daily_field}"],
    "pasteurization": False,
    "decay": 0,
    "neutralize": None,
}}

def prepare_minute_datas():
    mfe = MinuteFactorEngine()
    with h5py.File(mfe.h5_path, "r") as handle:
        minutes = pd.Index(handle["axis/minutes"][:].astype(str))
        if len(minutes) <= {lag}:
            raise ValueError("minute axis is shorter than required lag {lag}")
        current_minute = str(minutes[-1])
        lagged_minute = str(minutes[-1 - {lag}])
        current_value = load_single_minute(handle, "{minute_field}", current_minute)
        lagged_value = load_single_minute(handle, "{minute_field}", lagged_minute).replace(0.0, np.nan)
    minute_return = current_value.divide(lagged_value).subtract(1.0)
    return {{
        "{minute_field}_return_{lag}": minute_return,
    }}

def calc_factor(data_ctx, minute_ctx):
    daily_anchor = data_ctx["{daily_field}"].replace(0.0, np.nan)
    return minute_ctx["{minute_field}_return_{lag}"].divide(daily_anchor)
'''

    match = re.fullmatch(
        r"safe_div\(\s*(?:(mean|sum|ts_mean))\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        expression,
    )
    if match:
        agg_name, minute_field, window_text, daily_field = match.groups()
        if minute_field in minute_fields and daily_field in daily_pv_fields:
            window = int(window_text)
            lookback = window - 1
            groupby_op = "mean" if agg_name in {"mean", "ts_mean"} else "sum"
            return f'''import h5py
import numpy as np
import pandas as pd
from vendors.quant_lib.minute_tools import MinuteFactorEngine, load_single_minute
from quant_union.common.quantEnum import DomainType, CategoryType

TYPE = "regular"
META = {{
    "factor_name": "{target_task.factor_name}",
    "author": "quantaalpha",
    "level": "minutes",
    "domain": DomainType.pv,
    "tag": "",
    "category": "unknown",
}}
SETTING = {{
    "universe": "standards",
    "data_needed": ["{daily_field}"],
    "pasteurization": False,
    "decay": 0,
    "neutralize": None,
}}

def prepare_minute_datas():
    mfe = MinuteFactorEngine()
    with h5py.File(mfe.h5_path, "r") as handle:
        minutes = pd.Index(handle["axis/minutes"][:].astype(str))
        if len(minutes) < {window}:
            raise ValueError("minute axis is shorter than required window {window}")
        end_idx = len(minutes) - 1
        start_idx = end_idx - {lookback}
        agg_frames = [
            load_single_minute(handle, "{minute_field}", str(minute_text))
            for minute_text in minutes[start_idx : end_idx + 1]
        ]
    stacked = pd.concat(agg_frames, keys=range(len(agg_frames)), names=["offset"])
    aggregated = stacked.groupby(level=1).{groupby_op}()
    return {{
        "{minute_field}_{agg_name}_{window}": aggregated,
    }}

def calc_factor(data_ctx, minute_ctx):
    daily_anchor = data_ctx["{daily_field}"].replace(0.0, np.nan)
    return minute_ctx["{minute_field}_{agg_name}_{window}"].divide(daily_anchor)
'''

    match = re.fullmatch(r"ts_return\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)", expression)
    if match:
        minute_field, lag_text = match.groups()
        if minute_field not in minute_fields:
            return None

        lag = int(lag_text)
        return f'''import h5py
import numpy as np
import pandas as pd
from vendors.quant_lib.minute_tools import MinuteFactorEngine, load_single_minute
from quant_union.common.quantEnum import DomainType, CategoryType

TYPE = "regular"
META = {{
    "factor_name": "{target_task.factor_name}",
    "author": "quantaalpha",
    "level": "minutes",
    "domain": DomainType.pv,
    "tag": "",
    "category": "unknown",
}}
SETTING = {{
    "universe": "standards",
    "data_needed": [],
    "pasteurization": False,
    "decay": 0,
    "neutralize": None,
}}

def prepare_minute_datas():
    mfe = MinuteFactorEngine()
    with h5py.File(mfe.h5_path, "r") as handle:
        minutes = pd.Index(handle["axis/minutes"][:].astype(str))
        if len(minutes) <= {lag}:
            raise ValueError("minute axis is shorter than required lag {lag}")
        current_minute = str(minutes[-1])
        lagged_minute = str(minutes[-1 - {lag}])
        current_value = load_single_minute(handle, "{minute_field}", current_minute)
        lagged_value = load_single_minute(handle, "{minute_field}", lagged_minute)
    return {{
        "{minute_field}_current": current_value,
        "{minute_field}_lag_{lag}": lagged_value,
    }}

def calc_factor(data_ctx, minute_ctx):
    current_value = minute_ctx["{minute_field}_current"]
    lagged_value = minute_ctx["{minute_field}_lag_{lag}"]
    return current_value.divide(lagged_value).subtract(1.0)
'''

    match = re.fullmatch(r"(?:(mean|sum|ts_mean))\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)", expression)
    if match:
        agg_name, minute_field, window_text = match.groups()
        if minute_field not in minute_fields:
            return None

        window = int(window_text)
        lookback = window - 1
        groupby_op = "mean" if agg_name in {"mean", "ts_mean"} else "sum"
        return f'''import h5py
import numpy as np
import pandas as pd
from vendors.quant_lib.minute_tools import MinuteFactorEngine, load_single_minute
from quant_union.common.quantEnum import DomainType, CategoryType

TYPE = "regular"
META = {{
    "factor_name": "{target_task.factor_name}",
    "author": "quantaalpha",
    "level": "minutes",
    "domain": DomainType.pv,
    "tag": "",
    "category": "unknown",
}}
SETTING = {{
    "universe": "standards",
    "data_needed": [],
    "pasteurization": False,
    "decay": 0,
    "neutralize": None,
}}

def prepare_minute_datas():
    mfe = MinuteFactorEngine()
    with h5py.File(mfe.h5_path, "r") as handle:
        minutes = pd.Index(handle["axis/minutes"][:].astype(str))
        if len(minutes) < {window}:
            raise ValueError("minute axis is shorter than required window {window}")
        end_idx = len(minutes) - 1
        start_idx = end_idx - {lookback}
        agg_frames = [
            load_single_minute(handle, "{minute_field}", str(minute_text))
            for minute_text in minutes[start_idx : end_idx + 1]
        ]
    aggregated = pd.concat(agg_frames).groupby(level=0).{groupby_op}()
    return {{
        "{minute_field}_{groupby_op}_{window}": aggregated,
    }}

def calc_factor(data_ctx, minute_ctx):
    return minute_ctx["{minute_field}_{groupby_op}_{window}"]
'''

    match = re.fullmatch(r"(?:(mean|sum|ts_mean))\(\s*pct\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)\s*\)", expression)
    if not match:
        return None

    agg_name, minute_field, lag_text = match.groups()
    if minute_field not in minute_fields:
        return None

    lag = int(lag_text)
    groupby_op = "mean" if agg_name in {"mean", "ts_mean"} else "sum"
    return f'''import h5py
import numpy as np
import pandas as pd
from vendors.quant_lib.minute_tools import MinuteFactorEngine, load_single_minute
from quant_union.common.quantEnum import DomainType, CategoryType

TYPE = "regular"
META = {{
    "factor_name": "{target_task.factor_name}",
    "author": "quantaalpha",
    "level": "minutes",
    "domain": DomainType.pv,
    "tag": "",
    "category": "unknown",
}}
SETTING = {{
    "universe": "standards",
    "data_needed": [],
    "pasteurization": False,
    "decay": 0,
    "neutralize": None,
}}

def prepare_minute_datas():
    mfe = MinuteFactorEngine()
    with h5py.File(mfe.h5_path, "r") as handle:
        minutes = pd.Index(handle["axis/minutes"][:].astype(str))
        if len(minutes) <= {lag}:
            raise ValueError("minute axis is shorter than required lag {lag}")
        pct_frames = []
        for current_idx in range({lag}, len(minutes)):
            current_value = load_single_minute(handle, "{minute_field}", str(minutes[current_idx]))
            lagged_value = load_single_minute(handle, "{minute_field}", str(minutes[current_idx - {lag}])).replace(0.0, np.nan)
            pct_frames.append(current_value.divide(lagged_value).subtract(1.0))
    aggregated = pd.concat(pct_frames).groupby(level=0).{groupby_op}()
    return {{
        "{minute_field}_pct_{groupby_op}_{lag}": aggregated,
    }}

def calc_factor(data_ctx, minute_ctx):
    return minute_ctx["{minute_field}_pct_{groupby_op}_{lag}"]
'''


def _sanitize_similar_error_knowledge_pairs(knowledge_pairs):
    sanitized_pairs = []
    for error_content, pair in knowledge_pairs or []:
        if _mentions_allowlist_violation(error_content):
            continue
        sanitized_pairs.append((error_content, pair))
    return sanitized_pairs


def _flatten_similar_error_knowledge(knowledge_pairs) -> list[tuple]:
    if not knowledge_pairs:
        return []
    if isinstance(knowledge_pairs, dict):
        flattened = []
        for error_items in knowledge_pairs.values():
            flattened.extend(error_items or [])
        return flattened
    flattened = []
    for item in knowledge_pairs:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        _, pair = item
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            flattened.append((pair[0], pair[1]))
    return flattened


def _extract_expr_from_code(code_str: str) -> str:
    """Extract expr from generated factor code."""
    pattern = r'expr\s*=\s*["\']([^"\']*)["\']'
    match = re.search(pattern, code_str)
    if match:
        return match.group(1)
    return ""


def _minute_domain_active() -> bool:
    return "minutes" in factor_data_domains.resolve_factor_domains()


def _joint_pv_minutes_domain_active() -> bool:
    return is_joint_pv_minutes_run(factor_data_domains.resolve_factor_domains())


def _joint_pv_exclusive_fields() -> tuple[str, ...]:
    fields = getattr(factor_data_domains, "JOINT_PV_EXCLUSIVE_FIELDS", None)
    if fields is not None:
        return tuple(fields)
    pv_fields = tuple(getattr(factor_data_domains, "PV_FIELDS", ()))
    minute_fields = set(getattr(factor_data_domains, "MINUTE_FIELDS", ()))
    return tuple(field for field in pv_fields if field not in minute_fields)


def _joint_minute_exclusive_fields() -> tuple[str, ...]:
    fields = getattr(factor_data_domains, "JOINT_MINUTE_EXCLUSIVE_FIELDS", None)
    if fields is not None:
        return tuple(fields)
    minute_fields = tuple(getattr(factor_data_domains, "MINUTE_FIELDS", ()))
    pv_fields = set(getattr(factor_data_domains, "PV_FIELDS", ()))
    return tuple(field for field in minute_fields if field not in pv_fields)


def _joint_shared_fields() -> tuple[str, ...]:
    fields = getattr(factor_data_domains, "JOINT_PV_MINUTES_SHARED_FIELDS", None)
    if fields is not None:
        return tuple(fields)
    pv_fields = set(getattr(factor_data_domains, "PV_FIELDS", ()))
    minute_fields = set(getattr(factor_data_domains, "MINUTE_FIELDS", ()))
    return tuple(sorted(pv_fields & minute_fields))


def _build_factor_prompt_information(target_task: FactorTask) -> str:
    if not _minute_domain_active():
        return target_task.get_task_description()

    parts = [
        f"factor_name: {target_task.factor_name}",
        f"factor_description: {target_task.factor_description}",
    ]
    formulation = str(getattr(target_task, "factor_formulation", "") or "").strip()
    if formulation:
        parts.append(f"factor_formulation: {formulation}")
    expression = str(getattr(target_task, "factor_expression", "") or "").strip()
    if expression:
        parts.append(f"factor_expression: {expression}")
    variables = getattr(target_task, "variables", None)
    if variables:
        parts.append(f"variables: {variables}")
    if _joint_pv_minutes_domain_active():
        routing_note = _build_joint_pv_minutes_routing_note(target_task)
        if routing_note:
            parts.append(routing_note)
    return "\n".join(parts)


def _build_joint_pv_minutes_routing_note(target_task: FactorTask) -> str:
    expression = str(getattr(target_task, "factor_expression", "") or "").strip()
    if not expression:
        return ""
    check = validate_expression_against_registry(expression)
    used_fields = set(check.get("used_fields") or [])
    minute_explicit = sorted(used_fields & set(_joint_minute_exclusive_fields()))
    pv_explicit = sorted(used_fields & set(_joint_pv_exclusive_fields()))
    shared_fields = sorted(used_fields & set(_joint_shared_fields()))
    lines = [
        "joint_field_routing:",
        f"- explicit_minute_fields_in_expression: {minute_explicit or 'none'}",
        f"- explicit_daily_pv_fields_in_expression: {pv_explicit or 'none'}",
        f"- shared_fields_in_expression: {shared_fields or 'none'}",
        "- route minute fields only through prepare_minute_datas() via MinuteFactorEngine.run(...) or load_single_minute(...).",
        "- route daily pv fields only through data_ctx in calc_factor(...), and declare them in SETTING['data_needed'].",
        "- raw `opens/highs/lows/closes` belong only to the minute side in joint tasks; the daily pv side must use `hfq_opens/hfq_highs/hfq_lows/hfq_closes`.",
        "- never emit doubly adjusted names such as `hfq_hfq_closes`; use a single `hfq_*` prefix at most once.",
        "- use singular `volume`, `vwap`, and `turnover` for daily pv shared-field anchors in joint tasks; they route through data_ctx and are mapped back to runtime `volumes`, `vwaps`, and `turnovers` during backtest.",
        "- use plural `volumes`, `vwaps`, and `turnovers` only for raw minute data inside minute subtrees.",
        "- use `ts_return(field, lag)` only for lagged same-field returns; use `pct(a, b)` only for same-time tensor ratio `a / b - 1`.",
        "- after minute features are reduced to daily 2D frames, it is valid to combine them with daily pv operators in calc_factor(...).",
        "- once a minute reduce operator has already produced a daily 2D frame, any further `ts_rank/ts_zscore/ts_corr/...` in calc_factor(...) uses regular daily semantics, not intraday minute semantics.",
    ]
    return "\n".join(lines)


_STRUCTURED_MINUTE_FEATURE_KINDS = (
    "snapshot",
    "snapshot_return",
    "window_aggregate",
    "window_engine",
    "intraday_pct_aggregate",
    "engine_run",
    "vector_expr",
)
_STRUCTURED_MINUTE_AGGREGATIONS = {"mean", "sum", "min", "max", "std", "kurt"}
_STRUCTURED_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DAILY_FINAL_EXPRESSION_ALIASES = {
    "neutralize": "cs_neutralize",
    "scale": "cs_scale",
    "zscore": "cs_zscore",
}


def _python_literal(value) -> str:
    return repr(value)


def _normalize_feature_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not _STRUCTURED_IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"invalid minute feature name: {name!r}")
    return normalized


def _canonicalize_published_daily_aliases(text: str) -> str:
    return factor_data_domains.canonicalize_joint_pv_daily_aliases(text)


def _canonicalize_published_data_needed(fields: list[str]) -> list[str]:
    canonicalized: list[str] = []
    for field_name in fields:
        published = _canonicalize_published_daily_aliases(field_name)
        if published not in canonicalized:
            canonicalized.append(published)
    return canonicalized


def _require_int(value, field_name: str, *, min_value: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    number = int(value)
    if number < min_value:
        raise ValueError(f"{field_name} must be >= {min_value}")
    return number


def _require_string(value, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _minute_field_set() -> set[str]:
    return set(getattr(factor_data_domains, "MINUTE_FIELDS", ()))


def _daily_pv_field_set() -> set[str]:
    fields = set(getattr(factor_data_domains, "PV_FIELDS", ()))
    if _joint_pv_minutes_domain_active():
        fields.update(getattr(factor_data_domains, "JOINT_PV_DAILY_SHARED_FIELD_ALIASES", {}).keys())
    return fields


def _extract_embedded_data_ctx_fields(text: str) -> set[str]:
    if not text:
        return set()
    return {
        match.group(1).strip()
        for match in re.finditer(r'data_ctx\[\s*["\']([^"\']+)["\']\s*\]', str(text))
        if match.group(1).strip()
    }


def _extract_identifier_tokens(text: str) -> set[str]:
    if not text:
        return set()
    return {match.group(0) for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", str(text))}


def _registered_minute_operator_names() -> set[str]:
    return set(get_minute_op_registry().keys())


def _minute_op_arity(name: str) -> int | None:
    spec = get_minute_op_spec(name)
    return None if spec is None else spec.arity


def _normalize_engine_spec(name: str, feature: dict) -> dict:
    inputs = feature.get("inputs")
    if isinstance(inputs, str):
        inputs = [inputs]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError(f"engine_run feature `{name}` must provide non-empty inputs")
    normalized_inputs = [_require_string(item, f"minute_features[{name}].inputs") for item in inputs]
    invalid_inputs = sorted(set(normalized_inputs) - _minute_field_set())
    if invalid_inputs:
        raise ValueError(f"engine_run feature `{name}` has invalid minute inputs: {invalid_inputs}")

    preprocess = feature.get("preprocess")
    normalized_preprocess = None
    if preprocess is not None:
        if isinstance(preprocess, str):
            preprocess = [preprocess]
        if not isinstance(preprocess, list):
            raise ValueError(f"engine_run feature `{name}` preprocess must be a list")
        if len(preprocess) != len(normalized_inputs):
            raise ValueError(f"engine_run feature `{name}` preprocess length must match inputs length")
        registered_ops = _registered_minute_operator_names()
        normalized_preprocess = []
        for index, item in enumerate(preprocess):
            if item is None:
                normalized_preprocess.append(None)
                continue
            preprocess_name = _require_string(item, f"minute_features[{name}].preprocess[{index}]")
            if preprocess_name not in registered_ops:
                raise ValueError(
                    f"engine_run feature `{name}` preprocess uses unsupported operator `{preprocess_name}`"
                )
            if _minute_op_arity(preprocess_name) != 1:
                raise ValueError(
                    f"engine_run feature `{name}` preprocess operator `{preprocess_name}` must be unary"
                )
            normalized_preprocess.append(preprocess_name)

    operator = _require_string(feature.get("operator"), f"minute_features[{name}].operator")
    registered_ops = _registered_minute_operator_names()
    if operator not in registered_ops:
        raise ValueError(f"engine_run feature `{name}` uses unsupported operator `{operator}`")
    operator_params = feature.get("operator_params") or {}
    if operator_params and not isinstance(operator_params, dict):
        raise ValueError(f"engine_run feature `{name}` operator_params must be a dict")

    aggregator = feature.get("aggregator")
    if aggregator is not None:
        aggregator = _require_string(aggregator, f"minute_features[{name}].aggregator")
        if aggregator not in registered_ops:
            raise ValueError(f"engine_run feature `{name}` uses unsupported aggregator `{aggregator}`")
    aggregator_params = feature.get("aggregator_params") or {}
    if aggregator_params and not isinstance(aggregator_params, dict):
        raise ValueError(f"engine_run feature `{name}` aggregator_params must be a dict")

    kwargs = feature.get("kwargs") or {}
    if kwargs and not isinstance(kwargs, dict):
        raise ValueError(f"engine_run feature `{name}` kwargs must be a dict")

    return {
        "name": name,
        "kind": "engine_run",
        "inputs": normalized_inputs,
        "preprocess": normalized_preprocess,
        "operator": operator,
        "operator_params": operator_params,
        "aggregator": aggregator,
        "aggregator_params": aggregator_params,
        "kwargs": kwargs,
        "replace_zero_with_nan": bool(feature.get("replace_zero_with_nan", False)),
    }


def _normalize_minute_feature_spec(feature: dict) -> dict:
    if not isinstance(feature, dict):
        raise ValueError("minute feature spec must be an object")
    name = _normalize_feature_name(feature.get("name"))
    kind = _require_string(feature.get("kind"), f"minute_features[{name}].kind")
    if kind not in _STRUCTURED_MINUTE_FEATURE_KINDS:
        raise ValueError(f"unsupported minute feature kind `{kind}` for `{name}`")
    if kind == "engine_run":
        return _normalize_engine_spec(name, feature)
    if kind == "vector_expr":
        expression_code = _require_string(feature.get("expression_code"), f"minute_features[{name}].expression_code")
        inputs = feature.get("inputs")
        if isinstance(inputs, str):
            inputs = [inputs]
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(f"vector_expr feature `{name}` must provide non-empty inputs")
        normalized_inputs = [_require_string(item, f"minute_features[{name}].inputs") for item in inputs]
        invalid_inputs = sorted(set(normalized_inputs) - _minute_field_set())
        if invalid_inputs:
            raise ValueError(f"vector_expr feature `{name}` has invalid minute inputs: {invalid_inputs}")
        return {
            "name": name,
            "kind": "vector_expr",
            "inputs": normalized_inputs,
            "expression_code": expression_code,
            "replace_zero_with_nan": bool(feature.get("replace_zero_with_nan", False)),
        }
    if kind == "window_engine":
        normalized_engine = _normalize_engine_spec(name, feature)
        aggregation = _require_string(feature.get("aggregation"), f"minute_features[{name}].aggregation").lower()
        if aggregation not in _STRUCTURED_MINUTE_AGGREGATIONS:
            raise ValueError(f"window_engine feature `{name}` has unsupported aggregation `{aggregation}`")
        normalized_engine["kind"] = "window_engine"
        normalized_engine["window"] = _require_int(feature.get("window"), f"minute_features[{name}].window", min_value=1)
        normalized_engine["aggregation"] = aggregation
        normalized_engine["end_offset"] = _require_int(
            feature.get("end_offset", 0), f"minute_features[{name}].end_offset", min_value=0
        )
        return normalized_engine

    field_name = _require_string(feature.get("field"), f"minute_features[{name}].field")
    if field_name not in _minute_field_set():
        raise ValueError(f"minute feature `{name}` uses unsupported minute field `{field_name}`")
    normalized = {
        "name": name,
        "kind": kind,
        "field": field_name,
        "replace_zero_with_nan": bool(feature.get("replace_zero_with_nan", False)),
    }
    if kind == "snapshot":
        offset = feature.get("offset", 0)
        minute = feature.get("minute")
        if minute is not None and offset not in (None, 0):
            raise ValueError(f"snapshot feature `{name}` cannot set both minute and offset")
        normalized["offset"] = _require_int(offset or 0, f"minute_features[{name}].offset", min_value=0)
        if minute is not None:
            normalized["minute"] = _require_string(minute, f"minute_features[{name}].minute")
    elif kind == "snapshot_return":
        normalized["lag"] = _require_int(feature.get("lag"), f"minute_features[{name}].lag", min_value=1)
        normalized["end_offset"] = _require_int(
            feature.get("end_offset", 0), f"minute_features[{name}].end_offset", min_value=0
        )
    elif kind == "window_aggregate":
        aggregation = _require_string(feature.get("aggregation"), f"minute_features[{name}].aggregation").lower()
        if aggregation not in _STRUCTURED_MINUTE_AGGREGATIONS:
            raise ValueError(f"window_aggregate feature `{name}` has unsupported aggregation `{aggregation}`")
        normalized["window"] = _require_int(feature.get("window"), f"minute_features[{name}].window", min_value=1)
        normalized["aggregation"] = aggregation
        normalized["end_offset"] = _require_int(
            feature.get("end_offset", 0), f"minute_features[{name}].end_offset", min_value=0
        )
    elif kind == "intraday_pct_aggregate":
        aggregation = _require_string(feature.get("aggregation"), f"minute_features[{name}].aggregation").lower()
        if aggregation not in _STRUCTURED_MINUTE_AGGREGATIONS:
            raise ValueError(
                f"intraday_pct_aggregate feature `{name}` has unsupported aggregation `{aggregation}`"
            )
        normalized["lag"] = _require_int(feature.get("lag"), f"minute_features[{name}].lag", min_value=1)
        normalized["aggregation"] = aggregation
        normalized["end_offset"] = _require_int(
            feature.get("end_offset", 0), f"minute_features[{name}].end_offset", min_value=0
        )
    return normalized


def _normalize_structured_minute_spec(spec: dict, *, target_task: FactorTask) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("minute spec must be a JSON object")
    data_needed = spec.get("data_needed") or []
    if not isinstance(data_needed, list):
        raise ValueError("minute spec `data_needed` must be a list")
    normalized_data_needed = [_require_string(field, "data_needed item") for field in data_needed]
    duplicate_data_needed = [
        field_name for field_name in normalized_data_needed if normalized_data_needed.count(field_name) > 1
    ]
    if duplicate_data_needed:
        raise ValueError(f"duplicate daily fields in data_needed: {sorted(set(duplicate_data_needed))}")
    invalid_daily_fields = sorted(set(normalized_data_needed) - _daily_pv_field_set())
    if invalid_daily_fields:
        raise ValueError(f"data_needed contains unsupported daily pv fields: {invalid_daily_fields}")

    minute_features = spec.get("minute_features") or []
    if not isinstance(minute_features, list) or not minute_features:
        raise ValueError("minute spec must contain at least one minute feature")
    normalized_features = [_normalize_minute_feature_spec(feature) for feature in minute_features]
    feature_names = [feature["name"] for feature in normalized_features]
    duplicate_feature_names = [name for name in feature_names if feature_names.count(name) > 1]
    if duplicate_feature_names:
        raise ValueError(f"duplicate minute feature names: {sorted(set(duplicate_feature_names))}")

    final_expression = _canonicalize_final_expression(
        _require_string(spec.get("final_expression"), "final_expression"),
        normalized_data_needed,
    )
    final_expression_tokens = _extract_identifier_tokens(final_expression)
    referenced_daily_fields = sorted(_extract_embedded_data_ctx_fields(final_expression) & set(normalized_data_needed))
    referenced_feature_names = sorted(final_expression_tokens & set(feature_names))
    raw_minute_fields_in_expression = sorted(
        (final_expression_tokens & _minute_field_set()) - set(feature_names) - set(normalized_data_needed)
    )

    if raw_minute_fields_in_expression:
        raise ValueError(
            "final_expression must reference only rendered minute feature names, not raw minute fields directly: "
            f"{raw_minute_fields_in_expression}"
        )

    if _joint_pv_minutes_domain_active():
        if not normalized_data_needed:
            raise ValueError("joint pv/minutes spec must declare at least one daily pv field in data_needed")
        if not referenced_daily_fields:
            raise ValueError(
                "joint pv/minutes spec final_expression must reference at least one declared daily pv field via data_ctx[...]"
            )
        if not referenced_feature_names:
            raise ValueError(
                "joint pv/minutes spec final_expression must reference at least one minute feature output"
            )
    else:
        if normalized_data_needed:
            raise ValueError("pure minute spec must not declare daily pv fields in data_needed")
        if referenced_daily_fields or "data_ctx[" in final_expression:
            raise ValueError("pure minute spec final_expression must not access daily pv fields through data_ctx")
        if not referenced_feature_names:
            raise ValueError("pure minute spec final_expression must reference at least one minute feature output")

    published_data_needed = _canonicalize_published_data_needed(normalized_data_needed)
    published_final_expression = _canonicalize_published_daily_aliases(final_expression)
    published_original_expression = _canonicalize_published_daily_aliases(
        str(getattr(target_task, "factor_expression", "") or "").strip()
    )

    return {
        "factor_name": target_task.factor_name,
        "data_needed": published_data_needed,
        "minute_features": normalized_features,
        "final_expression": published_final_expression,
        "original_expression": published_original_expression,
    }


def _render_engine_operator_literal(name: str, params: dict) -> str:
    if params:
        return f"({_python_literal(name)}, {_python_literal(params)})"
    return _python_literal(name)


def _canonicalize_final_expression(final_expression: str, data_needed: list[str]) -> str:
    expression = str(final_expression or "").strip()
    if not expression:
        raise ValueError("final_expression must be non-empty")

    for alias_name, daily_name in _DAILY_FINAL_EXPRESSION_ALIASES.items():
        expression = re.sub(
            rf"\b{re.escape(alias_name)}\s*\(",
            f"{daily_name}(",
            expression,
        )

    declared_daily_fields = [
        field_name
        for field_name in data_needed
        if field_name in _daily_pv_field_set()
    ]
    for field_name in sorted(set(declared_daily_fields), key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(field_name)}\b")

        def replace_daily_field(match: re.Match[str]) -> str:
            prefix = expression[max(0, match.start() - 16) : match.start()]
            if prefix.endswith('data_ctx["') or prefix.endswith("data_ctx['"):
                return match.group(0)
            return f'data_ctx["{field_name}"]'

        expression = pattern.sub(replace_daily_field, expression)
    return expression


def _render_minute_feature_prepare_block(feature: dict) -> str:
    name = feature["name"]
    temp_name = f"__{name}"
    kind = feature["kind"]
    if kind == "snapshot":
        lines = []
        minute_value = feature.get("minute")
        if minute_value:
            lines.append(f'{temp_name}_minute = {_python_literal(minute_value)}')
        else:
            offset = feature.get("offset", 0)
            lines.extend(
                [
                    f"if len(minutes) <= {offset}:",
                    f'    raise ValueError("minute axis is shorter than required offset {offset} for {name}")',
                    f"{temp_name}_minute = str(minutes[-1 - {offset}])",
                ]
            )
        lines.append(
            f'{temp_name}_value = load_single_minute(handle, {_python_literal(feature["field"])}, {temp_name}_minute)'
        )
        if feature.get("replace_zero_with_nan"):
            lines.append(f"{temp_name}_value = {temp_name}_value.replace(0.0, np.nan)")
        lines.append(f'minute_ctx[{_python_literal(name)}] = {temp_name}_value')
        return "\n".join(lines)

    if kind == "snapshot_return":
        lag = feature["lag"]
        end_offset = feature.get("end_offset", 0)
        lines = [
            f"if len(minutes) <= {lag + end_offset}:",
            f'    raise ValueError("minute axis is shorter than required lag {lag} for {name}")',
            f"{temp_name}_current_idx = len(minutes) - 1 - {end_offset}",
            f"{temp_name}_current_minute = str(minutes[{temp_name}_current_idx])",
            f"{temp_name}_lagged_minute = str(minutes[{temp_name}_current_idx - {lag}])",
            f'{temp_name}_current = load_single_minute(handle, {_python_literal(feature["field"])}, {temp_name}_current_minute)',
            f'{temp_name}_lagged = load_single_minute(handle, {_python_literal(feature["field"])}, {temp_name}_lagged_minute)',
        ]
        if feature.get("replace_zero_with_nan", True):
            lines.append(f"{temp_name}_lagged = {temp_name}_lagged.replace(0.0, np.nan)")
        lines.append(
            f'minute_ctx[{_python_literal(name)}] = {temp_name}_current.divide({temp_name}_lagged).subtract(1.0)'
        )
        return "\n".join(lines)

    if kind == "window_aggregate":
        window = feature["window"]
        end_offset = feature.get("end_offset", 0)
        aggregation = feature["aggregation"]
        lines = [
            f"{temp_name}_available_window = len(minutes) - {end_offset}",
            f"if {temp_name}_available_window <= 0:",
            f'    raise ValueError("minute axis is shorter than required offset {end_offset} for {name}")',
            f"{temp_name}_window = min({window}, {temp_name}_available_window)",
            f"{temp_name}_end_idx = len(minutes) - 1 - {end_offset}",
            f"{temp_name}_start_idx = {temp_name}_end_idx - {temp_name}_window + 1",
            f"{temp_name}_startminute = str(minutes[{temp_name}_start_idx])",
            f"{temp_name}_endminute = str(minutes[{temp_name}_end_idx])",
            f"{temp_name}_value = mfe.run(",
            f"    inputs={_python_literal(feature['field'])},",
            '    operator="identity",',
            f"    aggregator={_python_literal(aggregation)},",
            f"    startminute={temp_name}_startminute,",
            f"    endminute={temp_name}_endminute,",
            ")",
        ]
        if feature.get("replace_zero_with_nan"):
            lines.append(f"{temp_name}_value = {temp_name}_value.replace(0.0, np.nan)")
        lines.append(f'minute_ctx[{_python_literal(name)}] = {temp_name}_value')
        return "\n".join(lines)

    if kind == "intraday_pct_aggregate":
        lag = feature["lag"]
        end_offset = feature.get("end_offset", 0)
        aggregation = feature["aggregation"]
        lines = [
            f"if len(minutes) <= {lag + end_offset}:",
            f'    raise ValueError("minute axis is shorter than required lag {lag} for {name}")',
            f"{temp_name}_upper_bound = len(minutes) - {end_offset}",
            f"{temp_name}_pct_frames = []",
            f"for {temp_name}_current_idx in range({lag}, {temp_name}_upper_bound):",
            f'    {temp_name}_current = load_single_minute(handle, {_python_literal(feature["field"])}, str(minutes[{temp_name}_current_idx]))',
            f'    {temp_name}_lagged = load_single_minute(handle, {_python_literal(feature["field"])}, str(minutes[{temp_name}_current_idx - {lag}]))',
        ]
        if feature.get("replace_zero_with_nan", True):
            lines.append(f"    {temp_name}_lagged = {temp_name}_lagged.replace(0.0, np.nan)")
        lines.extend(
            [
                f"    {temp_name}_pct_frames.append({temp_name}_current.divide({temp_name}_lagged).subtract(1.0))",
                f"{temp_name}_value = pd.concat({temp_name}_pct_frames).groupby(level=0).{aggregation}()",
            ]
        )
        if feature.get("replace_zero_with_nan"):
            lines.append(f"{temp_name}_value = {temp_name}_value.replace(0.0, np.nan)")
        lines.append(f'minute_ctx[{_python_literal(name)}] = {temp_name}_value')
        return "\n".join(lines)

    if kind == "engine_run":
        call_lines = [
            f"{temp_name}_value = mfe.run(",
            f"    inputs={_python_literal(feature['inputs'])},",
            f"    operator={_render_engine_operator_literal(feature['operator'], feature['operator_params'])},",
        ]
        preprocess = feature.get("preprocess")
        if preprocess is not None:
            call_lines.append(f"    preprocess={_python_literal(preprocess)},")
        aggregator = feature.get("aggregator")
        if aggregator is not None:
            call_lines.append(
                f"    aggregator={_render_engine_operator_literal(aggregator, feature['aggregator_params'])},"
            )
        for key, value in feature.get("kwargs", {}).items():
            call_lines.append(f"    {key}={_python_literal(value)},")
        call_lines.append(")")
        if feature.get("replace_zero_with_nan"):
            call_lines.append(f"{temp_name}_value = {temp_name}_value.replace(0.0, np.nan)")
        call_lines.append(f'minute_ctx[{_python_literal(name)}] = {temp_name}_value')
        return "\n".join(call_lines)

    if kind == "vector_expr":
        call_lines = [
            '__dates = pd.DatetimeIndex(handle["axis/dates"][:].astype(str))',
            '__stocks = pd.Index(handle["axis/stocks"][:].astype(str))',
            f"{temp_name}_value = np.full((len(__dates), len(__stocks)), np.nan, dtype=np.float32)",
        ]
        for field_name in feature["inputs"]:
            call_lines.append(f'__{name}_{field_name}_ds = handle["data/{field_name}"]')
        call_lines.append(f"for {temp_name}_day_idx in range(len(__dates)):")
        for field_name in feature["inputs"]:
            call_lines.append(
                f"    {field_name} = np.asarray(__{name}_{field_name}_ds[{temp_name}_day_idx, :, :], dtype=np.float32)"
            )
        call_lines.append(f"    {temp_name}_row = {feature['expression_code']}")
        call_lines.append(f"    {temp_name}_row = np.asarray({temp_name}_row, dtype=np.float32)")
        call_lines.append(f"    if {temp_name}_row.ndim != 1:")
        call_lines.append(
            f'        raise ValueError("vector_expr feature `{name}` must evaluate to a 1D stock vector per day")'
        )
        call_lines.append(f"    if {temp_name}_row.shape[0] != len(__stocks):")
        call_lines.append(
            f'        raise ValueError("vector_expr feature `{name}` returned wrong stock dimension")'
        )
        call_lines.append(f"    {temp_name}_value[{temp_name}_day_idx, :] = {temp_name}_row")
        call_lines.append(
            f"{temp_name}_frame = pd.DataFrame({temp_name}_value, index=__dates, columns=__stocks)"
        )
        if feature.get("replace_zero_with_nan"):
            call_lines.append(f"{temp_name}_frame = {temp_name}_frame.replace(0.0, np.nan)")
        call_lines.append(f'minute_ctx[{_python_literal(name)}] = {temp_name}_frame')
        return "\n".join(call_lines)

    if kind == "window_engine":
        window = feature["window"]
        end_offset = feature.get("end_offset", 0)
        call_lines = [
            f"{temp_name}_available_window = len(minutes) - {end_offset}",
            f"if {temp_name}_available_window <= 0:",
            f'    raise ValueError("minute axis is shorter than required offset {end_offset} for {name}")',
            f"{temp_name}_window = min({window}, {temp_name}_available_window)",
            f"{temp_name}_end_idx = len(minutes) - 1 - {end_offset}",
            f"{temp_name}_start_idx = {temp_name}_end_idx - {temp_name}_window + 1",
            f"{temp_name}_startminute = str(minutes[{temp_name}_start_idx])",
            f"{temp_name}_endminute = str(minutes[{temp_name}_end_idx])",
            f"{temp_name}_value = mfe.run(",
            f"    inputs={_python_literal(feature['inputs'])},",
            f"    operator={_render_engine_operator_literal(feature['operator'], feature['operator_params'])},",
            f"    aggregator={_render_engine_operator_literal(feature['aggregation'], feature['aggregator_params'])},",
        ]
        preprocess = feature.get("preprocess")
        if preprocess is not None:
            call_lines.append(f"    preprocess={_python_literal(preprocess)},")
        call_lines.extend(
            [
            f"    startminute={temp_name}_startminute,",
            f"    endminute={temp_name}_endminute,",
            ")",
            ]
        )
        if feature.get("replace_zero_with_nan"):
            call_lines.append(f"{temp_name}_value = {temp_name}_value.replace(0.0, np.nan)")
        call_lines.append(f'minute_ctx[{_python_literal(name)}] = {temp_name}_value')
        return "\n".join(call_lines)

    raise ValueError(f"unsupported minute feature kind `{kind}`")


def _render_structured_minute_module(target_task: FactorTask, spec: dict) -> str:
    normalized = _normalize_structured_minute_spec(spec, target_task=target_task)
    prepare_blocks = [_render_minute_feature_prepare_block(feature) for feature in normalized["minute_features"]]
    active_domains = factor_data_domains.resolve_factor_domains()
    template = joint_minute_code_template if is_joint_pv_minutes_run(active_domains) else minute_code_template
    minute_data_needed = _collect_structured_minute_inputs(normalized["minute_features"])
    return template.render(
        factor_name=normalized["factor_name"],
        original_expression=normalized["original_expression"],
        final_expression=normalized["final_expression"],
        data_needed_literal=_python_literal(normalized["data_needed"]),
        minute_data_needed_literal=_python_literal(minute_data_needed),
        prepare_blocks=prepare_blocks,
        render_failure_message_literal="None",
    )


def _render_structured_minute_failure_module(target_task: FactorTask, failure_reason: str | None) -> str:
    message = str(failure_reason or "Structured minute module render failed.").strip()
    active_domains = factor_data_domains.resolve_factor_domains()
    template = joint_minute_code_template if is_joint_pv_minutes_run(active_domains) else minute_code_template
    return template.render(
        factor_name=target_task.factor_name,
        original_expression=str(getattr(target_task, "factor_expression", "") or ""),
        final_expression="_raise_structured_render_failure()",
        data_needed_literal="[]",
        minute_data_needed_literal="[]",
        prepare_blocks=[],
        render_failure_message_literal=_python_literal(message),
    )


def _collect_structured_minute_inputs(features: list[dict]) -> list[str]:
    fields: set[str] = set()
    for feature in features:
        field_name = feature.get("field")
        if isinstance(field_name, str) and field_name:
            fields.add(field_name)
        inputs = feature.get("inputs")
        if isinstance(inputs, list):
            fields.update(str(item) for item in inputs if isinstance(item, str) and item)
    return sorted(fields)


def _render_expression_based_module(target_task: FactorTask) -> str | None:
    """
    Unified expression → template renderer shared by every strategy.

    - pv / fundamental: daily `code_template` (expression + data_needed).
    - minutes / joint : `minute_spec_compiler` → `_render_structured_minute_module`.

    Returns the rendered module text, or None if the expression cannot be handled
    deterministically (e.g. invalid pv expression, or minute expression whose shape
    the compiler does not yet cover). Callers decide whether to fall back to an
    LLM-assisted path or to emit a failure module. This function never raises.
    """
    expression = str(getattr(target_task, "factor_expression", "") or "").strip()
    if not expression:
        return None

    domains = factor_data_domains.resolve_factor_domains()
    if "minutes" in domains:
        try:
            compiled_spec = compile_expression_to_minute_spec(
                target_task,
                active_domains=domains,
            )
        except Exception:
            return None
        if compiled_spec is None:
            return None
        try:
            return _render_structured_minute_module(target_task, compiled_spec)
        except (ValueError, TypeError, KeyError):
            return None

    try:
        data_needed = _infer_expression_data_needed(expression)
    except Exception:
        return None
    domain_type = _daily_template_domain_type(domains)
    try:
        return _render_daily_expression_module(expression, target_task.factor_name, data_needed, domain_type)
    except Exception:
        return None


def _daily_template_domain_type(domains: tuple[str, ...]) -> str:
    resolved = factor_data_domains.parse_factor_domains(domains)
    if "fundamental" in resolved and len(resolved) == 1:
        return "fundamental"
    if "fundamental" in resolved and len(resolved) > 1:
        return "hybrid"
    return "pv"


def _build_deterministic_minute_spec(target_task: FactorTask) -> dict | None:
    expression = str(getattr(target_task, "factor_expression", "") or "").strip()
    if not expression:
        return None
    minute_fields = _minute_field_set()
    daily_pv_fields = _daily_pv_field_set()

    match = re.fullmatch(
        r"safe_div\(\s*ts_return\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        expression,
    )
    if match:
        minute_field, lag_text, daily_field = match.groups()
        if minute_field in minute_fields and daily_field in daily_pv_fields:
            feature_name = f"{minute_field}_return_{lag_text}"
            return {
                "data_needed": [daily_field],
                "minute_features": [
                    {
                        "name": feature_name,
                        "kind": "snapshot_return",
                        "field": minute_field,
                        "lag": int(lag_text),
                        "replace_zero_with_nan": True,
                    }
                ],
                "final_expression": (
                    f'{feature_name}.divide(data_ctx["{daily_field}"].replace(0.0, np.nan))'
                ),
            }

    match = re.fullmatch(
        r"safe_div\(\s*(?:(mean|sum|ts_mean))\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
        expression,
    )
    if match:
        agg_name, minute_field, window_text, daily_field = match.groups()
        if minute_field in minute_fields and daily_field in daily_pv_fields:
            feature_name = f"{minute_field}_{agg_name}_{window_text}"
            return {
                "data_needed": [daily_field],
                "minute_features": [
                    {
                        "name": feature_name,
                        "kind": "window_aggregate",
                        "field": minute_field,
                        "window": int(window_text),
                        "aggregation": "mean" if agg_name in {"mean", "ts_mean"} else "sum",
                    }
                ],
                "final_expression": (
                    f'{feature_name}.divide(data_ctx["{daily_field}"].replace(0.0, np.nan))'
                ),
            }

    match = re.fullmatch(r"ts_return\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)", expression)
    if match:
        minute_field, lag_text = match.groups()
        if minute_field in minute_fields:
            feature_name = f"{minute_field}_return_{lag_text}"
            return {
                "data_needed": [],
                "minute_features": [
                    {
                        "name": feature_name,
                        "kind": "snapshot_return",
                        "field": minute_field,
                        "lag": int(lag_text),
                    }
                ],
                "final_expression": feature_name,
            }

    match = re.fullmatch(r"(?:(mean|sum|ts_mean))\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)", expression)
    if match:
        agg_name, minute_field, window_text = match.groups()
        if minute_field in minute_fields:
            feature_name = f"{minute_field}_{agg_name}_{window_text}"
            return {
                "data_needed": [],
                "minute_features": [
                    {
                        "name": feature_name,
                        "kind": "window_aggregate",
                        "field": minute_field,
                        "window": int(window_text),
                        "aggregation": "mean" if agg_name in {"mean", "ts_mean"} else "sum",
                    }
                ],
                "final_expression": feature_name,
            }

    match = re.fullmatch(
        r"(?:(mean|sum|ts_mean))\(\s*pct\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)\s*\)",
        expression,
    )
    if match:
        agg_name, minute_field, lag_text = match.groups()
        if minute_field in minute_fields:
            feature_name = f"{minute_field}_pct_{agg_name}_{lag_text}"
            return {
                "data_needed": [],
                "minute_features": [
                    {
                        "name": feature_name,
                        "kind": "intraday_pct_aggregate",
                        "field": minute_field,
                        "lag": int(lag_text),
                        "aggregation": "mean" if agg_name in {"mean", "ts_mean"} else "sum",
                    }
                ],
                "final_expression": feature_name,
            }

    match = re.fullmatch(
        r"safe_div\(\s*ts_return\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)\s*,\s*ts_mean\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)\s*\)",
        expression,
    )
    if match:
        return_field, return_lag_text, mean_field, mean_window_text = match.groups()
        if return_field in minute_fields and mean_field in minute_fields:
            return_feature = f"{return_field}_return_{return_lag_text}"
            mean_feature = f"{mean_field}_mean_{mean_window_text}"
            return {
                "data_needed": [],
                "minute_features": [
                    {
                        "name": return_feature,
                        "kind": "snapshot_return",
                        "field": return_field,
                        "lag": int(return_lag_text),
                    },
                    {
                        "name": mean_feature,
                        "kind": "window_aggregate",
                        "field": mean_field,
                        "window": int(mean_window_text),
                        "aggregation": "mean",
                        "replace_zero_with_nan": True,
                    },
                ],
                "final_expression": f"{return_feature}.divide({mean_feature}.replace(0.0, np.nan))",
            }
    return None


def _build_structured_minute_prompt_payload(
    target_task: FactorTask,
    queried_similar_successful_knowledge: list,
    queried_similar_error_knowledge: dict,
    queried_former_failed_knowledge: list,
    latest_attempt_to_latest_successful_execution,
    scen,
) -> tuple[str, str]:
    scenario_text = scen.get_scenario_all_desc(target_task, filtered_tag="feature")
    schema_text = (
        "Return JSON only. Do not return Python code.\n"
        "Required JSON schema:\n"
        "{\n"
        '  "data_needed": ["hfq_closes"],\n'
        '  "minute_features": [\n'
        '    {"name": "close_ret_30", "kind": "snapshot_return", "field": "closes", "lag": 30},\n'
        '    {"name": "vol_mean_15", "kind": "window_aggregate", "field": "volumes", "window": 15, "aggregation": "mean"},\n'
        '    {"name": "close_1457", "kind": "snapshot", "field": "closes", "minute": "1457"},\n'
        '    {"name": "tail_vol_ratio", "kind": "engine_run", "inputs": ["volumes"], "operator": "mean", "kwargs": {"startminute": "1428", "endminute": "1457"}}\n'
        "  ],\n"
        '  "final_expression": \'close_ret_30.divide(data_ctx["turnovers"].replace(0.0, np.nan))\'\n'
        "}\n"
        "Feature-kind rules:\n"
        "- `snapshot`: load one raw minute slice via `load_single_minute`; use either `minute` or `offset`.\n"
        "- `snapshot_return`: compute the return between the latest snapshot and a lagged snapshot of the same raw minute field.\n"
        "- `window_aggregate`: aggregate the latest `window` snapshots of one raw minute field with one of mean/sum/min/max/std.\n"
        "- `intraday_pct_aggregate`: compute lagged intraday pct changes first, then aggregate them with mean/sum/min/max/std.\n"
        "- `engine_run`: call `MinuteFactorEngine.run(...)` declaratively with literal inputs/operator/aggregator/kwargs.\n"
        "Hard constraints:\n"
        "- `minute_features[*].name` must be a valid Python identifier.\n"
        "- Use `ts_return(field, lag)` only for lagged same-field returns. Do not overload `pct(field, lag)` for that meaning.\n"
        "- Use `pct(a, b)` only for same-time tensor ratio `a / b - 1` when both operands are raw minute tensors from the same intraday slice.\n"
        "- Raw minute fields may only appear inside `minute_features`; never use raw minute fields directly in `final_expression`.\n"
        "- Use daily pv fields only through `data_ctx[\"field\"]` inside `final_expression`, and declare them in `data_needed`.\n"
        "- In a pure minute task, `data_needed` must be empty and `final_expression` must use only rendered minute feature outputs.\n"
        "- In a joint pv/minutes task, `data_needed` must include at least one daily pv field, and `final_expression` must reference both minute feature outputs and at least one `data_ctx[\"daily_pv_field\"]`.\n"
        "- Once a minute feature has already been reduced to a daily 2D frame, any later `ts_rank/ts_zscore/ts_corr/...` in `final_expression` follows regular daily semantics, not intraday minute semantics.\n"
        "- Keep the spec executable. Use 1-3 minute features for simple ideas, and allow more only when each feature has a distinct role.\n"
    )
    system_prompt = (
        "You are converting a minute-domain factor idea into a structured execution spec for a fixed template.\n\n"
        f"Scenario:\n{scenario_text}\n\n"
        f"{build_runtime_field_constraints()}\n\n"
        f"{schema_text}"
    )

    prompt_sections = [
        "Target factor:",
        _build_factor_prompt_information(target_task),
    ]
    if queried_former_failed_knowledge:
        latest_failed = queried_former_failed_knowledge[-1]
        prompt_sections.extend(
            [
                "",
                "Latest failed implementation to correct:",
                latest_failed.implementation.code,
                "Latest failed feedback:",
                str(getattr(latest_failed, "feedback", "") or ""),
            ]
        )
    if latest_attempt_to_latest_successful_execution is not None:
        prompt_sections.extend(
            [
                "",
                "Latest execution-compatible attempt and feedback:",
                str(getattr(latest_attempt_to_latest_successful_execution.implementation, "code", "") or ""),
                str(getattr(latest_attempt_to_latest_successful_execution, "feedback", "") or ""),
            ]
        )
    if queried_similar_successful_knowledge:
        prompt_sections.extend(["", "Reference successful minute implementations:"])
        for knowledge in queried_similar_successful_knowledge[:2]:
            prompt_sections.extend(
                [
                    knowledge.target_task.get_task_description(),
                    knowledge.implementation.code,
                ]
            )
    flattened_errors = _flatten_similar_error_knowledge(queried_similar_error_knowledge)
    if flattened_errors:
        prompt_sections.extend(["", "Similar failure-to-fix references:"])
        for failed_knowledge, fixed_knowledge in flattened_errors[:2]:
            prompt_sections.extend(
                [
                    "Failed example:",
                    failed_knowledge.implementation.code,
                    str(getattr(failed_knowledge, "feedback", "") or ""),
                    "Corrected example:",
                    fixed_knowledge.implementation.code,
                ]
            )
    user_prompt = "\n".join(section for section in prompt_sections if section is not None).strip()
    return system_prompt, user_prompt


def _try_render_structured_minute_module(
    target_task: FactorTask,
    queried_similar_successful_knowledge: list,
    queried_similar_error_knowledge: dict,
    queried_former_failed_knowledge: list,
    latest_attempt_to_latest_successful_execution,
    scen,
) -> tuple[str | None, str | None]:
    """Deterministic minute module renderer.

    The LLM is intentionally not invoked here. Unsupported expression shapes
    produce a (None, error) result so the caller can surface a template-based
    failure module for evaluator feedback, and the next loop iteration can ask
    the upstream proposal stage for a simpler, compiler-friendly expression.
    """
    compiled_spec = compile_expression_to_minute_spec(
        target_task,
        active_domains=factor_data_domains.resolve_factor_domains(),
    )
    if compiled_spec is not None:
        try:
            return _render_structured_minute_module(target_task, compiled_spec), None
        except (ValueError, TypeError, KeyError) as exc:
            compiled_spec_error = str(exc).strip() or exc.__class__.__name__
    else:
        compiled_spec_error = None

    deterministic_spec = _build_deterministic_minute_spec(target_task)
    if deterministic_spec is not None:
        try:
            return _render_structured_minute_module(target_task, deterministic_spec), None
        except (ValueError, TypeError, KeyError) as exc:
            deterministic_error = str(exc).strip() or exc.__class__.__name__
    else:
        deterministic_error = None

    return None, compiled_spec_error or deterministic_error or (
        "minute_spec_compiler could not handle the current expression shape; "
        "upstream must produce a simpler expression using registered minute operators."
    )

class FactorMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_loop = 0
        self.haveSelected = False

    def extract_expr(self, code_str: str) -> str:
        return _extract_expr_from_code(code_str)


    def error_summary(
        self,
        target_task: FactorTask,
        queried_former_failed_knowledge_to_render: list,
        queried_similar_error_knowledge_to_render: list,
    ) -> str:
        error_summary_system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(implement_prompts["evolving_strategy_error_summary_v2_system"])
            .render(
                scenario=self.scen.get_scenario_all_desc(target_task),
                factor_information_str=target_task.get_task_information(),
                code_and_feedback=_format_implementation_and_feedback_for_prompt(
                    queried_former_failed_knowledge_to_render[-1]
                ),
            )
            .strip("\n")
        )
        for _ in range(10):  # max attempt to reduce the length of error_summary_user_prompt
            error_summary_user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(implement_prompts["evolving_strategy_error_summary_v2_user"])
                .render(
                    queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
                )
                .strip("\n")
            )
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt
                )
                < LLM_SETTINGS.chat_token_limit
            ):
                break
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]
        error_summary_critics = APIBackend(
            use_chat_cache=FACTOR_COSTEER_SETTINGS.coder_use_cache
        ).build_messages_and_create_chat_completion(
            user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt, json_mode=False
        )
        return _sanitize_feedback_text_for_prompt(error_summary_critics)

    def implement_one_task(
        self,
        target_task: FactorTask,
        queried_knowledge: CoSTEERQueriedKnowledge,
    ) -> str:
        # First-tier: deterministic expression → template path. This covers
        # pv/fundamental outright and the subset of minute/joint expressions
        # `minute_spec_compiler` can handle without any LLM help.
        rendered = _render_expression_based_module(target_task)
        if rendered is not None:
            return rendered

        # For non-minute domains, reach here only with a malformed expression.
        # Re-raise via the canonical registry check so the caller sees a precise
        # error instead of a silent failure. (_render_expression_based_module
        # swallowed the exception above; we want it surfaced.)
        if not _minute_domain_active():
            expression = str(target_task.factor_expression or "").strip()
            data_needed = _infer_expression_data_needed(expression)  # raises on invalid
            return _render_daily_expression_module(
                expression,
                target_task.factor_name,
                data_needed,
                _daily_template_domain_type(factor_data_domains.resolve_factor_domains()),
            )

        # Second-tier (minute/joint only): LLM-assisted JSON-spec path. The helper
        # internally calls the compiler once more and falls back to LLM-produced
        # structured specs for shapes the compiler does not yet cover.
        target_factor_task_information = target_task.get_task_information()
        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[target_factor_task_information]
            if queried_knowledge is not None
            else []
        )
        if isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2):
            queried_similar_error_knowledge = (
                queried_knowledge.task_to_similar_error_successful_knowledge[target_factor_task_information]
                if queried_knowledge is not None
                else {}
            )
        else:
            queried_similar_error_knowledge = {}
        queried_similar_error_knowledge = _sanitize_similar_error_knowledge_pairs(queried_similar_error_knowledge)

        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[target_factor_task_information][0]
            if queried_knowledge is not None
            else []
        )
        queried_former_failed_knowledge = [
            _clone_knowledge_with_sanitized_feedback(knowledge) for knowledge in queried_former_failed_knowledge
        ]

        latest_attempt_to_latest_successful_execution = _clone_knowledge_with_sanitized_feedback(
            queried_knowledge.task_to_former_failed_traces[target_factor_task_information][1]
            if queried_knowledge is not None
            else None
        )

        structured_module, structured_error = _try_render_structured_minute_module(
            target_task=target_task,
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
            queried_similar_error_knowledge=queried_similar_error_knowledge,
            queried_former_failed_knowledge=queried_former_failed_knowledge,
            latest_attempt_to_latest_successful_execution=latest_attempt_to_latest_successful_execution,
            scen=self.scen,
        )
        if structured_module is not None:
            return structured_module

        # Third-tier: template-based failure module so evaluator feedback can
        # repair the expression. The LLM is never asked to emit raw python.
        logger.warning(
            f"Structured minute module render failed for {target_task.factor_name}; "
            f"returning a template-based failure module for evaluator feedback. "
            f"reason={structured_error or 'unknown'}"
        )
        return _render_structured_minute_failure_module(target_task, structured_error)

    def assign_code_list_to_evo(self, code_list, evo):
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = FactorFBWorkspace(target_task=evo.sub_tasks[index])
            evo.sub_workspace_list[index].inject_code(**{"factor.py": code_list[index]})
        return evo



qa_implement_prompts = DomainPromptProxy(base_file="qa_prompts.yaml", prompt_dir=Path(__file__).parent)
class FactorParsingStrategy(MultiProcessEvolvingStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_loop = 0
        self.haveSelected = False

    def extract_expr(self, code_str: str) -> str:
        return _extract_expr_from_code(code_str)


    def implement_one_task(
        self,
        target_task: FactorTask,
        queried_knowledge: CoSTEERQueriedKnowledge,
    ) -> str:
        """Generate code for one factor task.

        First run: expression → template (domain-aware). On failure history, ask
        the LLM to correct the expression and re-render through the same template
        path. Never let the LLM produce raw python.
        """
        target_factor_task_information = target_task.get_task_information()

        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[target_factor_task_information]
            if queried_knowledge is not None
            else []
        )

        if isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2):
            queried_similar_error_knowledge = (
                queried_knowledge.task_to_similar_error_successful_knowledge[target_factor_task_information]
                if queried_knowledge is not None
                else {}
            )
        else:
            queried_similar_error_knowledge = {}
        queried_similar_error_knowledge = _sanitize_similar_error_knowledge_pairs(queried_similar_error_knowledge)

        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[target_factor_task_information][0]
            if queried_knowledge is not None
            else []
        )
        queried_former_failed_knowledge = [
            _clone_knowledge_with_sanitized_feedback(knowledge) for knowledge in queried_former_failed_knowledge
        ]

        queried_former_failed_knowledge_to_render = queried_former_failed_knowledge

        # First-tier: deterministic expression → template path for the current domain.
        if len(queried_former_failed_knowledge) == 0:
            rendered = _render_expression_based_module(target_task)
            if rendered is not None:
                return rendered
            if _minute_domain_active():
                structured_module, structured_error = _try_render_structured_minute_module(
                    target_task=target_task,
                    queried_similar_successful_knowledge=queried_similar_successful_knowledge,
                    queried_similar_error_knowledge=queried_similar_error_knowledge,
                    queried_former_failed_knowledge=queried_former_failed_knowledge,
                    latest_attempt_to_latest_successful_execution=None,
                    scen=self.scen,
                )
                if structured_module is not None:
                    return structured_module
                return _render_structured_minute_failure_module(target_task, structured_error)
            # pv/fundamental first-time invalid expression: surface the canonical
            # registry error as evaluator feedback instead of killing the worker.
            expression = str(target_task.factor_expression or "").strip()
            try:
                data_needed = _infer_expression_data_needed(expression)
            except Exception as exc:
                return _render_daily_expression_failure_module(
                    target_task,
                    f"invalid expression: {exc}",
                )
            return _render_daily_expression_module(
                expression,
                target_task.factor_name,
                data_needed,
                _daily_template_domain_type(factor_data_domains.resolve_factor_domains()),
            )

        latest_attempt_to_latest_successful_execution = _clone_knowledge_with_sanitized_feedback(
            queried_knowledge.task_to_former_failed_traces[target_factor_task_information][1]
            if queried_knowledge is not None
            else None
        )

        # Minute/joint retry path: defer to the structured-spec helper (compiler
        # → LLM JSON spec → failure module). No raw python is ever generated.
        if _minute_domain_active():
            structured_module, structured_error = _try_render_structured_minute_module(
                target_task=target_task,
                queried_similar_successful_knowledge=queried_similar_successful_knowledge,
                queried_similar_error_knowledge=queried_similar_error_knowledge,
                queried_former_failed_knowledge=queried_former_failed_knowledge,
                latest_attempt_to_latest_successful_execution=latest_attempt_to_latest_successful_execution,
                scen=self.scen,
            )
            if structured_module is not None:
                return structured_module
            return _render_structured_minute_failure_module(target_task, structured_error)

        # pv / fundamental retry path: ask the LLM for a corrected expression
        # only, then render through the same deterministic daily template.
        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(
                qa_implement_prompts["evolving_strategy_factor_implementation_v1_system"],
            )
            .render(
                scenario=self.scen.get_scenario_all_desc(target_task, filtered_tag="feature"),
            )
        )
        system_prompt = f"{system_prompt.rstrip()}\n\n{build_runtime_field_constraints()}"
        queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
        queried_similar_error_knowledge_to_render = queried_similar_error_knowledge

        for _ in range(10):
            if (
                isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2)
                and FACTOR_COSTEER_SETTINGS.v2_error_summary
                and len(queried_similar_error_knowledge_to_render) != 0
                and len(queried_former_failed_knowledge_to_render) != 0
            ):
                error_summary_critics = self.error_summary(
                    target_task,
                    queried_former_failed_knowledge_to_render,
                    queried_similar_error_knowledge_to_render,
                )
            else:
                error_summary_critics = None

            similar_successful_factor_description = ""
            similar_successful_expression = ""
            if len(queried_similar_successful_knowledge_to_render) > 0:
                similar_successful_factor_description = queried_similar_successful_knowledge_to_render[-1].target_task.get_task_description()
                similar_successful_expression = self.extract_expr(queried_similar_successful_knowledge_to_render[-1].implementation.code)

            user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(
                    qa_implement_prompts["evolving_strategy_factor_implementation_v2_user"],
                )
                .render(
                    factor_information_str=_build_factor_prompt_information(target_task),
                    queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
                    former_expression=self.extract_expr(queried_former_failed_knowledge_to_render[-1].implementation.code) if queried_former_failed_knowledge_to_render else "",
                    former_feedback=queried_former_failed_knowledge_to_render[-1].feedback if queried_former_failed_knowledge_to_render else "",
                    error_summary_critics=error_summary_critics,
                    similar_successful_factor_description=similar_successful_factor_description,
                    similar_successful_expression=similar_successful_expression,
                    latest_attempt_to_latest_successful_execution=latest_attempt_to_latest_successful_execution,
                )
                .strip("\n")
            )

            if (
                APIBackend().build_messages_and_calculate_token(user_prompt=user_prompt, system_prompt=system_prompt)
                < LLM_SETTINGS.chat_token_limit
            ):
                break
            elif len(queried_former_failed_knowledge_to_render) > 1:
                queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
            elif len(queried_similar_successful_knowledge_to_render) > len(
                queried_similar_error_knowledge_to_render,
            ):
                queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge_to_render[:-1]
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]

        for _ in range(10):
            try:
                expr = json.loads(
                    APIBackend(
                        use_chat_cache=FACTOR_COSTEER_SETTINGS.coder_use_cache
                    ).build_messages_and_create_chat_completion(
                        user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True, reasoning_flag=False
                    )
                )["expr"]
                # Re-render through the domain-aware helper. For pv this is the
                # daily template; for fundamental the same daily template; for
                # minute/joint this path is unreachable because we branched out
                # above, so this call would only be the pv/fundamental daily path.
                target_task.factor_expression = expr
                rendered = _render_expression_based_module(target_task)
                if rendered is not None:
                    return rendered
            except json.decoder.JSONDecodeError:
                pass
    
    def assign_code_list_to_evo(self, code_list, evo):
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = FactorFBWorkspace(target_task=evo.sub_tasks[index])
            evo.sub_workspace_list[index].inject_code(**{"factor.py": code_list[index]})
        return evo
    
    
    
class FactorRunningStrategy(MultiProcessEvolvingStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_loop = 0
        self.haveSelected = False


    def implement_one_task(
        self,
        target_task: FactorTask,
        queried_knowledge: CoSTEERQueriedKnowledge,
    ) -> str:
        rendered = _render_expression_based_module(target_task)
        if rendered is not None:
            return rendered
        # Compiler-driven renderer returned None. Minute/joint: emit a failure
        # module so the evaluator can provide corrective feedback on the next
        # loop. pv/fundamental: propagate the canonical registry error.
        if _minute_domain_active():
            return _render_structured_minute_failure_module(
                target_task,
                "Expression compiler could not produce a deterministic minute spec; "
                "rephrase the expression using registered minute operators on raw minute fields.",
            )
        expression = str(getattr(target_task, "factor_expression", "") or "").strip()
        try:
            data_needed = _infer_expression_data_needed(expression)
        except Exception as exc:
            return _render_daily_expression_failure_module(
                target_task,
                f"invalid expression: {exc}",
            )
        return _render_daily_expression_module(
            expression,
            target_task.factor_name,
            data_needed,
            _daily_template_domain_type(factor_data_domains.resolve_factor_domains()),
        )
        
    
    def assign_code_list_to_evo(self, code_list, evo):
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = FactorFBWorkspace(target_task=evo.sub_tasks[index])
            evo.sub_workspace_list[index].inject_code(**{"factor.py": code_list[index]})
        return evo
    
    
    def evolve(
        self,
        *,
        evo: EvolvingItem,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        **kwargs,
    ) -> EvolvingItem:
        # Find tasks to evolve
        to_be_finished_task_index = []
        for index, target_task in enumerate(evo.sub_tasks):
            to_be_finished_task_index.append(index)

        result = multiprocessing_wrapper(
            [
                (self.implement_one_task, (evo.sub_tasks[target_index], queried_knowledge))
                for target_index in to_be_finished_task_index
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )
        code_list = [None for _ in range(len(evo.sub_tasks))]
        for index, target_index in enumerate(to_be_finished_task_index):
            code_list[target_index] = result[index]

        evo = self.assign_code_list_to_evo(code_list, evo)
        evo.corresponding_selection = to_be_finished_task_index

        return evo
