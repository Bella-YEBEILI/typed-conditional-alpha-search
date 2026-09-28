"""Classify shared fields (volumes, vwaps, turnovers) into DAILY / MINUTE / BOTH
based on their position in the expression AST.

This module mirrors the domain-routing logic that ``minute_spec_compiler``
applies implicitly via the ``allow_shared_fields`` flag, but exposes the
classification *explicitly* so other pipeline stages (prompt construction,
contract validation, normaliser) can consume it.

Only meaningful in *joint pv/minutes* mode — callers must gate invocations
behind ``is_joint_pv_minutes_run()``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Literal

from quantaalpha.factors.coder.factor_ast import (
    BinaryOpNode,
    ConditionalNode,
    FunctionNode,
    ListNode,
    Node,
    NumberNode,
    UnaryOpNode,
    VarNode,
    parse_expression,
)
from quantaalpha.factors.coder.minute_op_registry import get_minute_op_spec
from quantaalpha.factors import data_domains as factor_data_domains


FieldDomain = Literal["DAILY", "MINUTE", "BOTH"]

# ---------------------------------------------------------------------------
# Minute-reducer detection — kept in sync with minute_spec_compiler
# ---------------------------------------------------------------------------

# Same map used by _infer_minute_subtree_shape for window-aggregate reducers.
_WINDOW_AGGREGATION_NAMES: frozenset[str] = frozenset(
    {
        "mean", "sum", "min", "max", "std", "kurt",
        "ts_mean", "ts_sum", "ts_std", "ts_min", "ts_max", "ts_kurt",
    }
)

MAX_INTRADAY_LAG = 1440  # mirrors minute_spec_compiler.MAX_INTRADAY_LAG


def _is_positive_int_node(node: Node) -> bool:
    """True when *node* is a positive integer literal (NumberNode > 0)."""
    return isinstance(node, NumberNode) and float(node.value) > 0


def _has_intraday_lag(node: Node) -> bool:
    """True when *node* is a positive integer within minute-bar range."""
    if not isinstance(node, NumberNode):
        return False
    try:
        value = int(float(node.value))
    except (ValueError, TypeError):
        return False
    return 0 < value <= MAX_INTRADAY_LAG


def _is_minute_reducer_call(node: FunctionNode) -> tuple[bool, int]:
    """Decide whether *node* creates a minute-context for its field arguments.

    Returns ``(is_reducer, field_arg_count)`` where *field_arg_count* is the
    number of leading arguments that live in minute context.

    The logic mirrors every ``allow_shared_fields=True`` trigger point inside
    ``minute_spec_compiler._infer_minute_subtree_shape``:

    1. ``_WINDOW_AGGREGATION_MAP`` entries with 2 args (field, window).
    2. ``ts_return(field, lag)`` when lag is an intraday-range positive int.
    3. ``pct(field, lag)`` when lag is an intraday-range positive int.
    4. Any registered minute op via ``get_minute_op_spec`` — all *arity*
       positional args enter minute context.
    """
    name = str(node.name)
    args = list(node.args)
    nargs = len(args)

    # 1) Window-aggregate reducers: func(field, window)
    if name in _WINDOW_AGGREGATION_NAMES and nargs == 2 and _is_positive_int_node(args[1]):
        return True, 1  # first arg is the field

    # 2) ts_return(field, lag)
    if name == "ts_return" and nargs == 2 and _has_intraday_lag(args[1]):
        return True, 1

    # 3) pct(field, lag) — only when lag looks like an intraday offset
    if name == "pct" and nargs == 2 and _has_intraday_lag(args[1]):
        return True, 1

    # 4) Registered minute ops (reduce, special, pointwise, cross_sectional)
    spec = get_minute_op_spec(name)
    if spec is not None and nargs == spec.total_args:
        return True, spec.arity  # first `arity` args are field inputs

    return False, 0


# ---------------------------------------------------------------------------
# safe_div / WHERE — these are passthrough (not reducers), but inside a minute
# subtree they keep allow_shared_fields propagating.  The compiler treats them
# by recursing with the SAME allow_shared_fields as the parent.  We mirror
# that by propagating ``in_minute_context`` unchanged.
# ---------------------------------------------------------------------------

_WHERE_FUNCTION_NAMES: frozenset[str] = frozenset(
    {"WHERE", "where", "IF_ELSE", "if_else", "ifelse"}
)


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

def _resolve_shared_fields() -> frozenset[str]:
    """Return the set of field names that exist in both PV_FIELDS and MINUTE_FIELDS."""
    shared = getattr(factor_data_domains, "JOINT_PV_MINUTES_SHARED_FIELDS", None)
    if shared is not None:
        return frozenset(shared)
    pv = set(getattr(factor_data_domains, "PV_FIELDS", ()))
    minute = set(getattr(factor_data_domains, "MINUTE_FIELDS", ()))
    return frozenset(pv & minute)


def classify_shared_fields(
    expression: str,
    *,
    shared_fields: frozenset[str] | None = None,
) -> dict[str, FieldDomain]:
    """Classify each shared field's domain usage within *expression*.

    Parameters
    ----------
    expression:
        The factor expression string (parsed via ``factor_ast.parse_expression``).
    shared_fields:
        Override the set of shared field names.  Defaults to
        ``JOINT_PV_MINUTES_SHARED_FIELDS`` from ``data_domains``.

    Returns
    -------
    A mapping ``{field_name: "DAILY" | "MINUTE" | "BOTH"}`` for every shared
    field that actually appears in the expression.  Fields not referenced are
    omitted.
    """
    if shared_fields is None:
        shared_fields = _resolve_shared_fields()

    try:
        root = parse_expression(expression)
    except Exception:
        return {}

    # Collect per-occurrence domain tags.
    usages: dict[str, set[str]] = defaultdict(set)

    def walk(node: Node, in_minute_context: bool) -> None:
        if isinstance(node, VarNode):
            if node.name in shared_fields:
                usages[node.name].add("MINUTE" if in_minute_context else "DAILY")
            return

        if isinstance(node, NumberNode):
            return

        if isinstance(node, UnaryOpNode):
            walk(node.operand, in_minute_context)
            return

        if isinstance(node, BinaryOpNode):
            walk(node.left, in_minute_context)
            walk(node.right, in_minute_context)
            return

        if isinstance(node, ConditionalNode):
            walk(node.condition, in_minute_context)
            walk(node.true_expr, in_minute_context)
            walk(node.false_expr, in_minute_context)
            return

        if isinstance(node, ListNode):
            for item in node.items:
                walk(item, in_minute_context)
            return

        if isinstance(node, FunctionNode):
            is_reducer, field_arg_count = _is_minute_reducer_call(node)
            args = list(node.args)

            if is_reducer:
                # The first `field_arg_count` args live in minute context;
                # remaining args (params like window, lag) keep parent context.
                for i, arg in enumerate(args):
                    if i < field_arg_count:
                        walk(arg, in_minute_context=True)
                    else:
                        walk(arg, in_minute_context)
            elif str(node.name) in _WHERE_FUNCTION_NAMES or str(node.name) == "safe_div":
                # Passthrough — propagate parent context unchanged.
                for arg in args:
                    walk(arg, in_minute_context)
            else:
                # Non-reducer, non-passthrough functions (cs_zscore, ts_corr at
                # daily level, user-defined, etc.) — propagate parent context.
                for arg in args:
                    walk(arg, in_minute_context)

            return

    walk(root, in_minute_context=False)

    # Collapse per-occurrence tags into a single domain.
    result: dict[str, FieldDomain] = {}
    for field, tags in sorted(usages.items()):
        if tags == {"DAILY"}:
            result[field] = "DAILY"
        elif tags == {"MINUTE"}:
            result[field] = "MINUTE"
        else:
            result[field] = "BOTH"

    return result


def classify_expression_domains(
    expression: str,
) -> dict[str, object]:
    """Full domain analysis of an expression for joint pv/minutes validation.

    Returns a dict with:
    - ``shared_field_usage``: per-field classification (DAILY / MINUTE / BOTH)
    - ``has_daily``: whether any field contributes to the daily side
    - ``has_minute``: whether any field contributes to the minute side
    - ``daily_evidence``: list of evidence strings
    - ``minute_evidence``: list of evidence strings
    """
    pv_exclusive = set(getattr(factor_data_domains, "JOINT_PV_EXCLUSIVE_FIELDS", ()))
    minute_exclusive = set(getattr(factor_data_domains, "JOINT_MINUTE_EXCLUSIVE_FIELDS", ()))
    shared_fields = _resolve_shared_fields()

    # Parse to extract all used fields.
    try:
        root = parse_expression(expression)
    except Exception:
        return {
            "shared_field_usage": {},
            "has_daily": False,
            "has_minute": False,
            "daily_evidence": [],
            "minute_evidence": [],
        }

    used_fields: set[str] = set()

    def collect_vars(node: Node) -> None:
        if isinstance(node, VarNode):
            used_fields.add(node.name)
        elif isinstance(node, FunctionNode):
            for arg in node.args:
                collect_vars(arg)
        elif isinstance(node, BinaryOpNode):
            collect_vars(node.left)
            collect_vars(node.right)
        elif isinstance(node, UnaryOpNode):
            collect_vars(node.operand)
        elif isinstance(node, ConditionalNode):
            collect_vars(node.condition)
            collect_vars(node.true_expr)
            collect_vars(node.false_expr)
        elif isinstance(node, ListNode):
            for item in node.items:
                collect_vars(item)

    collect_vars(root)

    classification = classify_shared_fields(expression, shared_fields=shared_fields)

    # Determine daily / minute presence.
    daily_evidence: list[str] = []
    minute_evidence: list[str] = []

    pv_ex_used = sorted(used_fields & pv_exclusive)
    min_ex_used = sorted(used_fields & minute_exclusive)

    if pv_ex_used:
        daily_evidence.append(f"pv_exclusive: {pv_ex_used}")
    if min_ex_used:
        minute_evidence.append(f"minute_exclusive: {min_ex_used}")

    for field, domain in sorted(classification.items()):
        if domain in ("DAILY", "BOTH"):
            daily_evidence.append(f"{field}={domain}")
        if domain in ("MINUTE", "BOTH"):
            minute_evidence.append(f"{field}={domain}")

    has_daily = bool(daily_evidence)
    has_minute = bool(minute_evidence)

    return {
        "shared_field_usage": classification,
        "has_daily": has_daily,
        "has_minute": has_minute,
        "daily_evidence": daily_evidence,
        "minute_evidence": minute_evidence,
        "pv_exclusive_used": pv_ex_used,
        "minute_exclusive_used": min_ex_used,
        "used_fields": sorted(used_fields),
    }
