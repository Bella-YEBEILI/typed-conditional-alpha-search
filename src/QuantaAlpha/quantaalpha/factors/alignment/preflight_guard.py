from typing import Optional, Set

from quantaalpha.factors.coder.factor_ast import (
    FunctionNode,
    VarNode,
    BinaryOpNode,
    ConditionalNode,
    UnaryOpNode,
    NumberNode,
    ListNode,
    parse_expression as parse_expression_ast,
)
from .registry import get_allowed_field_names, get_allowed_operator_names, get_operator_arity_constraints

_KNOWN_CONSTANTS = frozenset({
    "NaN", "nan", "NAN",
    "null", "NULL",
    "inf", "Inf", "INF",
    "True", "true", "TRUE",
    "False", "false", "FALSE",
})


def _normalize_var_name(name: str) -> str:
    return name[1:] if name.startswith("$") else name


def _collect_symbols(expression: str) -> tuple[set[str], set[str], Optional[str]]:
    try:
        tree = parse_expression_ast(expression)
    except Exception as e:
        return set(), set(), str(e)

    funcs: Set[str] = set()
    vars_: Set[str] = set()

    def walk(node):
        if isinstance(node, FunctionNode):
            funcs.add(str(node.name))
            for arg in node.args:
                walk(arg)
        elif isinstance(node, ListNode):
            for item in node.items:
                walk(item)
        elif isinstance(node, VarNode):
            normalized = _normalize_var_name(str(node.name))
            if normalized not in _KNOWN_CONSTANTS:
                vars_.add(normalized)
        elif isinstance(node, BinaryOpNode):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, ConditionalNode):
            walk(node.condition)
            walk(node.true_expr)
            walk(node.false_expr)
        elif isinstance(node, UnaryOpNode):
            walk(node.operand)

    walk(tree)
    return funcs, vars_, None


_NON_NEGATIVE_NUMERIC_ARG_RULES: dict[str, tuple[int, ...]] = {
    "delay": (1,),
    "ts_delay": (1,),
    "delta": (1,),
    "ts_delta": (1,),
    "ts_pct": (1,),
    "ts_return": (1,),
}

_PAIRWISE_TS_KERNELS: set[str] = {
    "ts_corr",
    "ts_covariance",
    "ts_regbeta",
    "ts_regresi",
}


def _extract_numeric_literal(node) -> float | None:
    if isinstance(node, NumberNode):
        return float(node.value)
    if isinstance(node, UnaryOpNode) and str(node.op) == "-" and isinstance(node.operand, NumberNode):
        return -float(node.operand.value)
    return None


def _contains_cross_sectional_aggregator(node) -> bool:
    if isinstance(node, FunctionNode):
        if str(node.name).startswith("cs_"):
            return True
        return any(_contains_cross_sectional_aggregator(arg) for arg in node.args)
    if isinstance(node, ListNode):
        return any(_contains_cross_sectional_aggregator(item) for item in node.items)
    if isinstance(node, BinaryOpNode):
        return _contains_cross_sectional_aggregator(node.left) or _contains_cross_sectional_aggregator(node.right)
    if isinstance(node, ConditionalNode):
        return (
            _contains_cross_sectional_aggregator(node.condition)
            or _contains_cross_sectional_aggregator(node.true_expr)
            or _contains_cross_sectional_aggregator(node.false_expr)
        )
    if isinstance(node, UnaryOpNode):
        return _contains_cross_sectional_aggregator(node.operand)
    return False


def _collect_semantic_errors(expression: str) -> list[str]:
    try:
        tree = parse_expression_ast(expression)
    except Exception:
        return []

    errors: list[str] = []
    arity_constraints = get_operator_arity_constraints()

    def walk(node):
        if isinstance(node, FunctionNode):
            fname = str(node.name)
            arity = arity_constraints.get(fname)
            if arity is not None:
                min_args, max_args = arity
                actual_args = len(node.args)
                if actual_args < min_args or (max_args is not None and actual_args > max_args):
                    if max_args is None:
                        expected = f"at least {min_args}"
                    elif min_args == max_args:
                        expected = str(min_args)
                    else:
                        expected = f"{min_args} to {max_args}"
                    errors.append(f"{fname} expects {expected} args, got {actual_args}")
            for arg_pos in _NON_NEGATIVE_NUMERIC_ARG_RULES.get(str(node.name), ()):
                if len(node.args) <= arg_pos:
                    continue
                arg = node.args[arg_pos]
                numeric_value = _extract_numeric_literal(arg)
                if numeric_value is not None and numeric_value < 0:
                    errors.append(
                        f"{fname} arg#{arg_pos + 1} must be >= 0 to avoid lookahead leakage, got {numeric_value}"
                    )

            if fname == "winsorize1" and len(node.args) >= 2:
                numeric_value = _extract_numeric_literal(node.args[1])
                if numeric_value is not None and 0 < numeric_value < 1:
                    errors.append(
                        "winsorize1 arg#2 is an n-sigma clip width, not a quantile; "
                        "use winsorize2(lower_bound, upper_bound) for percentile-style clipping"
                    )

            if fname in _PAIRWISE_TS_KERNELS and len(node.args) >= 2:
                if _contains_cross_sectional_aggregator(node.args[0]) or _contains_cross_sectional_aggregator(node.args[1]):
                    errors.append(
                        f"{fname} args #1/#2 cannot include cs_* aggregators because they collapse instrument dimension and can crash pairwise time-series kernels"
                    )

            for arg in node.args:
                walk(arg)
        elif isinstance(node, ListNode):
            for item in node.items:
                walk(item)
        elif isinstance(node, BinaryOpNode):
            walk(node.left)
            walk(node.right)
        elif isinstance(node, ConditionalNode):
            walk(node.condition)
            walk(node.true_expr)
            walk(node.false_expr)
        elif isinstance(node, UnaryOpNode):
            walk(node.operand)

    walk(tree)
    return errors


def validate_expression_against_registry(
    expression: str,
    allowed_operators: Optional[Set[str]] = None,
    allowed_fields: Optional[Set[str]] = None,
) -> dict:
    ops = set(allowed_operators) if allowed_operators is not None else get_allowed_operator_names()
    flds = set(allowed_fields) if allowed_fields is not None else get_allowed_field_names()

    called, used_vars, parse_error = _collect_symbols(expression)
    if parse_error is not None:
        return {
            "ok": False,
            "parse_error": parse_error,
            "operators_allowed": False,
            "fields_allowed": False,
            "unsupported_operators": [f"<parse_error:{parse_error}>"],
            "unsupported_fields": [f"<parse_error:{parse_error}>"],
            "called_operators": [],
            "used_fields": [],
        }

    unsupported_ops = sorted([x for x in called if x not in ops])
    unsupported_fields = sorted([x for x in used_vars if x not in flds])
    semantic_errors = _collect_semantic_errors(expression)
    surfaced_operator_errors = unsupported_ops + [f"<semantic:{msg}>" for msg in semantic_errors]

    return {
        "ok": len(surfaced_operator_errors) == 0 and len(unsupported_fields) == 0,
        "parse_error": None,
        "operators_allowed": len(surfaced_operator_errors) == 0,
        "fields_allowed": len(unsupported_fields) == 0,
        "unsupported_operators": surfaced_operator_errors,
        "unsupported_fields": unsupported_fields,
        "called_operators": sorted(called),
        "used_fields": sorted(used_vars),
        "semantic_errors": semantic_errors,
    }
