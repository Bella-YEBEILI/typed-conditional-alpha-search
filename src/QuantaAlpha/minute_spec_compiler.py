from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

import quantaalpha.backtest.minute_ops  # noqa: F401
from quantaalpha.backtest.minute_tools import MinuteFactorEngine
from quantaalpha.factors.coder.factor import FactorTask
from quantaalpha.factors.coder.factor_ast import (
    BinaryOpNode,
    ConditionalNode,
    FunctionNode,
    Node,
    NumberNode,
    UnaryOpNode,
    VarNode,
    parse_expression,
)
from quantaalpha.factors import data_domains as factor_data_domains


_WINDOW_AGGREGATION_MAP = {
    "mean": "mean",
    "sum": "sum",
    "min": "min",
    "max": "max",
    "std": "std",
    "kurt": "kurt",
    "ts_mean": "mean",
    "ts_std": "std",
}
_VECTOR_OUTPUT_MINUTE_OPS = {
    "mean",
    "std",
    "kurt",
    "sum",
    "min",
    "max",
    "last",
    "corr",
    "calc_guiji_impl",
    "calc_wcr_impl",
    "calc_wskew_impl",
    "calc_entropy_impl",
    "calc_corr_impl",
    "calc_extreme_res_RM_MAX_5",
    "calc_extreme_res_STD_MAX_5",
    "calc_amt_res_MAX_5",
    "calc_amt_res_MIN_5",
    "calc_pre_amt_MAX_30",
    "calc_pre_amt_MIN_30",
    "dazzling_vol",
    "dazzling_ret",
}


def _minute_op_param_order() -> dict[str, tuple]:
    return {str(name): tuple(params or ()) for name, params in MinuteFactorEngine._OPS_PARAM_ORDER.items()}


def _minute_op_arity(name: str) -> int | None:
    func = MinuteFactorEngine._OPS.get(name)
    if func is None:
        return None
    target = getattr(func, "py_func", func)
    positional_count = target.__code__.co_argcount if hasattr(target, "__code__") else None
    if positional_count is None:
        return None
    return max(0, positional_count - len(_minute_op_param_order().get(name, ())))


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def _extract_identifier_tokens(text: str) -> set[str]:
    if not text:
        return set()
    return {match.group(0) for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]*\b", str(text))}


def _require_positive_int(node: Node) -> int | None:
    if not isinstance(node, NumberNode):
        return None
    value = float(node.value)
    if not value.is_integer() or int(value) <= 0:
        return None
    return int(value)


def _render_node(node: Node) -> str:
    if isinstance(node, VarNode):
        return node.name
    if isinstance(node, NumberNode):
        return _format_number(node.value)
    if isinstance(node, UnaryOpNode):
        return f"({node.op}{_render_node(node.operand)})"
    if isinstance(node, BinaryOpNode):
        return f"({_render_node(node.left)} {node.op} {_render_node(node.right)})"
    if isinstance(node, ConditionalNode):
        return f"WHERE({_render_node(node.condition)}, {_render_node(node.true_expr)}, {_render_node(node.false_expr)})"
    if isinstance(node, FunctionNode):
        args_text = ", ".join(_render_node(arg) for arg in node.args)
        return f"{node.name}({args_text})"
    raise TypeError(f"unsupported AST node: {type(node).__name__}")


def _collect_field_usage(node: Node, minute_fields: set[str], daily_fields: set[str]) -> tuple[set[str], set[str]]:
    minute_used: set[str] = set()
    daily_used: set[str] = set()

    def visit(current: Node) -> None:
        if isinstance(current, VarNode):
            if current.name in minute_fields:
                minute_used.add(current.name)
            if current.name in daily_fields:
                daily_used.add(current.name)
            return
        if isinstance(current, FunctionNode):
            for arg in current.args:
                visit(arg)
            return
        if isinstance(current, BinaryOpNode):
            visit(current.left)
            visit(current.right)
            return
        if isinstance(current, UnaryOpNode):
            visit(current.operand)
            return
        if isinstance(current, ConditionalNode):
            visit(current.condition)
            visit(current.true_expr)
            visit(current.false_expr)

    visit(node)
    return minute_used, daily_used


def _resolve_compiler_field_sets(domains: tuple[str, ...]) -> tuple[set[str], set[str], set[str]]:
    minute_fields = set(getattr(factor_data_domains, "MINUTE_FIELDS", ()))
    if "pv" not in domains:
        return minute_fields, set(), set()
    daily_fields = set(getattr(factor_data_domains, "PV_FIELDS", ()))
    shared_fields = minute_fields & daily_fields
    return minute_fields, daily_fields, shared_fields


@dataclass
class _CompileState:
    minute_fields: set[str]
    daily_fields: set[str]
    shared_fields: set[str] = field(default_factory=set)
    features: list[dict] = field(default_factory=list)
    feature_by_signature: dict[tuple, str] = field(default_factory=dict)

    def add_feature(self, feature: dict, signature: tuple) -> str:
        existing = self.feature_by_signature.get(signature)
        if existing is not None:
            return existing
        name = str(feature["name"])
        self.feature_by_signature[signature] = name
        self.features.append(feature)
        return name


def _make_feature_name(parts: Iterable[object]) -> str:
    cleaned: list[str] = []
    for part in parts:
        text = str(part).strip().lower()
        if not text:
            continue
        normalized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
        normalized = normalized.strip("_")
        if normalized:
            cleaned.append(normalized)
    base = "_".join(cleaned) or "minute_feature"
    if base[0].isdigit():
        base = f"f_{base}"
    return base


def _compile_feature(node: Node, state: _CompileState) -> str | None:
    if (
        isinstance(node, VarNode)
        and node.name in state.minute_fields
        and (node.name not in state.shared_fields or not state.daily_fields)
    ):
        feature = {
            "name": _make_feature_name((node.name, "latest")),
            "kind": "snapshot",
            "field": node.name,
            "offset": 0,
        }
        return state.add_feature(feature, ("snapshot", node.name, 0))

    if not isinstance(node, FunctionNode):
        return None

    name = str(node.name)
    args = list(node.args)

    def resolve_engine_input(arg: Node) -> tuple[str, str | None, tuple] | None:
        if isinstance(arg, VarNode) and arg.name in state.minute_fields:
            return arg.name, None, ("raw", arg.name)
        if isinstance(arg, FunctionNode):
            preprocess_name = str(arg.name)
            preprocess_arity = _minute_op_arity(preprocess_name)
            if preprocess_arity == 1 and len(arg.args) == 1:
                inner_arg = arg.args[0]
                if isinstance(inner_arg, VarNode) and inner_arg.name in state.minute_fields:
                    return inner_arg.name, preprocess_name, ("preprocess", preprocess_name, inner_arg.name)
        return None

    if name in {"ts_return", "pct"} and len(args) == 2 and isinstance(args[0], VarNode) and args[0].name in state.minute_fields:
        lag = _require_positive_int(args[1])
        if lag is None:
            return None
        feature = {
            "name": _make_feature_name((args[0].name, "return", lag)),
            "kind": "snapshot_return",
            "field": args[0].name,
            "lag": lag,
        }
        return state.add_feature(feature, ("snapshot_return", args[0].name, lag))

    aggregation = _WINDOW_AGGREGATION_MAP.get(name)
    if aggregation and len(args) == 2:
        field_arg = args[0]
        window = _require_positive_int(args[1])
        if window is None:
            return None
        if isinstance(field_arg, VarNode) and field_arg.name in state.minute_fields:
            feature = {
                "name": _make_feature_name((field_arg.name, aggregation, window)),
                "kind": "window_aggregate",
                "field": field_arg.name,
                "window": window,
                "aggregation": aggregation,
            }
            return state.add_feature(feature, ("window_aggregate", field_arg.name, window, aggregation))
        if (
            isinstance(field_arg, FunctionNode)
            and str(field_arg.name) == "pct"
            and len(field_arg.args) == 2
            and isinstance(field_arg.args[0], VarNode)
            and field_arg.args[0].name in state.minute_fields
        ):
            lag = _require_positive_int(field_arg.args[1])
            if lag is None:
                return None
            feature = {
                "name": _make_feature_name((field_arg.args[0].name, "pct", aggregation, lag, window)),
                "kind": "intraday_pct_aggregate",
                "field": field_arg.args[0].name,
                "lag": lag,
                "aggregation": aggregation,
            }
            return state.add_feature(
                feature,
                ("intraday_pct_aggregate", field_arg.args[0].name, lag, window, aggregation),
            )
        if isinstance(field_arg, FunctionNode):
            inner_name = str(field_arg.name)
            inner_args = list(field_arg.args)
            registered_inner_arity = _minute_op_arity(inner_name)
            if registered_inner_arity is not None and len(inner_args) >= registered_inner_arity:
                input_args = inner_args[:registered_inner_arity]
                param_args = inner_args[registered_inner_arity:]
                resolved_inputs = [resolve_engine_input(arg) for arg in input_args]
                if all(item is not None for item in resolved_inputs):
                    param_order = _minute_op_param_order().get(inner_name, ())
                    if len(param_args) == len(param_order):
                        operator_params: dict[str, int | float] = {}
                        param_values: list[int | float] = []
                        for param_name, param_node in zip(param_order, param_args):
                            if not isinstance(param_node, NumberNode):
                                return None
                            value = int(param_node.value) if float(param_node.value).is_integer() else float(param_node.value)
                            operator_params[str(param_name)] = value
                            param_values.append(value)
                        input_fields = [item[0] for item in resolved_inputs if item is not None]
                        preprocess = [item[1] for item in resolved_inputs if item is not None]
                        feature = {
                            "name": _make_feature_name(
                                [aggregation, inner_name, *input_fields, *(item for item in preprocess if item), *param_values, window]
                            ),
                            "kind": "window_engine",
                            "inputs": input_fields,
                            "operator": inner_name,
                            "operator_params": operator_params,
                            "window": window,
                            "aggregation": aggregation,
                        }
                        if any(item is not None for item in preprocess):
                            feature["preprocess"] = preprocess
                        return state.add_feature(
                            feature,
                            (
                                "window_engine",
                                aggregation,
                                inner_name,
                                tuple(item[2] for item in resolved_inputs if item is not None),
                                tuple(param_values),
                                window,
                            ),
                        )

    registered_op_arity = _minute_op_arity(name)
    if registered_op_arity is not None and name in _VECTOR_OUTPUT_MINUTE_OPS and len(args) >= registered_op_arity:
        input_args = args[:registered_op_arity]
        param_args = args[registered_op_arity:]
        resolved_inputs = [resolve_engine_input(arg) for arg in input_args]
        if all(item is not None for item in resolved_inputs):
            param_order = _minute_op_param_order().get(name, ())
            if len(param_args) == len(param_order):
                operator_params: dict[str, int | float] = {}
                param_values: list[int | float] = []
                for param_name, param_node in zip(param_order, param_args):
                    if not isinstance(param_node, NumberNode):
                        return None
                    value = int(param_node.value) if float(param_node.value).is_integer() else float(param_node.value)
                    operator_params[str(param_name)] = value
                    param_values.append(value)
                input_fields = [item[0] for item in resolved_inputs if item is not None]
                preprocess = [item[1] for item in resolved_inputs if item is not None]
                feature = {
                    "name": _make_feature_name([name, *input_fields, *(item for item in preprocess if item), *param_values]),
                    "kind": "engine_run",
                    "inputs": input_fields,
                    "operator": name,
                    "operator_params": operator_params,
                    "kwargs": {},
                }
                if any(item is not None for item in preprocess):
                    feature["preprocess"] = preprocess
                return state.add_feature(
                    feature,
                    (
                        "engine_run",
                        name,
                        tuple(item[2] for item in resolved_inputs if item is not None),
                        tuple(param_values),
                    ),
                )

    return None


def _transform_node(node: Node, state: _CompileState) -> str | None:
    compiled_feature = _compile_feature(node, state)
    if compiled_feature is not None:
        return compiled_feature

    if isinstance(node, VarNode):
        return node.name
    if isinstance(node, NumberNode):
        return _format_number(node.value)
    if isinstance(node, UnaryOpNode):
        operand = _transform_node(node.operand, state)
        return None if operand is None else f"({node.op}{operand})"
    if isinstance(node, BinaryOpNode):
        left = _transform_node(node.left, state)
        right = _transform_node(node.right, state)
        if left is None or right is None:
            return None
        return f"({left} {node.op} {right})"
    if isinstance(node, ConditionalNode):
        condition = _transform_node(node.condition, state)
        true_expr = _transform_node(node.true_expr, state)
        false_expr = _transform_node(node.false_expr, state)
        if condition is None or true_expr is None or false_expr is None:
            return None
        return f"WHERE({condition}, {true_expr}, {false_expr})"
    if isinstance(node, FunctionNode):
        args = []
        for arg in node.args:
            transformed = _transform_node(arg, state)
            if transformed is None:
                return None
            args.append(transformed)
        return f"{node.name}({', '.join(args)})"
    return None


def _collect_var_names(node: Node) -> set[str]:
    names: set[str] = set()

    def visit(current: Node) -> None:
        if isinstance(current, VarNode):
            names.add(current.name)
            return
        if isinstance(current, FunctionNode):
            for arg in current.args:
                visit(arg)
            return
        if isinstance(current, BinaryOpNode):
            visit(current.left)
            visit(current.right)
            return
        if isinstance(current, UnaryOpNode):
            visit(current.operand)
            return
        if isinstance(current, ConditionalNode):
            visit(current.condition)
            visit(current.true_expr)
            visit(current.false_expr)

    visit(node)
    return names


def compile_expression_to_minute_spec(
    target_task: FactorTask,
    active_domains: Iterable[str] | None = None,
) -> dict | None:
    expression = str(getattr(target_task, "factor_expression", "") or "").strip()
    if not expression:
        return None

    domains = factor_data_domains.parse_factor_domains(active_domains or factor_data_domains.resolve_factor_domains())
    if "minutes" not in domains:
        return None

    try:
        root = parse_expression(expression)
    except Exception:
        return None

    minute_fields_all, daily_fields, shared_fields = _resolve_compiler_field_sets(domains)
    # Shared fields are routed by expression context: inside a compiled minute subtree they
    # stay on the minute side, while raw references that survive into final_expression are
    # treated as daily pv fields only when they are valid daily fields. Raw minute OHLC names
    # such as `opens/highs/lows/closes` are therefore allowed inside minute subtrees, but they
    # still cannot survive as raw daily anchors in final_expression.

    state = _CompileState(
        minute_fields=set(minute_fields_all),
        daily_fields=set(daily_fields),
        shared_fields=set(shared_fields),
    )
    final_expression = _transform_node(root, state)
    if final_expression is None or not state.features:
        return None

    final_expression_fields = _extract_identifier_tokens(final_expression) & state.daily_fields
    data_needed = sorted(final_expression_fields) if "pv" in domains else []

    if "pv" in domains:
        # Joint contract requires an explicit daily pv anchor and at least one compiled
        # minute feature. Shared non-OHLC fields such as `volumes` are already routed to
        # the minute side above, so we do not re-reject them here.
        if not data_needed:
            return None
        minute_used = _collect_minute_feature_input_fields(state.features)
        if not minute_used:
            return None

    return {
        "data_needed": data_needed,
        "minute_features": state.features,
        "final_expression": final_expression,
    }


def _collect_minute_feature_input_fields(features: list[dict]) -> set[str]:
    used: set[str] = set()
    for feature in features:
        field_name = feature.get("field")
        if isinstance(field_name, str):
            used.add(field_name)
        inputs = feature.get("inputs")
        if isinstance(inputs, list):
            for item in inputs:
                if isinstance(item, str):
                    used.add(item)
    return used
