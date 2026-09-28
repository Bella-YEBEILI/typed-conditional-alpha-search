from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Literal
from typing import Iterable

import quantaalpha.backtest.minute_ops  # noqa: F401
from quantaalpha.factors.coder.factor import FactorTask
from quantaalpha.factors.coder.minute_op_registry import (
    get_minute_op_spec,
)
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
from quantaalpha.factors import data_domains as factor_data_domains


_WINDOW_AGGREGATION_MAP = {
    "mean": "mean",
    "sum": "sum",
    "min": "min",
    "max": "max",
    "std": "std",
    "kurt": "kurt",
    "ts_mean": "mean",
    "ts_sum": "sum",
    "ts_std": "std",
    "ts_min": "min",
    "ts_max": "max",
    "ts_kurt": "kurt",
}
_POINTWISE_BINARY_OP_SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def _render_unary_pointwise_on_vector(name: str, operand: str) -> str | None:
    # Numba-compiled unary minute ops assume a 2D (minutes, stocks) buffer. When
    # the caller hands them a reduced 1D (stocks,) vector — e.g. the result of
    # `ts_regression(...)` — they trap at `nd, ns = arr.shape`. Fall back to
    # numpy equivalents that preserve NaN semantics from the original kernels.
    wrapped = f"np.asarray({operand}, dtype=np.float32)"
    if name == "neg":
        return f"(-{wrapped})"
    if name == "pos":
        return f"np.where(np.isfinite({wrapped}), np.maximum({wrapped}, np.float32(0.0)), np.float32('nan')).astype(np.float32, copy=False)"
    if name == "neg_part":
        return f"np.where(np.isfinite({wrapped}), np.minimum({wrapped}, np.float32(0.0)), np.float32('nan')).astype(np.float32, copy=False)"
    if name == "abs":
        return f"np.abs({wrapped})"
    if name == "nozero_abs":
        return f"np.where(np.abs({wrapped}) < np.float32(1e-12), np.float32('nan'), np.abs({wrapped})).astype(np.float32, copy=False)"
    if name == "identity":
        return wrapped
    if name == "log":
        return f"np.where({wrapped} > np.float32(0), np.log(np.where({wrapped} > np.float32(0), {wrapped}, np.float32(1.0))), np.float32('nan')).astype(np.float32, copy=False)"
    return None


_WHERE_FUNCTION_NAMES = ("WHERE", "where", "IF_ELSE", "if_else", "ifelse")
_MinuteExprShape = Literal["scalar", "tensor", "vector"]
MAX_INTRADAY_LAG = 1440

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


def _require_intraday_lag(node: Node) -> int | None:
    lag = _require_positive_int(node)
    if lag is None or lag > MAX_INTRADAY_LAG:
        return None
    return lag


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
    if isinstance(node, ListNode):
        return "[" + ", ".join(_render_node(item) for item in node.items) + "]"
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
        if isinstance(current, ListNode):
            for item in current.items:
                visit(item)
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
    if "minutes" in domains:
        daily_fields.update(getattr(factor_data_domains, "JOINT_PV_DAILY_SHARED_FIELD_ALIASES", {}).keys())
    shared_fields = minute_fields & daily_fields
    return minute_fields, daily_fields, shared_fields


@dataclass
class _CompileState:
    minute_fields: set[str]
    daily_fields: set[str]
    shared_fields: set[str] = field(default_factory=set)
    features: list[dict] = field(default_factory=list)
    feature_by_signature: dict[tuple, str] = field(default_factory=dict)
    used_feature_names: set[str] = field(default_factory=set)

    def add_feature(self, feature: dict, signature: tuple) -> str:
        existing = self.feature_by_signature.get(signature)
        if existing is not None:
            return existing
        name = self._unique_feature_name(str(feature["name"]), signature)
        if name != feature.get("name"):
            feature = dict(feature)
            feature["name"] = name
        self.feature_by_signature[signature] = name
        self.used_feature_names.add(name)
        self.features.append(feature)
        return name

    def _unique_feature_name(self, base_name: str, signature: tuple) -> str:
        if base_name not in self.used_feature_names:
            return base_name
        digest = hashlib.sha1(repr(signature).encode("utf-8")).hexdigest()[:8]
        candidate = f"{base_name}_{digest}"
        index = 2
        while candidate in self.used_feature_names:
            candidate = f"{base_name}_{digest}_{index}"
            index += 1
        return candidate


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


def _shape_compatible(left: _MinuteExprShape, right: _MinuteExprShape) -> _MinuteExprShape | None:
    if left == right:
        return left
    if left == "scalar":
        return right
    if right == "scalar":
        return left
    # A (stocks,) minute-reduced vector broadcasts against a (minutes, stocks) tensor
    # under numpy semantics and yields a tensor. This lets per-day normalizations like
    # `div(ts_mean(volumes, 20), volumes)` flow through pointwise minute ops.
    if {left, right} == {"vector", "tensor"}:
        return "tensor"
    return None


def _infer_minute_subtree_shape(
    node: Node,
    state: _CompileState,
    *,
    allow_shared_fields: bool = False,
) -> _MinuteExprShape | None:
    if isinstance(node, NumberNode):
        return "scalar"
    if isinstance(node, VarNode):
        if node.name not in state.minute_fields:
            return None
        if node.name in state.shared_fields and state.daily_fields and not allow_shared_fields:
            return None
        return "tensor"
    if isinstance(node, UnaryOpNode):
        return _infer_minute_subtree_shape(node.operand, state, allow_shared_fields=allow_shared_fields)
    if isinstance(node, BinaryOpNode):
        left = _infer_minute_subtree_shape(node.left, state, allow_shared_fields=allow_shared_fields)
        right = _infer_minute_subtree_shape(node.right, state, allow_shared_fields=allow_shared_fields)
        if left is None or right is None:
            return None
        return _shape_compatible(left, right)
    if isinstance(node, ConditionalNode):
        condition = _infer_minute_subtree_shape(node.condition, state, allow_shared_fields=allow_shared_fields)
        true_expr = _infer_minute_subtree_shape(node.true_expr, state, allow_shared_fields=allow_shared_fields)
        false_expr = _infer_minute_subtree_shape(node.false_expr, state, allow_shared_fields=allow_shared_fields)
        if condition is None or true_expr is None or false_expr is None:
            return None
        result_shape = _shape_compatible(true_expr, false_expr)
        if result_shape is None:
            return None
        condition_shape = _shape_compatible(condition, result_shape)
        if condition_shape is None:
            return None
        return result_shape
    if isinstance(node, ListNode):
        return None
    if not isinstance(node, FunctionNode):
        return None

    name = str(node.name)
    args = list(node.args)
    if name in _WHERE_FUNCTION_NAMES and len(args) in {2, 3}:
        cond_shape = _infer_minute_subtree_shape(args[0], state, allow_shared_fields=allow_shared_fields)
        true_shape = _infer_minute_subtree_shape(args[1], state, allow_shared_fields=allow_shared_fields)
        false_shape = (
            _infer_minute_subtree_shape(args[2], state, allow_shared_fields=allow_shared_fields)
            if len(args) == 3
            else "scalar"
        )
        if cond_shape is None or true_shape is None or false_shape is None:
            return None
        result_shape = _shape_compatible(true_shape, false_shape)
        if result_shape is None:
            return None
        condition_shape = _shape_compatible(cond_shape, result_shape)
        if condition_shape is None:
            return None
        return condition_shape
    if name == "safe_div":
        if len(args) not in {2, 3}:
            return None
        left = _infer_minute_subtree_shape(args[0], state, allow_shared_fields=allow_shared_fields)
        right = _infer_minute_subtree_shape(args[1], state, allow_shared_fields=allow_shared_fields)
        if left is None or right is None:
            return None
        if len(args) == 3 and not isinstance(args[2], NumberNode):
            return None
        return _shape_compatible(left, right)
    if name == "ts_return":
        if len(args) != 2 or _require_intraday_lag(args[1]) is None:
            return None
        field_shape = _infer_minute_subtree_shape(args[0], state, allow_shared_fields=True)
        return "vector" if field_shape == "tensor" else None
    if name == "pct" and len(args) == 2 and _require_intraday_lag(args[1]) is not None:
        field_shape = _infer_minute_subtree_shape(args[0], state, allow_shared_fields=True)
        if field_shape == "tensor":
            return "vector"

    aggregation = _WINDOW_AGGREGATION_MAP.get(name)
    if aggregation and len(args) == 2:
        field_shape = _infer_minute_subtree_shape(args[0], state, allow_shared_fields=True)
        return "vector" if field_shape == "tensor" and isinstance(args[1], NumberNode) else None

    spec = get_minute_op_spec(name)
    if spec is None or len(args) != spec.total_args:
        return None
    input_shapes = [
        _infer_minute_subtree_shape(arg, state, allow_shared_fields=True)
        for arg in args[: spec.arity]
    ]
    if any(shape is None for shape in input_shapes):
        return None
    for param_node in args[spec.arity :]:
        if not isinstance(param_node, NumberNode):
            return None
    # Pointwise ops are rendered with numpy operators and tolerate broadcasting
    # (e.g. a reduced 1D vector against a 2D tensor). The output follows the
    # broadest input shape rather than the declared output_kind so downstream
    # shape checks stay accurate.
    if spec.category == "pointwise":
        combined: _MinuteExprShape = "scalar"
        for shape in input_shapes:
            merged = _shape_compatible(combined, shape)
            if merged is None:
                return None
            combined = merged
        return combined
    if tuple(input_shapes) != spec.input_kinds:
        return None
    return spec.output_kind


def _parse_number_params(param_order: tuple[str, ...], param_args: list[Node]) -> tuple[dict[str, int | float], list[int | float]] | None:
    if len(param_args) != len(param_order):
        return None
    operator_params: dict[str, int | float] = {}
    param_values: list[int | float] = []
    for param_name, param_node in zip(param_order, param_args):
        if not isinstance(param_node, NumberNode):
            return None
        value = int(param_node.value) if float(param_node.value).is_integer() else float(param_node.value)
        operator_params[str(param_name)] = value
        param_values.append(value)
    return operator_params, param_values


def _compile_latest_snapshot_expr(node: Node, state: _CompileState) -> str | None:
    if _infer_minute_subtree_shape(node, state, allow_shared_fields=True) != "tensor":
        return None
    if isinstance(node, NumberNode):
        return _format_number(node.value)
    if isinstance(node, VarNode):
        if node.name not in state.minute_fields:
            return None
        feature = {
            "name": _make_feature_name((node.name, "latest")),
            "kind": "snapshot",
            "field": node.name,
            "offset": 0,
        }
        return state.add_feature(feature, ("snapshot", node.name, 0))
    if isinstance(node, UnaryOpNode):
        operand = _compile_latest_snapshot_expr(node.operand, state)
        return None if operand is None else f"({node.op}{operand})"
    if isinstance(node, BinaryOpNode):
        left = _compile_latest_snapshot_expr(node.left, state)
        right = _compile_latest_snapshot_expr(node.right, state)
        if left is None or right is None:
            return None
        return f"({left} {node.op} {right})"
    if isinstance(node, ConditionalNode):
        condition = _compile_latest_snapshot_expr(node.condition, state)
        true_expr = _compile_latest_snapshot_expr(node.true_expr, state)
        false_expr = _compile_latest_snapshot_expr(node.false_expr, state)
        if condition is None or true_expr is None or false_expr is None:
            return None
        return f"WHERE({condition}, {true_expr}, {false_expr})"
    if not isinstance(node, FunctionNode):
        return None

    name = str(node.name)
    args = list(node.args)
    if name in _WHERE_FUNCTION_NAMES and len(args) in {2, 3}:
        condition = _compile_latest_snapshot_expr(args[0], state)
        true_expr = _compile_latest_snapshot_expr(args[1], state)
        false_expr = _compile_latest_snapshot_expr(args[2], state) if len(args) == 3 else "0"
        if condition is None or true_expr is None or false_expr is None:
            return None
        return f"WHERE({condition}, {true_expr}, {false_expr})"
    if name == "safe_div":
        if len(args) not in {2, 3}:
            return None
        left = _compile_latest_snapshot_expr(args[0], state)
        right = _compile_latest_snapshot_expr(args[1], state)
        if left is None or right is None:
            return None
        eps = _format_number(args[2].value) if len(args) == 3 and isinstance(args[2], NumberNode) else "5e-2"
        if len(args) == 3 and not isinstance(args[2], NumberNode):
            return None
        return f"safe_div({left}, {right}, eps={eps})"
    if name in _POINTWISE_BINARY_OP_SYMBOL and len(args) == 2:
        left = _compile_latest_snapshot_expr(args[0], state)
        right = _compile_latest_snapshot_expr(args[1], state)
        if left is None or right is None:
            return None
        return f"({left} {_POINTWISE_BINARY_OP_SYMBOL[name]} {right})"
    if name in {"identity", "neg", "abs", "pos", "neg_part", "log"} and len(args) == 1:
        operand = _compile_latest_snapshot_expr(args[0], state)
        if operand is None:
            return None
        if name == "identity":
            return operand
        if name == "neg":
            return f"(-{operand})"
        return f"{name}({operand})"
    return None


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

    if name == "last" and len(args) == 1:
        rendered_latest = _compile_latest_snapshot_expr(args[0], state)
        if rendered_latest is not None:
            return rendered_latest

    def resolve_engine_input(arg: Node) -> tuple[str, str | None, tuple] | None:
        if isinstance(arg, VarNode) and arg.name in state.minute_fields:
            return arg.name, None, ("raw", arg.name)
        if isinstance(arg, FunctionNode):
            preprocess_spec = get_minute_op_spec(str(arg.name))
            if (
                preprocess_spec is not None
                and preprocess_spec.output_kind == "tensor"
                and preprocess_spec.input_kinds == ("tensor",)
                and len(arg.args) == preprocess_spec.total_args
                and not preprocess_spec.param_order
            ):
                inner_arg = arg.args[0]
                if isinstance(inner_arg, VarNode) and inner_arg.name in state.minute_fields:
                    preprocess_name = str(arg.name)
                    return inner_arg.name, preprocess_name, ("preprocess", preprocess_name, inner_arg.name)
        return None

    if name in {"ts_return", "pct"} and len(args) == 2 and isinstance(args[0], VarNode) and args[0].name in state.minute_fields:
        lag = _require_intraday_lag(args[1])
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
            lag = _require_intraday_lag(field_arg.args[1])
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
            inner_spec = get_minute_op_spec(str(field_arg.name))
            if inner_spec is not None and inner_spec.output_kind == "tensor" and len(field_arg.args) == inner_spec.total_args:
                inner_name = str(field_arg.name)
                inner_args = list(field_arg.args)
                input_args = inner_args[: inner_spec.arity]
                param_args = inner_args[inner_spec.arity :]
                resolved_inputs = [resolve_engine_input(arg) for arg in input_args]
                if all(item is not None for item in resolved_inputs):
                    parsed_params = _parse_number_params(inner_spec.param_order, param_args)
                    if parsed_params is not None:
                        operator_params, param_values = parsed_params
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
        inner_shape = _infer_minute_subtree_shape(field_arg, state, allow_shared_fields=True)
        if inner_shape == "tensor":
            runtime_expr = _render_minute_runtime_expr(field_arg, state, allow_shared_fields=True)
            if runtime_expr is not None:
                used_fields = sorted(_collect_var_names(field_arg) & state.minute_fields)
                if used_fields:
                    expression_code = (
                        f'_qa_minute_op("{aggregation}", '
                        f"np.ascontiguousarray(np.asarray({runtime_expr}, dtype=np.float32)[-{window}:, :]))"
                    )
                    feature = {
                        "name": _make_feature_name((aggregation, _render_node(field_arg), *used_fields, window)),
                        "kind": "vector_expr",
                        "inputs": used_fields,
                        "expression_code": expression_code,
                    }
                    return state.add_feature(
                        feature,
                        ("window_tensor_expr", aggregation, runtime_expr, window),
                    )

    spec = get_minute_op_spec(name)
    if spec is not None and spec.output_kind == "vector" and len(args) == spec.total_args:
        input_args = args[: spec.arity]
        param_args = args[spec.arity :]
        resolved_inputs = [resolve_engine_input(arg) for arg in input_args]
        if all(item is not None for item in resolved_inputs):
            parsed_params = _parse_number_params(spec.param_order, param_args)
            if parsed_params is not None:
                operator_params, param_values = parsed_params
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


def _render_minute_runtime_expr(
    node: Node,
    state: _CompileState,
    *,
    allow_shared_fields: bool = True,
) -> str | None:
    if isinstance(node, VarNode):
        shape = _infer_minute_subtree_shape(node, state, allow_shared_fields=allow_shared_fields)
        return node.name if shape == "tensor" else None
    if isinstance(node, NumberNode):
        return _format_number(node.value)
    if isinstance(node, UnaryOpNode):
        operand = _render_minute_runtime_expr(node.operand, state, allow_shared_fields=allow_shared_fields)
        if operand is None:
            return None
        return f"({node.op}{operand})"
    if isinstance(node, BinaryOpNode):
        shape = _infer_minute_subtree_shape(node, state, allow_shared_fields=allow_shared_fields)
        if shape is None:
            return None
        left = _render_minute_runtime_expr(node.left, state, allow_shared_fields=allow_shared_fields)
        right = _render_minute_runtime_expr(node.right, state, allow_shared_fields=allow_shared_fields)
        if left is None or right is None:
            return None
        if node.op == "/":
            return f"_qa_minute_div({left}, {right})"
        return f"({left} {node.op} {right})"
    if isinstance(node, ConditionalNode):
        shape = _infer_minute_subtree_shape(node, state, allow_shared_fields=allow_shared_fields)
        if shape is None:
            return None
        condition = _render_minute_runtime_expr(node.condition, state, allow_shared_fields=allow_shared_fields)
        true_expr = _render_minute_runtime_expr(node.true_expr, state, allow_shared_fields=allow_shared_fields)
        false_expr = _render_minute_runtime_expr(node.false_expr, state, allow_shared_fields=allow_shared_fields)
        if condition is None or true_expr is None or false_expr is None:
            return None
        return f"np.where({condition}, {true_expr}, {false_expr})"
    if isinstance(node, ListNode):
        return None
    if not isinstance(node, FunctionNode):
        return None

    name = str(node.name)
    args = list(node.args)
    if name in _WHERE_FUNCTION_NAMES and len(args) in {2, 3}:
        if _infer_minute_subtree_shape(node, state, allow_shared_fields=allow_shared_fields) is None:
            return None
        cond = _render_minute_runtime_expr(args[0], state, allow_shared_fields=allow_shared_fields)
        true_expr = _render_minute_runtime_expr(args[1], state, allow_shared_fields=allow_shared_fields)
        false_expr = _render_minute_runtime_expr(args[2], state, allow_shared_fields=allow_shared_fields) if len(args) == 3 else "0"
        if cond is None or true_expr is None or false_expr is None:
            return None
        return f"np.where({cond}, {true_expr}, {false_expr})"
    if name == "safe_div":
        if _infer_minute_subtree_shape(node, state, allow_shared_fields=allow_shared_fields) is None:
            return None
        if len(args) not in {2, 3}:
            return None
        left = _render_minute_runtime_expr(args[0], state, allow_shared_fields=allow_shared_fields)
        right = _render_minute_runtime_expr(args[1], state, allow_shared_fields=allow_shared_fields)
        if left is None or right is None:
            return None
        eps = _format_number(args[2].value) if len(args) == 3 and isinstance(args[2], NumberNode) else "5e-2"
        if len(args) == 3 and not isinstance(args[2], NumberNode):
            return None
        return f"_qa_minute_safe_div({left}, {right}, eps={eps})"

    if name in _POINTWISE_BINARY_OP_SYMBOL and len(args) == 2:
        if _infer_minute_subtree_shape(node, state, allow_shared_fields=allow_shared_fields) is None:
            return None
        left = _render_minute_runtime_expr(args[0], state, allow_shared_fields=allow_shared_fields)
        right = _render_minute_runtime_expr(args[1], state, allow_shared_fields=allow_shared_fields)
        if left is None or right is None:
            return None
        if name == "div":
            return f"_qa_minute_div({left}, {right})"
        op_symbol = _POINTWISE_BINARY_OP_SYMBOL[name]
        return f"({left} {op_symbol} {right})"

    agg_alias = _WINDOW_AGGREGATION_MAP.get(name)
    if agg_alias and len(args) == 2 and isinstance(args[1], NumberNode):
        window = _require_positive_int(args[1])
        if window is None:
            return None
        inner_shape = _infer_minute_subtree_shape(args[0], state, allow_shared_fields=allow_shared_fields)
        if inner_shape != "tensor":
            return None
        inner = _render_minute_runtime_expr(args[0], state, allow_shared_fields=allow_shared_fields)
        if inner is None:
            return None
        return (
            f'_qa_minute_op("{agg_alias}", '
            f'np.ascontiguousarray(np.asarray({inner}, dtype=np.float32)[-{window}:, :]))'
        )

    spec = get_minute_op_spec(name)
    if spec is None:
        return None
    if _infer_minute_subtree_shape(node, state, allow_shared_fields=allow_shared_fields) is None:
        return None
    if len(args) != spec.total_args:
        return None
    if (
        spec.category == "pointwise"
        and spec.arity == 1
        and not spec.param_order
    ):
        operand_shape = _infer_minute_subtree_shape(args[0], state, allow_shared_fields=allow_shared_fields)
        if operand_shape in ("vector", "scalar"):
            operand_rendered = _render_minute_runtime_expr(
                args[0], state, allow_shared_fields=allow_shared_fields
            )
            if operand_rendered is None:
                return None
            numpy_rendered = _render_unary_pointwise_on_vector(name, operand_rendered)
            if numpy_rendered is not None:
                return numpy_rendered
    rendered_inputs: list[str] = []
    for expected_kind, arg in zip(spec.input_kinds, args[: spec.arity]):
        rendered = _render_minute_runtime_expr(arg, state, allow_shared_fields=allow_shared_fields)
        if rendered is None:
            return None
        actual_kind = _infer_minute_subtree_shape(arg, state, allow_shared_fields=allow_shared_fields)
        if actual_kind != expected_kind:
            if spec.category == "pointwise":
                # Pointwise ops are fine with scalar/vector/tensor inputs because
                # they are lowered via numpy operators above. Anything reaching
                # here is a non-pointwise path, so keep the strict check for
                # reducers (ts_regression, ts_beta, ts_corr, ts_rank, ts_zscore,
                # mean/std/... via their arity-1 form, etc).
                pass
            else:
                return None
        # Non-pointwise reducers expect a contiguous 2D float32 buffer; the
        # generic expression path may produce numpy broadcasts whose memory
        # layout breaks numba's 2D kernels, so materialise here.
        if spec.category != "pointwise" and expected_kind == "tensor":
            rendered = f"np.ascontiguousarray(np.asarray({rendered}, dtype=np.float32))"
        rendered_inputs.append(rendered)
    rendered_params: list[str] = []
    for param_name, param_node in zip(spec.param_order, args[spec.arity :]):
        if not isinstance(param_node, NumberNode):
            return None
        rendered_params.append(f"{param_name}={_format_number(param_node.value)}")
    joined = ", ".join(rendered_inputs + rendered_params)
    return f'_qa_minute_op("{name}", {joined})'


def _compile_custom_vector_feature(node: Node, state: _CompileState) -> str | None:
    if _infer_minute_subtree_shape(node, state) != "vector":
        return None
    runtime_expr = _render_minute_runtime_expr(node, state)
    if runtime_expr is None:
        return None
    used_fields = sorted(_collect_var_names(node) & state.minute_fields)
    if not used_fields:
        return None
    param_parts: list[str] = []
    if isinstance(node, FunctionNode):
        spec = get_minute_op_spec(str(node.name))
        if spec is not None:
            param_parts = [
                _format_number(arg.value)
                for arg in node.args[spec.arity :]
                if isinstance(arg, NumberNode)
            ]
    feature = {
        "name": _make_feature_name((_render_node(node), *used_fields, *param_parts)),
        "kind": "vector_expr",
        "inputs": used_fields,
        "expression_code": runtime_expr,
    }
    return state.add_feature(feature, ("vector_expr", runtime_expr))


def _compile_custom_tensor_snapshot_feature(node: Node, state: _CompileState) -> str | None:
    if _infer_minute_subtree_shape(node, state, allow_shared_fields=True) != "tensor":
        return None
    runtime_expr = _render_minute_runtime_expr(node, state, allow_shared_fields=True)
    if runtime_expr is None:
        return None
    used_fields = sorted(_collect_var_names(node) & state.minute_fields)
    if not used_fields:
        return None
    expression_code = (
        f"np.asarray(np.asarray({runtime_expr}, dtype=np.float32)[-1, :], dtype=np.float32)"
    )
    feature = {
        "name": _make_feature_name(("last", _render_node(node), *used_fields)),
        "kind": "vector_expr",
        "inputs": used_fields,
        "expression_code": expression_code,
    }
    return state.add_feature(feature, ("tensor_snapshot_vector_expr", expression_code))


def _should_prefer_custom_vector_feature(node: Node, state: _CompileState) -> bool:
    if not isinstance(node, FunctionNode):
        return False
    spec = get_minute_op_spec(str(node.name))
    if spec is None or spec.output_kind != "vector":
        return False
    args = list(node.args)
    if len(args) != spec.total_args:
        return False
    for arg in args[: spec.arity]:
        if isinstance(arg, VarNode):
            continue
        if (
            isinstance(arg, FunctionNode)
            and get_minute_op_spec(str(arg.name)) is not None
            and get_minute_op_spec(str(arg.name)).output_kind == "tensor"
        ):
            return True
        if _infer_minute_subtree_shape(arg, state, allow_shared_fields=True) == "tensor":
            return True
    return False


def _recursive_transform(node: Node, state: _CompileState) -> str | None:
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
    if isinstance(node, ListNode):
        items = []
        for item in node.items:
            transformed = _transform_node(item, state)
            if transformed is None:
                return None
            items.append(transformed)
        return "[" + ", ".join(items) + "]"
    if isinstance(node, FunctionNode):
        args = []
        for arg in node.args:
            transformed = _transform_node(arg, state)
            if transformed is None:
                return None
            args.append(transformed)
        return f"{node.name}({', '.join(args)})"
    return None


def _transform_node(node: Node, state: _CompileState) -> str | None:
    # Prefer deterministic structured feature kinds (snapshot, window_aggregate,
    # engine_run, window_engine, ...) first — they produce cleaner, reusable,
    # engine-level calls.
    compiled_feature = _compile_feature(node, state)
    if compiled_feature is not None:
        return compiled_feature
    minute_shape = _infer_minute_subtree_shape(node, state)
    # A raw tensor subtree cannot survive into the final (daily) expression on
    # its own — it needs to be consumed by a minute reducer. Try to fold the
    # enclosing tensor-shaped node into a latest-per-day `vector_expr` instead
    # of dropping the trajectory.
    if minute_shape == "tensor":
        return _compile_custom_tensor_snapshot_feature(node, state)
    if minute_shape == "vector" and _should_prefer_custom_vector_feature(node, state):
        custom_vector = _compile_custom_vector_feature(node, state)
        if custom_vector is not None:
            return custom_vector
    # Recurse first so that vector-shaped expressions whose parts decompose
    # cleanly (e.g. `safe_div(ts_mean(a, n), ts_std(b, m))`) stay as two small
    # window_aggregate features combined at the daily layer.
    recursed = _recursive_transform(node, state)
    if recursed is not None:
        return recursed
    # Fallback for vector-shaped expressions that contain composite tensor
    # subtrees feeding a reducer (e.g. `ts_regression(div(x, y), z, 20)`). The
    # whole subtree is lowered to a single per-day `vector_expr`.
    if minute_shape == "vector":
        return _compile_custom_vector_feature(node, state)
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
        if isinstance(current, ListNode):
            for item in current.items:
                visit(item)
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
