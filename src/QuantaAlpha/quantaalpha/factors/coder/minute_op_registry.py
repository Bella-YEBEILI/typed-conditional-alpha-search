from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import inspect

import quantaalpha.backtest.minute_ops  # noqa: F401
from quantaalpha.backtest.minute_tools import MinuteFactorEngine


@dataclass(frozen=True)
class MinuteOpSpec:
    name: str
    arity: int
    input_kinds: tuple[str, ...]
    output_kind: str  # "tensor" or "vector"
    category: str
    param_order: tuple[str, ...] = ()

    @property
    def total_args(self) -> int:
        return self.arity + len(self.param_order)

    @property
    def arity_constraint(self) -> tuple[int, int]:
        return self.total_args, self.total_args

    @property
    def signature_hint(self) -> str:
        input_parts = [f"arg{i + 1}" for i in range(self.arity)]
        return f"{self.name}({', '.join([*input_parts, *self.param_order])})"


_OUTPUT_KIND_BY_NAME: dict[str, tuple[str, str]] = {
    "mean": ("vector", "reduce"),
    "std": ("vector", "reduce"),
    "kurt": ("vector", "reduce"),
    "sum": ("vector", "reduce"),
    "min": ("vector", "reduce"),
    "max": ("vector", "reduce"),
    "last": ("vector", "reduce"),
    "corr": ("vector", "reduce"),
    "ts_rank": ("vector", "reduce"),
    "ts_zscore": ("vector", "reduce"),
    "ts_corr": ("vector", "reduce"),
    "ts_beta": ("vector", "reduce"),
    "ts_regression": ("vector", "reduce"),
    "calc_guiji_impl": ("vector", "special"),
    "calc_wcr_impl": ("vector", "special"),
    "calc_wskew_impl": ("vector", "special"),
    "calc_entropy_impl": ("vector", "special"),
    "calc_corr_impl": ("vector", "special"),
    "calc_extreme_res_RM_MAX_5": ("vector", "special"),
    "calc_extreme_res_STD_MAX_5": ("vector", "special"),
    "calc_amt_res_MAX_5": ("vector", "special"),
    "calc_amt_res_MIN_5": ("vector", "special"),
    "calc_pre_amt_MAX_30": ("vector", "special"),
    "calc_pre_amt_MIN_30": ("vector", "special"),
    "dazzling_vol": ("vector", "special"),
    "dazzling_ret": ("vector", "special"),
    "identity": ("tensor", "pointwise"),
    "add": ("tensor", "pointwise"),
    "sub": ("tensor", "pointwise"),
    "mul": ("tensor", "pointwise"),
    "div": ("tensor", "pointwise"),
    "safe_div": ("tensor", "pointwise"),
    "pct": ("tensor", "pointwise"),
    "log": ("tensor", "pointwise"),
    "abs": ("tensor", "pointwise"),
    "pos": ("tensor", "pointwise"),
    "neg": ("tensor", "pointwise"),
    "neg_part": ("tensor", "pointwise"),
    "rank": ("tensor", "cross_sectional"),
    "neutralize": ("tensor", "cross_sectional"),
    "zscore": ("tensor", "cross_sectional"),
    "scale": ("tensor", "cross_sectional"),
    "nozero_abs": ("tensor", "pointwise"),
}


def _infer_registered_minute_arity(name: str, func) -> int:
    target = getattr(func, "py_func", func)
    positional_count = target.__code__.co_argcount if hasattr(target, "__code__") else None
    param_order = tuple(MinuteFactorEngine._OPS_PARAM_ORDER.get(name) or ())
    if positional_count is None:
        raise ValueError(f"cannot infer arity for minute operator `{name}`")
    return max(0, positional_count - len(param_order))


@lru_cache(maxsize=1)
def get_minute_op_registry() -> dict[str, MinuteOpSpec]:
    registry: dict[str, MinuteOpSpec] = {}
    for name, func in MinuteFactorEngine._OPS.items():
        output_kind, category = _OUTPUT_KIND_BY_NAME.get(name, ("", ""))
        if not output_kind:
            raise ValueError(f"minute operator `{name}` is missing output-kind metadata")
        arity = _infer_registered_minute_arity(name, func)
        param_order = tuple(MinuteFactorEngine._OPS_PARAM_ORDER.get(name) or ())
        registry[str(name)] = MinuteOpSpec(
            name=str(name),
            arity=arity,
            input_kinds=tuple("tensor" for _ in range(arity)),
            output_kind=output_kind,
            category=category,
            param_order=param_order,
        )
    return registry


def get_minute_op_spec(name: str) -> MinuteOpSpec | None:
    return get_minute_op_registry().get(str(name))


def get_tensor_output_minute_ops() -> set[str]:
    return {
        name
        for name, spec in get_minute_op_registry().items()
        if spec.output_kind == "tensor"
    }


def get_vector_output_minute_ops() -> set[str]:
    return {
        name
        for name, spec in get_minute_op_registry().items()
        if spec.output_kind == "vector"
    }


def get_registered_minute_ops_missing_metadata() -> set[str]:
    registered = set(MinuteFactorEngine._OPS.keys())
    covered = set(_OUTPUT_KIND_BY_NAME.keys())
    return registered - covered


def get_minute_operator_arity_constraints() -> dict[str, tuple[int, int]]:
    return {
        name: spec.arity_constraint
        for name, spec in get_minute_op_registry().items()
    }


def get_minute_operator_signature_hints() -> dict[str, str]:
    hints: dict[str, str] = {}
    for name, func in MinuteFactorEngine._OPS.items():
        try:
            hints[name] = f"{name}{inspect.signature(func)}"
        except (TypeError, ValueError):
            hints[name] = get_minute_op_registry()[name].signature_hint
    return hints
