import inspect
from typing import Iterable

import quantaalpha.backtest.minute_ops  # noqa: F401
from quantaalpha.backtest.minute_tools import MinuteFactorEngine
from quantaalpha.factors import accelerated_ops
from quantaalpha.factors.coder.minute_op_registry import (
    get_minute_op_registry,
    get_minute_operator_arity_constraints,
    get_minute_operator_signature_hints,
)
from quantaalpha.factors.data_domains import get_domain_field_names, parse_factor_domains, resolve_factor_domains


def _discover_allowed_operator_names() -> set[str]:
    return {
        name
        for name in dir(accelerated_ops)
        if not name.startswith("_")
        and (inspect.isfunction(getattr(accelerated_ops, name)) or inspect.isbuiltin(getattr(accelerated_ops, name)))
        and not getattr(getattr(accelerated_ops, name), "__module__", "").startswith("numba.")
    }


def _discover_minute_operator_names() -> set[str]:
    return set(get_minute_op_registry().keys())


_TEMPLATE_OPERATOR_ALIASES = frozenset({"ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"})
_TEMPLATE_OPERATOR_ALIAS_ARITIES = {
    "ADD": (2, 2),
    "SUBTRACT": (2, 2),
    "MULTIPLY": (2, 2),
    "DIVIDE": (2, 2),
}
_TEMPLATE_OPERATOR_ALIAS_SIGNATURE_HINTS = {
    "ADD": "ADD(x, y)",
    "SUBTRACT": "SUBTRACT(x, y)",
    "MULTIPLY": "MULTIPLY(x, y)",
    "DIVIDE": "DIVIDE(x, y)",
}

_BASE_ALLOWED_OPERATOR_NAMES = frozenset(_discover_allowed_operator_names()).union(_TEMPLATE_OPERATOR_ALIASES)
_MINUTE_ALLOWED_OPERATOR_NAMES = frozenset(_discover_minute_operator_names())
_BASE_ALLOWED_FIELD_NAMES = frozenset({})
_MINUTE_PLUS_DAILY_ALLOWED_OPERATOR_NAMES = frozenset(
    set(_BASE_ALLOWED_OPERATOR_NAMES).union(_MINUTE_ALLOWED_OPERATOR_NAMES)
)


def _callable_positional_arity(fn) -> tuple[int, int | None] | None:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None

    required = 0
    positional = 0
    has_varargs = False
    for param in sig.parameters.values():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional += 1
            if param.default is inspect._empty:
                required += 1
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            has_varargs = True

    return required, None if has_varargs else positional


def _discover_base_operator_arity_constraints() -> dict[str, tuple[int, int | None]]:
    constraints: dict[str, tuple[int, int | None]] = {}
    for name in _BASE_ALLOWED_OPERATOR_NAMES:
        fn = getattr(accelerated_ops, name, None)
        arity = _callable_positional_arity(fn)
        if arity is not None:
            constraints[name] = arity
    return constraints


def _discover_minute_operator_arity_constraints() -> dict[str, tuple[int, int | None]]:
    return dict(get_minute_operator_arity_constraints())


def _discover_base_operator_signature_hints() -> dict[str, str]:
    hints: dict[str, str] = {}
    for name in _BASE_ALLOWED_OPERATOR_NAMES:
        fn = getattr(accelerated_ops, name, None)
        try:
            hints[name] = f"{name}{inspect.signature(fn)}"
        except (TypeError, ValueError):
            continue
    return hints


def _discover_minute_operator_signature_hints() -> dict[str, str]:
    return dict(get_minute_operator_signature_hints())


_BASE_OPERATOR_ARITY_CONSTRAINTS = _discover_base_operator_arity_constraints()
_BASE_OPERATOR_ARITY_CONSTRAINTS.update(_TEMPLATE_OPERATOR_ALIAS_ARITIES)
_MINUTE_OPERATOR_ARITY_CONSTRAINTS = _discover_minute_operator_arity_constraints()
_BASE_OPERATOR_SIGNATURE_HINTS = _discover_base_operator_signature_hints()
_BASE_OPERATOR_SIGNATURE_HINTS.update(_TEMPLATE_OPERATOR_ALIAS_SIGNATURE_HINTS)
_MINUTE_OPERATOR_SIGNATURE_HINTS = _discover_minute_operator_signature_hints()

# Domain overrides are complete per-domain allowlists.
# Active domains are merged by union so domain-specific operators remain available.
_DOMAIN_OPERATOR_OVERRIDES: dict[str, frozenset[str] | None] = {
    "pv": None,
    "fundamental": None,
    # Minute factors first consume raw minute tensors through MinuteFactorEngine
    # operators, then continue combining the reduced daily 2D frames with the
    # regular accelerated daily operators inside calc_factor.
    "minutes": _MINUTE_PLUS_DAILY_ALLOWED_OPERATOR_NAMES,
}


def get_allowed_operator_names(domains: Iterable[str] | None = None) -> set[str]:
    resolved_domains = parse_factor_domains(domains) if domains is not None else resolve_factor_domains()
    if not resolved_domains:
        return set(_BASE_ALLOWED_OPERATOR_NAMES)
    allowed: set[str] = set()
    for domain in resolved_domains:
        override = _DOMAIN_OPERATOR_OVERRIDES.get(domain)
        if override is None:
            allowed.update(_BASE_ALLOWED_OPERATOR_NAMES)
        else:
            allowed.update(override)
    return allowed


def get_prompt_operator_names(domains: Iterable[str] | None = None) -> set[str]:
    """Operators to show to the LLM. Internal template aliases stay validation-only."""
    return get_allowed_operator_names(domains).difference(_TEMPLATE_OPERATOR_ALIASES)


def get_allowed_field_names(domains: Iterable[str] | None = None) -> set[str]:
    resolved_domains = tuple(domains) if domains is not None else resolve_factor_domains()
    return set(_BASE_ALLOWED_FIELD_NAMES).union(get_domain_field_names(resolved_domains))


def _merge_arity(
    existing: tuple[int, int | None] | None,
    new: tuple[int, int | None],
) -> tuple[int, int | None]:
    if existing is None:
        return new
    existing_min, existing_max = existing
    new_min, new_max = new
    merged_min = min(existing_min, new_min)
    merged_max = None if existing_max is None or new_max is None else max(existing_max, new_max)
    return merged_min, merged_max


def get_operator_arity_constraints(domains: Iterable[str] | None = None) -> dict[str, tuple[int, int | None]]:
    resolved_domains = parse_factor_domains(domains) if domains is not None else resolve_factor_domains()
    if not resolved_domains:
        return dict(_BASE_OPERATOR_ARITY_CONSTRAINTS)

    constraints: dict[str, tuple[int, int | None]] = {}
    for domain in resolved_domains:
        for name, arity in _BASE_OPERATOR_ARITY_CONSTRAINTS.items():
            constraints[name] = _merge_arity(constraints.get(name), arity)
        if domain == "minutes":
            for name, arity in _MINUTE_OPERATOR_ARITY_CONSTRAINTS.items():
                constraints[name] = _merge_arity(constraints.get(name), arity)
    return constraints


def get_operator_signature_hints(domains: Iterable[str] | None = None) -> dict[str, str]:
    resolved_domains = parse_factor_domains(domains) if domains is not None else resolve_factor_domains()
    if not resolved_domains:
        return dict(_BASE_OPERATOR_SIGNATURE_HINTS)

    hints: dict[str, str] = {}
    for domain in resolved_domains:
        hints.update(_BASE_OPERATOR_SIGNATURE_HINTS)
        if domain == "minutes":
            hints.update(_MINUTE_OPERATOR_SIGNATURE_HINTS)
    return hints


def get_prompt_operator_signature_hints(domains: Iterable[str] | None = None) -> dict[str, str]:
    hints = get_operator_signature_hints(domains)
    for alias in _TEMPLATE_OPERATOR_ALIASES:
        hints.pop(alias, None)
    return hints


def get_base_operator_names() -> set[str]:
    return set(_BASE_ALLOWED_OPERATOR_NAMES).difference(_TEMPLATE_OPERATOR_ALIASES)


def get_minute_operator_names() -> set[str]:
    return set(_MINUTE_ALLOWED_OPERATOR_NAMES)


def get_operator_semantic_notes(domains: Iterable[str] | None = None) -> list[str]:
    resolved_domains = parse_factor_domains(domains) if domains is not None else resolve_factor_domains()
    allowed = get_allowed_operator_names(resolved_domains)
    notes: list[str] = []

    if "delay" in allowed or "ts_delay" in allowed:
        notes.append(
            "`delay(df, n)` and `ts_delay(df, n)` both shift the series backward by `n` periods on the time axis; "
            "they do not compute returns or differences."
        )
    if "ts_return" in allowed or "ts_pct" in allowed:
        notes.append(
            "`ts_return(df, n)` is the same lagged same-series return as `ts_pct(df, n)`, i.e. `df / delay(df, n) - 1`."
        )
    if "pct" in allowed and "minutes" in resolved_domains:
        notes.append(
            "In minutes-mode, `pct(a, b)` means same-time ratio `a / b - 1` on minute tensors; do not use it as a lagged return shorthand."
        )
    if "bucket" in allowed:
        notes.append(
            "`bucket(df, n)` is a per-date cross-sectional bucketing operator that assigns values into `0..n-1` groups; "
            "it is not a time aggregation or minute-bar resampler."
        )
    if "winsorize1" in allowed:
        notes.append(
            "`winsorize1(df, n)` clips cross-sectional values at `mean +/- n * std`; its second argument is an n-sigma width, not a percentile."
        )
    if "winsorize2" in allowed:
        notes.append(
            "`winsorize2(df, lower_bound, upper_bound)` clips cross-sectional values by lower/upper quantiles."
        )
    if {"WHERE", "where"} & allowed:
        notes.append(
            "Conditional logic is supported through allowlisted conditional operators such as `WHERE(cond, true_value, false_value)`, "
            "`IF_ELSE(cond, true_value, false_value)`, lowercase `where(...)`, `if_else(...)`, and `ifelse(...)`; "
            "ternary expressions `cond ? true_value : false_value` are rewritten to `WHERE(...)` by the runtime."
        )
    if {"AND", "OR", "and_", "or_"} & allowed:
        notes.append(
            "Logical composition is supported through allowlisted logical operators such as `AND(cond1, cond2)`, `OR(cond1, cond2)`, "
            "`and_(...)`, and `or_(...)`; infix `&`, `|`, `&&`, `||` are rewritten to aligned logical functions."
        )
    if {"GT", "LT", "GE", "LE", "EQ", "NE", "gt", "lt", "ge", "le", "eq", "ne"} & allowed:
        notes.append(
            "Comparison operators are supported through allowlisted comparison functions such as `GT/LT/GE/LE/EQ/NE` and `gt/lt/ge/le/eq/ne`; "
            "infix comparisons such as `a > b` and `a >= b` are rewritten to aligned comparison functions."
        )
    if "ts_reg" in allowed:
        notes.append(
            "`ts_reg(x, y, n, min_periods=None, rettype=0)` returns exactly one panel selected by `rettype`: "
            "`0=residual`, `1=slope/beta`, `2=intercept`, `3=R^2`."
        )
    if "ts_regression" in allowed:
        notes.append(
            "`ts_regression(x, y, n)` returns the rolling regression slope/beta only; it is equivalent to `ts_reg(x, y, n, rettype=1)`."
        )
    if "ts_regression2" in allowed:
        notes.append(
            "`ts_regression2(x, y, n, min_periods=None, rettype=0)` is the explicit-`rettype` variant of `ts_reg`; "
            "it returns one selected panel, not a tuple."
        )
    if "cs_multireg" in allowed:
        notes.append(
            "`cs_multireg([x1, x2], y, min_cs=30)` returns the cross-sectional residual of `y` after regressing on the control list; "
            "the first argument must be a list such as `[control1, control2]`."
        )
    if "ts_multireg" in allowed:
        notes.append(
            "`ts_multireg([x1, x2], y, n, min_periods=None, rettype=0)` returns a rolling regression-derived panel using a list of controls; "
            "the first argument must be a list such as `[control1, control2]`."
        )
    if "cs_neutralize" in allowed:
        notes.append(
            "`cs_neutralize(df)` accepts one argument only. Do not call `cs_neutralize(df, [controls])`; "
            "use `cs_multireg([controls], df)` for residualizing against explicit controls."
        )
    if {"cs_zscore", "zscore"} & allowed:
        notes.append(
            "`cs_zscore(df)` and `zscore(values)` accept one argument only. Do not pass thresholds, eps, or clip widths; "
            "use `safe_div(a, b, eps)` for denominator protection and `winsorize1/2` for clipping."
        )
    if "ts_beta" in allowed:
        notes.append(
            "`ts_beta(x, y, n)` returns the rolling regression slope/beta."
        )

    return notes
