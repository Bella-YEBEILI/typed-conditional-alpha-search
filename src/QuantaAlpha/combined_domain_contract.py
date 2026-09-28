from __future__ import annotations

import ast
import inspect
import re

import quantaalpha.backtest.minute_ops  # noqa: F401
from quantaalpha.backtest.minute_tools import MinuteFactorEngine
from quantaalpha.factors.data_domains import MINUTE_FIELDS, PV_FIELDS, parse_factor_domains

RAW_DAILY_OHLC_FIELDS = frozenset({"opens", "highs", "lows", "closes"})
DOUBLE_HFQ_PATTERN = re.compile(r"\bhfq_hfq_[A-Za-z0-9_]+\b")


def is_joint_pv_minutes_run(active_domains: tuple[str, ...] | list[str] | None = None) -> bool:
    domains = set(parse_factor_domains(active_domains))
    return "pv" in domains and "minutes" in domains


def infer_module_level(module_text: str) -> str:
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


def _extract_string_literal(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except Exception:
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _extract_string_collection(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    try:
        value = ast.literal_eval(node)
    except Exception:
        return set()
    if isinstance(value, str):
        text = value.strip()
        return {text} if text else set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return set()


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


def _extract_call_argument(
    node: ast.Call,
    positional_index: int,
    keyword_name: str,
) -> ast.AST | None:
    if len(node.args) > positional_index:
        return node.args[positional_index]
    for kw in node.keywords:
        if kw.arg == keyword_name:
            return kw.value
    return None


def _parse_op_spec_literal(node: ast.AST | None) -> tuple[str | None, tuple | dict | None, bool]:
    if node is None:
        return None, None, True
    try:
        value = ast.literal_eval(node)
    except Exception:
        return None, None, False
    if value is None:
        return None, None, True
    if isinstance(value, str):
        text = value.strip()
        return (text if text else None), (), True
    if isinstance(value, tuple) and len(value) >= 1 and isinstance(value[0], str):
        name = value[0].strip()
        if len(value) == 1:
            return name, (), True
        if len(value) == 2 and isinstance(value[1], dict):
            return name, value[1], True
        return name, tuple(value[1:]), True
    return None, None, False


def _registered_minute_ops() -> set[str]:
    return set(MinuteFactorEngine._OPS.keys())


def _minute_op_base_arity(name: str) -> int | None:
    func = MinuteFactorEngine._OPS.get(name)
    if func is None:
        return None
    target = getattr(func, "py_func", func)
    try:
        signature = inspect.signature(target)
    except Exception:
        return None
    param_order = tuple(MinuteFactorEngine._OPS_PARAM_ORDER.get(name) or ())
    positional_params = [
        p
        for p in signature.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return max(0, len(positional_params) - len(param_order))


def _validate_minute_engine_calls(module_text: str) -> list[str]:
    text = str(module_text or "").strip()
    if not text:
        return []
    try:
        tree = ast.parse(text)
    except Exception:
        return []

    registered_ops = _registered_minute_ops()
    errors: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"run", "cached_run"}:
            continue

        inputs_node = _extract_call_argument(node, 0, "inputs")
        operator_node = _extract_call_argument(node, 1, "operator")
        aggregator_node = _extract_call_argument(node, 2, "aggregator")

        input_fields = _extract_string_collection(inputs_node)
        if not input_fields:
            errors.append(
                "MinuteFactorEngine.run/cached_run must use literal raw minute field names in `inputs`."
            )
            continue

        invalid_input_fields = sorted(input_fields - set(MINUTE_FIELDS))
        if invalid_input_fields:
            errors.append(
                "MinuteFactorEngine.run/cached_run only accepts raw minute fields in `inputs`; "
                f"found {invalid_input_fields}."
            )

        operator_name, operator_params, operator_ok = _parse_op_spec_literal(operator_node)
        if not operator_ok or not operator_name:
            errors.append(
                "MinuteFactorEngine.run/cached_run must use a literal registered operator spec, "
                "for example `operator=\"mean\"` or `operator=(\"corr\", {\"shift\": 1})`."
            )
            continue
        if operator_name not in registered_ops:
            errors.append(
                f"MinuteFactorEngine operator `{operator_name}` is not registered. "
                f"Available operators: {sorted(registered_ops)}."
            )
            continue

        operator_arity = _minute_op_base_arity(operator_name)
        if operator_arity is not None and len(input_fields) != operator_arity:
            errors.append(
                f"MinuteFactorEngine operator `{operator_name}` expects {operator_arity} input field(s), "
                f"but the call provides {len(input_fields)}: {sorted(input_fields)}."
            )

        if isinstance(operator_params, dict):
            allowed_param_order = tuple(MinuteFactorEngine._OPS_PARAM_ORDER.get(operator_name) or ())
            unknown_keys = sorted(set(operator_params) - set(allowed_param_order))
            if unknown_keys:
                errors.append(
                    f"MinuteFactorEngine operator `{operator_name}` does not accept parameter keys {unknown_keys}; "
                    f"allowed keys are {list(allowed_param_order)}."
                )

        aggregator_name, aggregator_params, aggregator_ok = _parse_op_spec_literal(aggregator_node)
        if aggregator_node is not None and not aggregator_ok:
            errors.append(
                "MinuteFactorEngine.run/cached_run aggregator must be a literal registered operator name or tuple spec."
            )
            continue
        if aggregator_name:
            if aggregator_name not in registered_ops:
                errors.append(
                    f"MinuteFactorEngine aggregator `{aggregator_name}` is not registered. "
                    f"Available aggregators/operators: {sorted(registered_ops)}."
                )
                continue
            aggregator_arity = _minute_op_base_arity(aggregator_name)
            if aggregator_arity is not None and aggregator_arity != 1:
                errors.append(
                    f"MinuteFactorEngine aggregator `{aggregator_name}` is invalid here because it expects "
                    f"{aggregator_arity} inputs; aggregators must be unary reductions like mean/std/sum/min/max."
                )
            if isinstance(aggregator_params, dict):
                allowed_agg_params = tuple(MinuteFactorEngine._OPS_PARAM_ORDER.get(aggregator_name) or ())
                unknown_agg_keys = sorted(set(aggregator_params) - set(allowed_agg_params))
                if unknown_agg_keys:
                    errors.append(
                        f"MinuteFactorEngine aggregator `{aggregator_name}` does not accept parameter keys {unknown_agg_keys}; "
                        f"allowed keys are {list(allowed_agg_params)}."
                    )

    return errors


def extract_setting_data_needed(module_text: str) -> set[str]:
    text = str(module_text or "").strip()
    if not text:
        return set()
    try:
        tree = ast.parse(text)
    except Exception:
        return set()

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "SETTING":
            continue
        if not isinstance(node.value, ast.Dict):
            return set()
        for key_node, value_node in zip(node.value.keys, node.value.values):
            if _extract_string_literal(key_node) == "data_needed":
                return _extract_string_collection(value_node)
        return set()
    return set()


def extract_data_ctx_fields(module_text: str) -> set[str]:
    text = str(module_text or "").strip()
    if not text:
        return set()
    try:
        tree = ast.parse(text)
    except Exception:
        return set()

    fields: set[str] = set()
    embedded_expression_fields: set[str] = set()
    declared_daily_fields = extract_setting_data_needed(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "data_ctx":
            field_name = _extract_string_literal(node.slice)
            if field_name:
                fields.add(field_name)
            continue

        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "data_ctx":
            continue
        if node.func.attr not in {"get", "pop"}:
            continue
        if not node.args:
            continue
        field_name = _extract_string_literal(node.args[0])
        if field_name:
            fields.add(field_name)
        continue

    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in {"FINAL_EXPRESSION", "final_expression"}:
            continue
        final_expression_text = _extract_string_literal(node.value) or ""
        embedded_expression_fields.update(_extract_embedded_data_ctx_fields(final_expression_text))
        embedded_expression_fields.update(_extract_identifier_tokens(final_expression_text) & declared_daily_fields)
    return fields | embedded_expression_fields


def extract_minute_inputs(module_text: str) -> set[str]:
    text = str(module_text or "").strip()
    if not text:
        return set()
    try:
        tree = ast.parse(text)
    except Exception:
        return set()

    minute_fields: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_single_minute":
            candidates: list[ast.AST] = []
            if len(node.args) >= 2:
                candidates.append(node.args[1])
            for kw in node.keywords:
                if kw.arg in {"fld", "field_name"}:
                    candidates.append(kw.value)
            for candidate in candidates:
                minute_fields.update(_extract_string_collection(candidate))
            continue

        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"run", "cached_run"}:
            continue
        candidates = []
        if node.args:
            candidates.append(node.args[0])
        for kw in node.keywords:
            if kw.arg == "inputs":
                candidates.append(kw.value)
        for candidate in candidates:
            minute_fields.update(_extract_string_collection(candidate))
    return minute_fields


def validate_joint_pv_minutes_contract(
    module_text: str,
    active_domains: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    if not is_joint_pv_minutes_run(active_domains):
        return []

    text = str(module_text or "").strip()
    if not text:
        return ["combined pv/minutes run requires a concrete minute factor module, but the module text is empty."]

    try:
        ast.parse(text)
    except SyntaxError as exc:
        return [
            "combined pv/minutes module has a syntax error before runtime validation: "
            f"{exc.msg} (line {exc.lineno}, column {exc.offset})."
        ]

    errors: list[str] = []
    module_level = infer_module_level(text)
    minute_inputs = extract_minute_inputs(text)
    data_needed = extract_setting_data_needed(text)
    data_ctx_fields = extract_data_ctx_fields(text)

    if module_level != "minutes":
        errors.append(
            "combined pv/minutes factor must be a minute module with META['level'] = 'minutes' and prepare_minute_datas()."
        )

    if "prepare_minute_datas" not in text or "calc_factor" not in text:
        errors.append(
            "combined pv/minutes factor must define both prepare_minute_datas() and calc_factor(data_ctx, minute_ctx)."
        )

    if not minute_inputs:
        errors.append(
            "combined pv/minutes factor must use at least one raw minute field via MinuteFactorEngine.run(...) or load_single_minute(...)."
        )
    else:
        invalid_minute_inputs = sorted(minute_inputs - set(MINUTE_FIELDS))
        if invalid_minute_inputs:
            errors.append(
                "minute runtime may only consume raw minute fields "
                f"{sorted(MINUTE_FIELDS)}; found invalid minute inputs {invalid_minute_inputs}."
            )

    raw_daily_ohlc_in_code = sorted(data_ctx_fields & RAW_DAILY_OHLC_FIELDS)
    if raw_daily_ohlc_in_code:
        errors.append(
            "daily pv access inside combined pv/minutes factors must use explicit hfq_* fields; "
            f"raw daily OHLC via data_ctx is not allowed: {raw_daily_ohlc_in_code}."
        )

    raw_daily_ohlc_in_declared = sorted(data_needed & RAW_DAILY_OHLC_FIELDS)
    if raw_daily_ohlc_in_declared:
        errors.append(
            "SETTING['data_needed'] for combined pv/minutes factors must not declare raw daily OHLC; "
            f"use hfq_* fields instead: {raw_daily_ohlc_in_declared}."
        )

    double_hfq_fields = sorted(
        set(DOUBLE_HFQ_PATTERN.findall(text))
        | {field for field in data_needed if field.startswith("hfq_hfq_")}
        | {field for field in data_ctx_fields if field.startswith("hfq_hfq_")}
    )
    if double_hfq_fields:
        errors.append(
            "combined pv/minutes factor must not use doubly adjusted daily pv field names; "
            f"found {double_hfq_fields}."
        )

    pv_daily_fields_used = sorted(data_ctx_fields & set(PV_FIELDS))
    if not pv_daily_fields_used:
        errors.append(
            "combined pv/minutes factor must use at least one daily pv field from data_ctx, such as hfq_closes or turnover."
        )

    declared_pv_fields = sorted(data_needed & set(PV_FIELDS))
    if not declared_pv_fields:
        errors.append(
            "combined pv/minutes factor must declare at least one daily pv field in SETTING['data_needed']."
        )

    undeclared_daily_fields = sorted((data_ctx_fields & set(PV_FIELDS)) - data_needed)
    if undeclared_daily_fields:
        errors.append(
            "every daily pv field accessed through data_ctx must also appear in SETTING['data_needed']; "
            f"missing declarations: {undeclared_daily_fields}."
        )

    errors.extend(_validate_minute_engine_calls(text))

    return errors
