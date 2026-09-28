from __future__ import annotations

import importlib
import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pandas as pd

from quantaalpha.factors.alignment.preflight_guard import validate_expression_against_registry
from quantaalpha.factors.coder.expr_parser import parse_expression, parse_symbol
from quantaalpha.factors.data_domains import (
    JOINT_PV_SHARED_TO_SINGULAR,
    parse_factor_domains,
    resolve_factor_domains,
)

from .performance_engine import DEFAULT_PARAMS, FactorPerformanceEngine
from .profiles import DEFAULT_PROFILE_ID
from .result_engine import FactorResultEngine

if TYPE_CHECKING:
    from quantaalpha.backtest.provider import FilesystemDataProvider


def _format_native_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    return value


def _tqstrategyserver_root() -> Path | None:
    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root.parent / "TQStrategyServer"
    if candidate.exists():
        return candidate
    return None


@contextmanager
def _factor_module_import_context():
    injected: list[str] = []
    for candidate in (Path(__file__).resolve().parents[2], _tqstrategyserver_root()):
        if candidate is None:
            continue
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
            injected.append(text)
    try:
        yield
    finally:
        for text in injected:
            try:
                sys.path.remove(text)
            except ValueError:
                pass


class TQStandaloneEvaluator:
    _DAILY_RAW_OHLC_FIELDS = frozenset({"opens", "highs", "lows", "closes"})
    _PV_LIKE_DOMAINS = frozenset({"pv", "hybrid"})
    _DOUBLE_HFQ_PREFIX = "hfq_hfq_"

    def __init__(self, data_provider: FilesystemDataProvider):
        self.dp = data_provider
        self.result_engine = FactorResultEngine(data_provider)
        self.performance_engine = FactorPerformanceEngine()
        self.analysis = importlib.import_module("quantaalpha.backtest.analysis")

    @classmethod
    def _normalize_meta_text(cls, value: Any) -> str:
        return str(getattr(value, "value", value) or "").strip().lower()

    @classmethod
    def _validate_daily_pv_field_contract(cls, meta: dict[str, Any], setting: dict[str, Any]) -> None:
        level = cls._normalize_meta_text(meta.get("level"))
        domain = cls._normalize_meta_text(meta.get("domain"))
        if level != "days" or domain not in cls._PV_LIKE_DOMAINS:
            return

        data_needed = {
            str(field_name).strip()
            for field_name in (setting.get("data_needed") or [])
            if str(field_name).strip()
        }
        raw_ohlc_fields = sorted(data_needed.intersection(cls._DAILY_RAW_OHLC_FIELDS))
        if raw_ohlc_fields:
            raise ValueError(
                "daily pv factors must use explicit hfq_* price fields instead of raw OHLC: "
                f"{raw_ohlc_fields}"
            )
        double_hfq_fields = sorted(field_name for field_name in data_needed if field_name.startswith(cls._DOUBLE_HFQ_PREFIX))
        if double_hfq_fields:
            raise ValueError(
                "daily pv factors must not use doubly adjusted hfq field names: "
                f"{double_hfq_fields}"
            )

    @classmethod
    def _validate_daily_pv_expression_fields(
        cls,
        used_fields: set[str],
        domains: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        resolved_domains = set(parse_factor_domains(domains or resolve_factor_domains()))
        if "pv" not in resolved_domains or "minutes" in resolved_domains:
            return
        raw_ohlc_fields = sorted(set(used_fields).intersection(cls._DAILY_RAW_OHLC_FIELDS))
        if raw_ohlc_fields:
            raise ValueError(
                "daily pv expressions must use explicit hfq_* price fields instead of raw OHLC: "
                f"{raw_ohlc_fields}"
            )
        double_hfq_fields = sorted(
            field_name for field_name in set(used_fields) if str(field_name).startswith(cls._DOUBLE_HFQ_PREFIX)
        )
        if double_hfq_fields:
            raise ValueError(
                "daily pv expressions must not use doubly adjusted hfq field names: "
                f"{double_hfq_fields}"
            )

    def _resolve_data_field_name(self, field_name: str, field_aliases: dict[str, str] | None = None) -> str:
        aliases = dict(field_aliases or {})
        return str(aliases.get(field_name) or field_name)

    def _exec_globals(self, data_needed: set[str], field_aliases: dict[str, str] | None = None) -> dict[str, Any]:
        exec_globals: dict[str, Any] = {"np": np, "pd": pd}
        for name, obj in vars(self.analysis).items():
            if name.startswith("_"):
                continue
            exec_globals[name] = obj
        for field_name in sorted(data_needed):
            runtime_field_name = self._resolve_data_field_name(field_name, field_aliases)
            exec_globals[field_name] = self.dp.get_single_data(runtime_field_name)
        exec_globals.update(
            {
                "ADD": exec_globals["add"],
                "SUBTRACT": exec_globals["sub"],
                "MULTIPLY": exec_globals["mul"],
                "DIVIDE": exec_globals["div"],
                "GT": lambda a, b: a > b,
                "LT": lambda a, b: a < b,
                "GE": lambda a, b: a >= b,
                "LE": lambda a, b: a <= b,
                "EQ": lambda a, b: a == b,
                "NE": lambda a, b: a != b,
                "AND": lambda a, b: a & b,
                "OR": lambda a, b: a | b,
                "WHERE": lambda cond, a, b: a.where(cond, b),
            }
        )
        for plural_name, singular_name in JOINT_PV_SHARED_TO_SINGULAR.items():
            if plural_name in exec_globals and singular_name not in exec_globals:
                exec_globals[singular_name] = exec_globals[plural_name]
            elif singular_name in exec_globals and plural_name not in exec_globals:
                exec_globals[plural_name] = exec_globals[singular_name]
        return exec_globals

    @staticmethod
    def _coerce_factor_value(raw_value: Any, factor_name: str) -> pd.DataFrame:
        if isinstance(raw_value, pd.Series):
            factor_value = raw_value.to_frame(name=factor_name)
        elif isinstance(raw_value, pd.DataFrame):
            factor_value = raw_value.copy()
        else:
            raise TypeError(f"calc_factor for {factor_name} must return Series/DataFrame, got {type(raw_value)}")
        factor_value = factor_value.astype(float)
        factor_value.index.name = "datetime"
        factor_value.columns.name = None
        return factor_value

    @staticmethod
    def _resolve_execution_constraints(execution_constraints: dict[str, Any] | None) -> dict[str, bool]:
        raw = dict(execution_constraints or {})
        return {
            "use_tradables": bool(raw.get("use_tradables", True)),
            "use_limit_masks": bool(raw.get("use_limit_masks", True)),
        }

    def _prepare_data_ctx(
        self,
        data_needed: list[str],
        univ: pd.DataFrame,
        pasteurization: bool,
        field_aliases: dict[str, str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        data_ctx = {
            field_name: self.dp.get_single_data(self._resolve_data_field_name(field_name, field_aliases))
            for field_name in data_needed
        }
        if pasteurization:
            for fld, data in data_ctx.items():
                data_ctx[fld] = data.where(univ)
        return data_ctx

    def _apply_module_setting_postprocess(self, factor_value: pd.DataFrame, setting: dict[str, Any]) -> pd.DataFrame:
        out = factor_value
        decay = int(setting.get("decay", 0) or 0)
        neutralize = setting.get("neutralize", None)

        if decay > 0:
            out = self.analysis.ts_decay_linear(out, decay)

        if neutralize == "industry":
            industrys = self.dp.get_single_data("industrys").reindex(index=out.index)
            out = self.analysis.group_neutralize(out, industrys)
        elif neutralize == "size":
            size = self.dp.get_single_data("Size").reindex(index=out.index)
            nlsize = self.dp.get_single_data("Nlsize").reindex(index=out.index)
            out = self.analysis.cs_multireg([size, nlsize], out)
        elif neutralize == "ram":
            rev = self.dp.get_single_data("Rev").reindex(index=out.index)
            mom = self.dp.get_single_data("Mom").reindex(index=out.index)
            out = self.analysis.cs_multireg([rev, mom], out)
        elif neutralize == "styles":
            x_list = [
                self.dp.get_single_data(fld).reindex(index=out.index)
                for fld in ["Beta", "Liq", "Mom", "Nlsize", "Rev", "Size", "Vol"]
            ]
            out = self.analysis.cs_multireg(x_list, out)
        elif neutralize == "complete":
            industrys = self.dp.get_single_data("industrys").reindex(index=out.index)
            out = self.analysis.group_neutralize(out, industrys)
            x_list = []
            for fld in ["Beta", "Liq", "Mom", "Nlsize", "Rev", "Size", "Vol"]:
                x = self.dp.get_single_data(fld).reindex(index=out.index)
                x = self.analysis.group_neutralize(x, industrys)
                x_list.append(x)
            out = self.analysis.cs_multireg(x_list, out)

        return out

    def _resolve_runtime_universe_name(self, requested_universe: str | None, level: str) -> str:
        normalized = str(requested_universe or "").strip() or "standards"
        if level != "minutes":
            return normalized
        available_fields = set(self.dp.list_datas())
        if normalized in available_fields:
            return normalized
        if "standards" in available_fields:
            return "standards"
        return normalized

    def infer_data_needed(self, expression: str) -> set[str]:
        check = validate_expression_against_registry(expression)
        if not check["ok"]:
            problems = []
            if check["unsupported_operators"]:
                problems.append(f"unsupported operators: {check['unsupported_operators']}")
            if check["unsupported_fields"]:
                problems.append(f"unsupported fields: {check['unsupported_fields']}")
            raise ValueError("; ".join(problems) or "expression validation failed")
        used_fields = set(check["used_fields"])
        self._validate_daily_pv_expression_fields(used_fields)
        return used_fields

    def evaluate_expression(
        self,
        factor_name: str,
        expression: str,
        field_aliases: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        data_needed = self.infer_data_needed(expression)
        parsed_expr = parse_expression(parse_symbol(expression, sorted(data_needed)))
        result = eval(parsed_expr, self._exec_globals(data_needed, field_aliases=field_aliases), {})
        return self._coerce_factor_value(result, factor_name)

    def evaluate_factor_module(
        self,
        module_globals: dict[str, Any],
        field_aliases: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        factor_type = str(module_globals["TYPE"])
        if factor_type != "regular":
            raise ValueError(f"unsupported factor type for standalone evaluator: {factor_type}")

        meta = module_globals["META"]
        setting = module_globals["SETTING"]
        self._validate_daily_pv_field_contract(meta, setting)
        calc_factor = module_globals["calc_factor"]
        factor_name = str(meta["factor_name"])
        level = str(meta["level"])
        if level not in {"days", "minutes"}:
            raise ValueError(f"unsupported factor level for standalone evaluator: {level}")

        data_needed = list(setting.get("data_needed") or [])
        universe_name = self._resolve_runtime_universe_name(setting.get("universe"), level)
        pasteurization = bool(setting.get("pasteurization", False))

        univ = self.dp.get_single_data(universe_name).astype(bool)
        data_ctx = self._prepare_data_ctx(data_needed, univ, pasteurization, field_aliases=field_aliases)
        for plural_name, singular_name in JOINT_PV_SHARED_TO_SINGULAR.items():
            if plural_name in data_ctx and singular_name not in data_ctx:
                data_ctx[singular_name] = data_ctx[plural_name]
            elif singular_name in data_ctx and plural_name not in data_ctx:
                data_ctx[plural_name] = data_ctx[singular_name]
            if plural_name in data_ctx:
                module_globals[plural_name] = data_ctx[plural_name]
            if singular_name in data_ctx:
                module_globals[singular_name] = data_ctx[singular_name]

        sig = inspect.signature(calc_factor)
        if level == "minutes":
            prepare_minute_datas = module_globals.get("prepare_minute_datas")
            if not callable(prepare_minute_datas):
                raise ValueError("minutes regular factor must define callable prepare_minute_datas()")
            if len(sig.parameters) != 2:
                raise ValueError("minutes regular factor calc_factor must have exactly two parameters: data_ctx, minute_ctx")
            import quantaalpha.backtest.minute_ops  # noqa: F401
            minute_sig = inspect.signature(prepare_minute_datas)
            if len(minute_sig.parameters) == 0:
                minute_ctx = prepare_minute_datas()
            elif len(minute_sig.parameters) == 1:
                minute_h5_path = module_globals.get("MinuteFactorEngine")
                if minute_h5_path is None:
                    from quantaalpha.backtest.minute_tools import MinuteFactorEngine
                else:
                    MinuteFactorEngine = minute_h5_path
                with h5py.File(MinuteFactorEngine().h5_path, "r") as handle:
                    minute_ctx = prepare_minute_datas(handle)
            else:
                raise ValueError("prepare_minute_datas must take either zero parameters or a single h5 handle")
            if not isinstance(minute_ctx, dict):
                raise ValueError("prepare_minute_datas must return dict")
            if pasteurization:
                for key, value in list(minute_ctx.items()):
                    if isinstance(value, pd.DataFrame):
                        minute_ctx[key] = value.where(univ.reindex(index=value.index, columns=value.columns))
            raw_value = calc_factor(data_ctx=data_ctx, minute_ctx=minute_ctx)
        else:
            if len(sig.parameters) != 1:
                raise ValueError("regular factor calc_factor must have exactly one parameter: data_ctx")
            raw_value = calc_factor(data_ctx=data_ctx)
        out = self._coerce_factor_value(raw_value, factor_name)
        out = out.where(univ.reindex(index=out.index, columns=out.columns))
        out = self._apply_module_setting_postprocess(out, setting)
        return out

    def evaluate_factor_module_path(
        self,
        path: str | Path,
        field_aliases: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        source_path = self.validate_factor_module(path)
        module_globals: dict[str, Any] = {}
        with _factor_module_import_context():
            exec(compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec"), module_globals, module_globals)
        return self.evaluate_factor_module(module_globals, field_aliases=field_aliases)

    def evaluate_factor_value(
        self,
        factor_name: str,
        factor_value: pd.DataFrame,
        profile_id: str = DEFAULT_PROFILE_ID,
        params: dict[str, Any] | None = None,
        execution_constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        full_params = dict(DEFAULT_PARAMS)
        if params:
            full_params.update(params)
        resolved_constraints = self._resolve_execution_constraints(execution_constraints)
        factor_result = self.result_engine.calc_result(
            factor_value,
            profile_id=profile_id,
            apply_suspend_mask=resolved_constraints["use_tradables"],
            apply_limit_mask=resolved_constraints["use_limit_masks"],
        )
        factor_performance = self.performance_engine.calc_basic_performance(
            factor_result,
            params=full_params,
        )
        return {
            "factor_value": factor_value,
            "factor_result": factor_result,
            "factor_performance": factor_performance,
            "summary": self.summarize_metrics(factor_name, factor_performance),
        }

    def evaluate_expression_factor(
        self,
        factor_name: str,
        expression: str,
        profile_id: str = DEFAULT_PROFILE_ID,
        params: dict[str, Any] | None = None,
        execution_constraints: dict[str, Any] | None = None,
        field_aliases: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        factor_value = self.evaluate_expression(factor_name, expression, field_aliases=field_aliases)
        return self.evaluate_factor_value(
            factor_name=factor_name,
            factor_value=factor_value,
            profile_id=profile_id,
            params=params,
            execution_constraints=execution_constraints,
        )

    @staticmethod
    def render_factor_module_text(spec: dict[str, Any]) -> str:
        data_needed = repr(list(spec.get("data_needed") or []))
        tag = str(spec.get("tag") or "")
        category = str(spec.get("category") or "unknown")
        domain = str(spec.get("domain") or "pv")
        decay = int(spec.get("decay", 0) or 0)
        neutralize = repr(spec.get("neutralize", None))
        lines = [
            "import numpy as np",
            "import pandas as pd",
            "from vendors.quant_lib.analysis import *",
            "from quant_union.common.quantEnum import DomainType, CategoryType",
            "",
            'TYPE = "regular"',
            "",
            "META = {",
            f'    "factor_name": "{spec["factor_name"]}",',
            f'    "author": "{spec.get("author", "quantaalpha")}",',
            f'    "level": "{spec.get("level", "days")}",',
            f'    "domain": DomainType.{domain},',
            f'    "tag": "{tag}",',
            f'    "category": "{category}",',
            "}",
            "",
            "SETTING = {",
            f"    \"data_needed\": {data_needed},",
            f'    "universe": "{spec.get("universe", "standards")}",',
            f'    "pasteurization": {bool(spec.get("pasteurization", True))},',
            f'    "decay": {decay},',
            f'    "neutralize": {neutralize},',
            "}",
            "",
            "def calc_factor(data_ctx):",
        ]
        for field_name in spec.get("data_needed") or []:
            lines.append(f'    {field_name} = data_ctx["{field_name}"]')
        lines.append(f'    return {spec["formula"]}')
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def validate_factor_module(path: str | Path) -> Path:
        module_globals: dict[str, Any] = {}
        source_path = Path(path)
        with _factor_module_import_context():
            exec(compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec"), module_globals, module_globals)
        required = {"TYPE", "META", "SETTING", "calc_factor"}
        missing = sorted(required - set(module_globals))
        if missing:
            raise ValueError(f"invalid factor module, missing: {missing}")
        if not callable(module_globals["calc_factor"]):
            raise ValueError("calc_factor must be callable")
        meta = module_globals["META"]
        setting = module_globals["SETTING"]
        TQStandaloneEvaluator._validate_daily_pv_field_contract(meta, setting)
        level = str(meta.get("level") or "")
        calc_factor = module_globals["calc_factor"]
        sig = inspect.signature(calc_factor)
        if level == "minutes":
            if not callable(module_globals.get("prepare_minute_datas")):
                raise ValueError("minutes factor module must define callable prepare_minute_datas()")
            if len(sig.parameters) != 2:
                raise ValueError("minutes regular factor calc_factor must have exactly two parameters: data_ctx, minute_ctx")
        elif level == "days":
            if len(sig.parameters) != 1:
                raise ValueError("regular factor calc_factor must have exactly one parameter: data_ctx")
        else:
            raise ValueError(f"unsupported factor level for standalone evaluator: {level}")
        return source_path

    @staticmethod
    def summarize_metrics(factor_name: str, metrics: dict[str, Any]) -> pd.DataFrame:
        summary = pd.DataFrame({factor_name: pd.Series({k: _format_native_value(v) for k, v in metrics.items()})})
        summary.index.name = "metric"
        return summary
