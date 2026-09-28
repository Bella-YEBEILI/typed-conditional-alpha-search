from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantaalpha.backtest.evaluator import TQStandaloneEvaluator, _factor_module_import_context
from quantaalpha.factors.alignment.preflight_guard import validate_expression_against_registry
from quantaalpha.factors.alignment.registry import get_allowed_field_names
from quantaalpha.factors.data_domains import (
    JOINT_PV_DAILY_SHARED_FIELD_ALIASES,
    resolve_factor_domains,
    validate_multi_domain_field_coverage,
)

from .analysis import cs_multireg, group_neutralize
from .config import (
    EXECUTION_CONSTRAINT_FIELD_REQUIREMENTS,
    REQUIRED_BACKTEST_FIELDS,
    TRANSFORM_FIELD_REQUIREMENTS,
    load_backtest_config,
)
from .provider import FilesystemDataProvider, StaticFieldDataProvider
from .submission_check import evaluate_submission_metrics

SPEC_FALLBACK_FIELDS = frozenset(
    {
        "opens",
        "closes",
        "highs",
        "lows",
        "volumes",
        "returns",
        "ctc_returns",
        "hfq_closes",
        "volume",
        "vwap",
        "turnover",
        "industrys",
        "tradables",
        "adj_factors",
        "standards",
        "hs300s",
        "zz1000s",
        "Beta",
        "Liq",
        "Mom",
        "Nlsize",
        "Rev",
        "Size",
        "Vol",
    }
)

TRANSFORM_ROW_SUFFIXES = ("hs300s", "zz1000s", "complete")


def _normalize_transform_entries(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        raw_items = values
    else:
        raw_items = [values]
    normalized: list[str] = []
    for item in raw_items:
        text = str(item).strip().lower()
        if text and text not in normalized:
            normalized.append(text)
    return normalized
def summarize_tq_check_result(
    summary_df: pd.DataFrame,
    factor_name: str,
    metric_mode: str = "long_only",
    use_net_metrics: bool = False,
    profile_id: str | None = None,
) -> pd.Series:
    if factor_name not in summary_df.index:
        raise KeyError(f"factor '{factor_name}' not found in TQ check summary")

    row = summary_df.loc[factor_name]
    result: dict[str, Any] = {
        "tq_profile_id": profile_id,
        "tq_metric_mode": metric_mode,
        "tq_use_net_metrics": bool(use_net_metrics),
    }
    for key, value in row.items():
        result.setdefault(str(key), value)

    for suffix in TRANSFORM_ROW_SUFFIXES:
        row_name = f"{factor_name}_{suffix}"
        if row_name not in summary_df.index:
            continue
        trow = summary_df.loc[row_name]
        for key, value in trow.items():
            result[f"{suffix}_{key}"] = value

    return pd.Series(result, name=factor_name)


class TQUpstreamBridge:
    """QuantaAlpha-local TQ-compatible bridge without external TQStrategyServer dependency."""

    def __init__(self, config_path: str | Path | None = None):
        self.project_root = Path(__file__).resolve().parents[2]
        self.config = load_backtest_config(config_path)
        self.factor_base_dir = self._resolve_factor_base_dir()
        self.candidate_dir = self._resolve_path(self.config["output"]["candidate_dir"])
        data_dir = self.config["tq"].get("data_dir")
        self.data_dir = None if data_dir in (None, "") else self._resolve_path(data_dir)
        self.data_start_date = str(self.config["tq"].get("data_start_date") or "20150101")

        self._runtime_dp: FilesystemDataProvider | None = None
        self._spec_dp = None
        self._evaluator: TQStandaloneEvaluator | None = None

        self.factor_base_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _merge_params(base: dict[str, Any] | None, override: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = dict(base or {})
        if isinstance(override, dict):
            merged.update(override)
        return merged

    @staticmethod
    def _normalize_execution_constraints(values: dict[str, Any] | None) -> dict[str, bool]:
        raw = dict(values or {})
        return {
            "use_tradables": bool(raw.get("use_tradables", True)),
            "use_limit_masks": bool(raw.get("use_limit_masks", True)),
        }

    def get_eval_params(self, context: str = "default") -> dict[str, Any]:
        base_params = dict(self.config.get("params") or {})
        if context == "standalone":
            override = ((self.config.get("standalone") or {}).get("params") or {})
        elif context == "mining":
            override = ((self.config.get("mining") or {}).get("params") or {})
        else:
            override = {}
        return self._merge_params(base_params, override)

    def get_factor_field_aliases(self, context: str = "default") -> dict[str, str]:
        del context
        return dict(JOINT_PV_DAILY_SHARED_FIELD_ALIASES)

    def get_execution_constraints(self, context: str = "default") -> dict[str, bool]:
        base = self._normalize_execution_constraints(self.config.get("execution_constraints") or {})
        if context == "standalone":
            override = ((self.config.get("standalone") or {}).get("execution_constraints") or {})
        elif context == "mining":
            override = ((self.config.get("mining") or {}).get("execution_constraints") or {})
        else:
            override = {}
        merged = dict(base)
        merged.update({
            str(key): bool(value)
            for key, value in dict(override).items()
            if str(key) in {"use_tradables", "use_limit_masks"}
        })
        return self._normalize_execution_constraints(merged)

    def get_factor_value_window(self, context: str = "default") -> tuple[str | None, str | None]:
        factor_window = self.config.get("factor_value_window") or {}
        if not isinstance(factor_window, dict):
            factor_window = {}
        start = factor_window.get("start")
        end = factor_window.get("end")
        if start in (None, ""):
            standalone_params = ((self.config.get("standalone") or {}).get("params") or {})
            start = standalone_params.get("start")
        if start in (None, ""):
            mining_periods = ((self.config.get("mining") or {}).get("periods") or {})
            train_period = mining_periods.get("train") or {}
            start = train_period.get("start")
        if end in (None, ""):
            standalone_params = ((self.config.get("standalone") or {}).get("params") or {})
            end = standalone_params.get("end")
        if end in (None, ""):
            mining_periods = ((self.config.get("mining") or {}).get("periods") or {})
            test_period = mining_periods.get("test") or {}
            end = test_period.get("end")
        return (
            str(start).strip() if start not in (None, "") else None,
            str(end).strip() if end not in (None, "") else None,
        )

    @staticmethod
    def _apply_factor_value_window(
        factor_value: pd.DataFrame,
        start: str | None,
        end: str | None,
    ) -> pd.DataFrame:
        if start is None and end is None:
            return factor_value
        return factor_value.loc[start:end].copy()

    def get_mining_period_params(self) -> dict[str, dict[str, Any]]:
        periods = ((self.config.get("mining") or {}).get("periods") or {})
        base_params = self.get_eval_params("mining")
        return {
            str(name): self._merge_params(base_params, spec if isinstance(spec, dict) else {})
            for name, spec in periods.items()
        }

    def get_backtest_universe(self) -> str:
        return str((self.config.get("spec_defaults") or {}).get("universe") or "standards").strip() or "standards"

    def get_transform_spec(self) -> dict[str, list[str]]:
        raw = self.config.get("transform_spec") or {}
        return {
            "subuniverse": _normalize_transform_entries(raw.get("subuniverse")),
            "neutralize": _normalize_transform_entries(raw.get("neutralize")),
        }

    def get_primary_mining_split(self) -> str:
        periods = self.get_mining_period_params()
        primary = str(((self.config.get("mining") or {}).get("primary_split") or "train")).strip() or "train"
        if primary in periods:
            return primary
        if "train" in periods:
            return "train"
        return next(iter(periods), "train")

    def get_train_mining_splits(self) -> list[str]:
        periods = self.get_mining_period_params()
        mining_cfg = self.config.get("mining") or {}
        configured = mining_cfg.get("train_splits") or []
        normalized = [
            str(name).strip()
            for name in configured
            if str(name).strip() and str(name).strip() in periods
        ]
        if normalized:
            return normalized
        primary = self.get_primary_mining_split()
        return [primary] if primary in periods else []

    def get_test_mining_split(self) -> str:
        periods = self.get_mining_period_params()
        test_split = str(((self.config.get("mining") or {}).get("test_split") or "")).strip()
        if test_split and test_split in periods:
            return test_split
        return ""

    def summarize_factor_result(
        self,
        factor_name: str,
        factor_result: dict[str, Any],
        params: dict[str, Any] | None = None,
        context: str = "default",
    ) -> pd.Series:
        perf = self._get_evaluator().performance_engine.calc_basic_performance(
            factor_result,
            params=params or self.get_eval_params(context),
        )
        return TQStandaloneEvaluator.summarize_metrics(factor_name, perf)[factor_name].copy()

    def summarize_factor_value(
        self,
        factor_name: str,
        factor_value: pd.DataFrame,
        params: dict[str, Any] | None = None,
        context: str = "default",
        execution_constraints: dict[str, Any] | None = None,
    ) -> pd.Series:
        effective_params = dict(params or self.get_eval_params(context))
        effective_constraints = self._normalize_execution_constraints(
            execution_constraints or self.get_execution_constraints(context)
        )
        perf = self._get_evaluator().evaluate_factor_value(
            factor_name=factor_name,
            factor_value=factor_value.copy(),
            profile_id=self.config["profile_id"],
            params=effective_params,
            execution_constraints=effective_constraints,
        )["factor_performance"]
        return TQStandaloneEvaluator.summarize_metrics(factor_name, perf)[factor_name].copy()

    def evaluate_factor_value(
        self,
        factor_name: str,
        factor_value: pd.DataFrame,
        profile_id: str | None = None,
        params: dict[str, Any] | None = None,
        context: str = "default",
        execution_constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        effective_params = dict(params or self.get_eval_params(context))
        effective_constraints = self._normalize_execution_constraints(
            execution_constraints or self.get_execution_constraints(context)
        )
        return self._get_evaluator().evaluate_factor_value(
            factor_name=factor_name,
            factor_value=factor_value.copy(),
            profile_id=profile_id or self.config["profile_id"],
            params=effective_params,
            execution_constraints=effective_constraints,
        )

    def _align_aux_frame(self, field_name: str, factor_value: pd.DataFrame) -> pd.DataFrame:
        frame = self._get_runtime_data_provider(require_data=True).get_single_data(field_name)
        return frame.reindex(index=factor_value.index, columns=factor_value.columns)

    def _apply_subuniverse_mask(self, factor_value: pd.DataFrame, subuniverse_name: str) -> pd.DataFrame:
        mask = self._align_aux_frame(subuniverse_name, factor_value).eq(True)
        return factor_value.where(mask)

    def _apply_size_neutralize(self, factor_value: pd.DataFrame) -> pd.DataFrame:
        regressors = [
            self._align_aux_frame("Size", factor_value),
            self._align_aux_frame("Nlsize", factor_value),
        ]
        return cs_multireg(regressors, factor_value)

    def _apply_styles_neutralize(self, factor_value: pd.DataFrame) -> pd.DataFrame:
        regressors = [
            self._align_aux_frame(field_name, factor_value)
            for field_name in ("Beta", "Liq", "Mom", "Nlsize", "Rev", "Size", "Vol")
        ]
        return cs_multireg(regressors, factor_value)

    def _apply_complete_neutralize(self, factor_value: pd.DataFrame) -> pd.DataFrame:
        industry_df = self._align_aux_frame("industrys", factor_value)
        neutralized_factor = group_neutralize(factor_value, industry_df)
        regressors = [
            group_neutralize(self._align_aux_frame(field_name, factor_value), industry_df)
            for field_name in ("Beta", "Liq", "Mom", "Nlsize", "Rev", "Size", "Vol")
        ]
        return cs_multireg(regressors, neutralized_factor)

    def _apply_neutralize_transform(self, factor_value: pd.DataFrame, neutralize_name: str) -> pd.DataFrame:
        if neutralize_name == "industry":
            industry_df = self._align_aux_frame("industrys", factor_value)
            return group_neutralize(factor_value, industry_df)
        if neutralize_name == "size":
            return self._apply_size_neutralize(factor_value)
        if neutralize_name == "styles":
            return self._apply_styles_neutralize(factor_value)
        if neutralize_name == "complete":
            return self._apply_complete_neutralize(factor_value)
        raise ValueError(f"Unsupported neutralize transform: {neutralize_name}")

    def apply_primary_transform(self, factor_value: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        transform_spec = self.get_transform_spec()
        transformed = factor_value.copy()
        applied_steps: list[str] = []

        for subuniverse_name in transform_spec["subuniverse"]:
            transformed = self._apply_subuniverse_mask(transformed, subuniverse_name)
            applied_steps.append(f"subuniverse={subuniverse_name}")

        for neutralize_name in transform_spec["neutralize"]:
            transformed = self._apply_neutralize_transform(transformed, neutralize_name)
            applied_steps.append(f"neutralize={neutralize_name}")

        active = bool(applied_steps)
        return transformed, {
            "active": active,
            "base_universe": self.get_backtest_universe(),
            "subuniverse": list(transform_spec["subuniverse"]),
            "neutralize": list(transform_spec["neutralize"]),
            "metric_source": "transformed" if active else "primary",
            "applied_steps": applied_steps,
            "description": "primary" if not active else ", ".join(applied_steps),
        }

    def evaluate_factor_file_value(self, file_path: str | Path, context: str = "default") -> pd.DataFrame:
        runtime_check = self.check_runtime_fields(context=context)
        if not runtime_check["ok"]:
            raise ValueError(f"TQ runtime field check failed: {runtime_check['reason']} -> {runtime_check['missing_required']}")
        factor_value = self._get_evaluator().evaluate_factor_module_path(
            file_path,
            field_aliases=self.get_factor_field_aliases(context),
        )
        factor_start, factor_end = self.get_factor_value_window(context)
        return self._apply_factor_value_window(factor_value, factor_start, factor_end)

    def evaluate_submission_checks_for_value(
        self,
        factor_name: str,
        factor_value: pd.DataFrame,
        params: dict[str, Any] | None = None,
        context: str = "standalone",
        profile_id: str | None = None,
        execution_constraints: dict[str, Any] | None = None,
        precomputed_stage_payloads: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        effective_params = dict(params or self.get_eval_params(context))
        effective_constraints = self._normalize_execution_constraints(
            execution_constraints or self.get_execution_constraints(context)
        )
        precomputed_stage_payloads = dict(precomputed_stage_payloads or {})

        stage_values = {
            "raw": factor_value,
            "zz1000s": self._apply_subuniverse_mask(factor_value, "zz1000s"),
            "complete": self._apply_complete_neutralize(factor_value),
        }

        metrics_by_stage: dict[str, dict[str, Any]] = {}
        summary_rows: dict[str, dict[str, Any]] = {}
        stage_payloads: dict[str, dict[str, Any]] = {}
        row_names = {
            "raw": factor_name,
            "zz1000s": f"{factor_name}_zz1000s",
            "complete": f"{factor_name}_complete",
        }

        for stage_name, stage_factor_value in stage_values.items():
            precomputed_payload = precomputed_stage_payloads.get(stage_name)
            if isinstance(precomputed_payload, dict):
                evaluation_payload = {
                    "factor_performance": dict(precomputed_payload.get("factor_performance") or {}),
                    "factor_result": precomputed_payload.get("factor_result"),
                    "summary": precomputed_payload.get("summary").copy()
                    if isinstance(precomputed_payload.get("summary"), pd.DataFrame)
                    else pd.DataFrame(precomputed_payload.get("summary") or {}),
                }
            else:
                evaluation = self.evaluate_factor_value(
                    factor_name=factor_name,
                    factor_value=stage_factor_value,
                    profile_id=profile_id or self.config["profile_id"],
                    params=effective_params,
                    context=context,
                    execution_constraints=effective_constraints,
                )
                evaluation_payload = {
                    "factor_performance": dict(evaluation["factor_performance"]),
                    "factor_result": evaluation["factor_result"],
                    "summary": evaluation["summary"].copy(),
                }
            metrics = dict(evaluation_payload["factor_performance"])
            metrics_by_stage[stage_name] = metrics
            stage_payloads[stage_name] = evaluation_payload
            summary_rows[row_names[stage_name]] = dict(metrics)

        summary = pd.DataFrame(summary_rows).T
        summary.index.name = "factor"
        summary["max_prod_corr"] = np.nan
        summary["max_prod_corr_factor"] = ""
        summary["avg_prod_corr"] = np.nan

        criteria_result = evaluate_submission_metrics(metrics_by_stage)
        return {
            "factor_name": factor_name,
            "summary": summary,
            "metrics_by_stage": metrics_by_stage,
            "stage_payloads": stage_payloads,
            "criteria": criteria_result,
        }

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.project_root / path

    def _resolve_factor_base_dir(self) -> Path:
        base_dir = self.config["tq"].get("factor_base_dir")
        if base_dir not in (None, ""):
            return self._resolve_path(base_dir)
        return self.project_root / "data" / "backtest" / "factor_base"

    def _get_runtime_data_provider(self, require_data: bool = True):
        if self._runtime_dp is None:
            if self.data_dir is None:
                if require_data:
                    raise ValueError(
                        "TQ runtime data is not configured. Please set QUANTAALPHA_DATA_ROOT in .env."
                    )
                return None
            self._runtime_dp = FilesystemDataProvider(self.data_dir, self.data_start_date)
        return self._runtime_dp

    def _get_spec_data_provider(self):
        if self._spec_dp is None:
            if self.data_dir is not None:
                self._spec_dp = self._get_runtime_data_provider(require_data=True)
            else:
                self._spec_dp = StaticFieldDataProvider(SPEC_FALLBACK_FIELDS)
        return self._spec_dp

    def _get_expression_allowed_fields(self) -> set[str]:
        domains = resolve_factor_domains()
        domain_allowed_fields = set(get_allowed_field_names(domains))
        runtime_available_fields = set(self._get_spec_data_provider().list_datas())
        return domain_allowed_fields.intersection(runtime_available_fields)

    def _get_evaluator(self) -> TQStandaloneEvaluator:
        if self._evaluator is None:
            self._evaluator = TQStandaloneEvaluator(self._get_runtime_data_provider(require_data=True))
        return self._evaluator

    def check_runtime_fields(self, context: str = "default") -> dict[str, Any]:
        required_fields = set(REQUIRED_BACKTEST_FIELDS)
        execution_constraints = self.get_execution_constraints(context)
        for flag, fields in EXECUTION_CONSTRAINT_FIELD_REQUIREMENTS.items():
            if execution_constraints.get(flag):
                required_fields.update(fields)
        transform_spec = self.config.get("transform_spec") or {}
        for field in transform_spec.get("subuniverse", []):
            required_fields.update(TRANSFORM_FIELD_REQUIREMENTS["subuniverse"].get(field, set()))
        for field in transform_spec.get("neutralize", []):
            required_fields.update(TRANSFORM_FIELD_REQUIREMENTS["neutralize"].get(field, set()))

        if self.data_dir is None:
            return {
                "ok": False,
                "available_fields": [],
                "missing_required": sorted(required_fields),
                "reason": "tq.data_dir not configured",
            }

        available = set(self._get_runtime_data_provider(require_data=True).list_datas())
        missing_required = sorted(required_fields - available)
        return {
            "ok": len(missing_required) == 0,
            "available_fields": sorted(available),
            "missing_required": missing_required,
            "reason": "" if len(missing_required) == 0 else "missing TQ runtime fields",
        }

    def build_spec(
        self,
        expression: str,
        factor_name: str,
        spec_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = dict(self.config["spec_defaults"])
        if spec_overrides:
            spec.update(spec_overrides)
        allowed_fields = self._get_expression_allowed_fields()
        check = validate_expression_against_registry(
            expression,
            allowed_fields=allowed_fields,
        )
        if not check["ok"]:
            raise ValueError(
                f"invalid expression for {factor_name}: "
                f"operators={check['unsupported_operators']} fields={check['unsupported_fields']}"
            )
        domain_coverage = validate_multi_domain_field_coverage(check["used_fields"], resolve_factor_domains())
        if not domain_coverage["ok"]:
            raise ValueError(
                f"invalid expression for {factor_name}: missing active domains={domain_coverage['missing_domains']} "
                f"used_fields={domain_coverage['used_fields']}"
            )
        spec["factor_name"] = factor_name
        spec["formula"] = expression
        spec["data_needed"] = sorted(check["used_fields"])
        spec["factor_needed"] = []
        spec["type"] = "regular"
        return spec

    def validate_factor_file(self, file_path: str | Path) -> Path:
        return TQStandaloneEvaluator.validate_factor_module(file_path)

    def render_factor_module(
        self,
        expression: str,
        factor_name: str,
        output_dir: str | Path | None = None,
        spec_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        out_dir = self.candidate_dir if output_dir is None else self._resolve_path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = self.build_spec(expression, factor_name, spec_overrides)
        file_path = out_dir / f"{factor_name}.py"
        file_path.write_text(TQStandaloneEvaluator.render_factor_module_text(spec), encoding="utf-8")
        self.validate_factor_file(file_path)
        return {
            "factor_name": factor_name,
            "factor_path": str(file_path),
            "spec": spec,
        }

    def export_task(
        self,
        task,
        output_dir: str | Path | None = None,
        spec_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        expression = getattr(task, "factor_expression", "")
        factor_name = getattr(task, "factor_name", getattr(task, "name", "unknown_factor"))
        if not expression:
            raise ValueError(f"factor '{factor_name}' has empty factor_expression")
        return self.render_factor_module(expression, factor_name, output_dir=output_dir, spec_overrides=spec_overrides)

    def export_experiment(
        self,
        experiment,
        output_dir: str | Path | None = None,
        spec_overrides: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        exports = []
        for task in getattr(experiment, "sub_tasks", []) or []:
            exports.append(self.export_task(task, output_dir=output_dir, spec_overrides=spec_overrides))
        return exports

    def _load_local_factor_module(self, file_path: str | Path) -> dict[str, Any]:
        module_globals: dict[str, Any] = {}
        path = Path(file_path)
        with _factor_module_import_context():
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module_globals, module_globals)
        return module_globals

    def evaluate(
        self,
        file_path: str | Path,
        profile_id: str | None = None,
        params: dict[str, Any] | None = None,
        prod_corr: bool = False,
        plot: bool = False,
        transform_spec: dict[str, Any] | None = None,
        silent: bool = True,
        context: str = "standalone",
        execution_constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del prod_corr, plot, transform_spec, silent
        runtime_check = self.check_runtime_fields(context=context)
        if not runtime_check["ok"]:
            raise ValueError(f"TQ runtime field check failed: {runtime_check['reason']} -> {runtime_check['missing_required']}")

        module = self._load_local_factor_module(file_path)
        factor_name = module["META"]["factor_name"]
        factor_value = self._get_evaluator().evaluate_factor_module(
            module,
            field_aliases=self.get_factor_field_aliases(context),
        )

        return self._get_evaluator().evaluate_factor_value(
            factor_name=factor_name,
            factor_value=factor_value,
            profile_id=profile_id or self.config["profile_id"],
            params=params or self.get_eval_params(context),
            execution_constraints=execution_constraints or self.get_execution_constraints(context),
        )

    def check_submission(self, file_path: str | Path) -> pd.DataFrame:
        runtime_check = self.check_runtime_fields(context="standalone")
        if not runtime_check["ok"]:
            raise ValueError(f"TQ runtime field check failed: {runtime_check['reason']} -> {runtime_check['missing_required']}")
        return self.evaluate_submission_checks(file_path)["summary"]

    def evaluate_submission_checks(self, file_path: str | Path) -> dict[str, Any]:
        runtime_check = self.check_runtime_fields(context="standalone")
        if not runtime_check["ok"]:
            raise ValueError(f"TQ runtime field check failed: {runtime_check['reason']} -> {runtime_check['missing_required']}")
        module = self._load_local_factor_module(file_path)
        factor_name = module["META"]["factor_name"]
        factor_value = self._get_evaluator().evaluate_factor_module(
            module,
            field_aliases=self.get_factor_field_aliases("standalone"),
        )
        return self.evaluate_submission_checks_for_value(
            factor_name=factor_name,
            factor_value=factor_value,
            profile_id=self.config["profile_id"],
            params=self.get_eval_params("standalone"),
            context="standalone",
            execution_constraints=self.get_execution_constraints("standalone"),
        )

    def submit(self, file_path: str | Path) -> dict[str, Any]:
        summary = self.check_submission(file_path)
        factor_name = str(summary.index[0]) if not summary.empty else Path(file_path).stem
        return {
            "status": "local_only",
            "factor_name": factor_name,
            "factor_path": str(file_path),
            "message": "Standalone QuantaAlpha mode does not submit to external TQ repositories.",
        }

    def _compute_submission_summary(self, file_path: str, profile_id: str) -> pd.DataFrame:
        del profile_id
        return self.evaluate_submission_checks(file_path)["summary"]

