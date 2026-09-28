"""
Factor library manager: save experiment output to unified JSON factor library.
Called from quantaalpha/pipeline/loop.py feedback step.
"""

import hashlib
import ast
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from filelock import FileLock

from quantaalpha.factors.data_domains import canonicalize_joint_pv_daily_aliases, resolve_factor_domains

logger = logging.getLogger(__name__)

DEFAULT_FACTOR_CACHE_DIR = os.environ.get(
    "FACTOR_CACHE_DIR",
    "data/results/factor_cache",
)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def resolve_factor_cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    return Path(cache_dir or DEFAULT_FACTOR_CACHE_DIR)


def factor_expression_md5(factor_expression: str) -> str:
    return hashlib.md5(str(factor_expression).encode()).hexdigest()


def get_factor_cache_path(factor_expression: str, cache_dir: str | os.PathLike[str] | None = None) -> Path:
    return resolve_factor_cache_dir(cache_dir) / f"{factor_expression_md5(factor_expression)}.pkl"


def load_factor_value_from_cache(
    factor_expression: str,
    cache_dir: str | os.PathLike[str] | None = None,
) -> pd.DataFrame | None:
    expr = str(factor_expression or "").strip()
    if not expr:
        return None
    cache_path = get_factor_cache_path(expr, cache_dir=cache_dir)
    if not cache_path.exists():
        return None
    try:
        cached = pd.read_pickle(cache_path)
    except Exception as exc:
        logger.warning("Failed to read factor cache %s: %s", cache_path, exc)
        return None
    if isinstance(cached, pd.Series):
        return cached.to_frame(name="factor")
    if isinstance(cached, pd.DataFrame):
        return cached
    logger.warning("Unsupported cached factor value type at %s: %s", cache_path, type(cached))
    return None


def save_factor_value_to_cache(
    factor_expression: str,
    factor_value: pd.DataFrame | pd.Series,
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path | None:
    expr = str(factor_expression or "").strip()
    if not expr or factor_value is None:
        return None
    cache_path = get_factor_cache_path(expr, cache_dir=cache_dir)
    if isinstance(factor_value, pd.Series):
        factor_value = factor_value.to_frame(name=factor_value.name or "factor")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        factor_value.to_pickle(cache_path)
        return cache_path
    except Exception as exc:
        logger.warning("Failed to save factor cache %s: %s", cache_path, exc)
        return None


def _extract_factor_level_from_code(code_text: str) -> str | None:
    text = str(code_text or "").strip()
    if not text:
        return None
    try:
        tree = ast.parse(text)
    except Exception:
        if "prepare_minute_datas" in text:
            return "minutes"
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id != "META":
            continue
        try:
            meta = ast.literal_eval(node.value)
        except Exception:
            meta = None
        if isinstance(meta, dict):
            level = str(meta.get("level") or "").strip().lower()
            if level in {"days", "minutes"}:
                return level
    if any(
        isinstance(node, ast.FunctionDef) and node.name == "prepare_minute_datas"
        for node in tree.body
    ):
        return "minutes"
    return None


def _extract_factor_module_text(code_dict: dict[str, str] | None) -> str:
    if not isinstance(code_dict, dict):
        return ""
    for preferred in ("factor.py", "minute.py", "regular.py", "super.py"):
        value = code_dict.get(preferred)
        if isinstance(value, str) and value.strip():
            return value
    for value in code_dict.values():
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _load_factor_module_text_from_manifest_entry(manifest_entry: dict[str, Any] | None) -> str:
    if not isinstance(manifest_entry, dict):
        return ""
    factor_path = str(manifest_entry.get("factor_path") or "").strip()
    if not factor_path:
        return ""
    path = Path(factor_path)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to read generated factor module %s: %s", path, exc)
        return ""


def _is_valid_python_module_text(module_text: str) -> bool:
    text = str(module_text or "").strip()
    if not text:
        return False
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


def _prefer_factor_module_text(
    code_module_text: str,
    manifest_module_text: str,
) -> str:
    code_text = str(code_module_text or "").strip()
    manifest_text = str(manifest_module_text or "").strip()
    if not code_text:
        return manifest_text
    if not manifest_text:
        return code_text

    code_level = _extract_factor_level_from_code(code_text)
    manifest_level = _extract_factor_level_from_code(manifest_text)

    # Minute factors are ultimately evaluated from the generated candidate module.
    # Prefer that artifact whenever it parses, or when the draft code is invalid.
    if manifest_level == "minutes":
        if _is_valid_python_module_text(manifest_text):
            return manifest_text
        if not _is_valid_python_module_text(code_text):
            return manifest_text
    return code_text


class FactorLibraryManager:
    """Manage unified factor library (CRUD + TQ upstream export metadata)."""

    # Compact mode keeps standalone-backtest-critical fields and prunes optional heavy payloads.
    COMPACT_MODE = _env_flag("FACTOR_LIBRARY_COMPACT", True)
    INCLUDE_CODE = _env_flag("FACTOR_LIBRARY_INCLUDE_CODE", False)
    INCLUDE_TQ_UPSTREAM = _env_flag("FACTOR_LIBRARY_INCLUDE_TQ_UPSTREAM", False)
    KEEP_RESULT_H5 = _env_flag("FACTOR_LIBRARY_KEEP_RESULT_H5", False)

    def __init__(self, library_path: str):
        self.library_path = Path(library_path)
        self.data = self._load()
        self._tq_bridge = None
        self._tq_bridge_error: str | None = None

    @property
    def _lock_path(self) -> Path:
        return self.library_path.with_name(f"{self.library_path.name}.lock")

    def _load(self) -> dict:
        if self.library_path.exists():
            try:
                with open(self.library_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Factor library file corrupted, recreating: {e}")
        return {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_factors": 0,
                "version": "1.0",
            },
            "factors": {},
        }

    def _save(self):
        self.data["metadata"]["last_updated"] = datetime.now().isoformat()
        self.data["metadata"]["total_factors"] = len(self.data["factors"])
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.library_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _cleanup_result_h5(h5_path: str) -> bool:
        try:
            Path(h5_path).unlink(missing_ok=True)
            return True
        except Exception as exc:
            logger.warning("Failed to remove result.h5 %s: %s", h5_path, exc)
            return False

    def _get_tq_bridge(self):
        if self._tq_bridge_error is not None:
            return None
        if self._tq_bridge is None:
            try:
                from quantaalpha.backtest import TQUpstreamBridge

                self._tq_bridge = TQUpstreamBridge(os.getenv("TQ_UPSTREAM_CONFIG_PATH"))
            except Exception as exc:
                self._tq_bridge_error = str(exc)
                logger.warning("TQ upstream bridge unavailable, skip TQ export: %s", exc)
                return None
        return self._tq_bridge

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): FactorLibraryManager._jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [FactorLibraryManager._jsonable(item) for item in value]
        if isinstance(value, pd.Series):
            return {str(key): FactorLibraryManager._jsonable(val) for key, val in value.items()}
        if isinstance(value, pd.DataFrame):
            return FactorLibraryManager._jsonable(value.to_dict(orient="index"))
        if isinstance(value, np.ndarray):
            return [FactorLibraryManager._jsonable(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return None
            return float(value)
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    @staticmethod
    def _is_empty(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0
        return False

    @classmethod
    def _prune_empty(cls, value: Any) -> Any:
        if isinstance(value, dict):
            pruned = {}
            for k, v in value.items():
                vv = cls._prune_empty(v)
                if cls._is_empty(vv):
                    continue
                pruned[k] = vv
            return pruned
        if isinstance(value, list):
            pruned = [cls._prune_empty(v) for v in value]
            return [v for v in pruned if not cls._is_empty(v)]
        return value

    @classmethod
    def _compact_run_context(cls, run_context: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(run_context, dict):
            return {}
        keep_keys = {
            "backtest_engine",
            "backtest_mode",
            "profile_id",
            "profile_mode",
            "metric_mode",
            "use_net_metrics",
            "data_mode",
            "prompt_mode",
            "data_domains",
            "discovery_split",
            "train_splits",
            "primary_split",
            "test_split",
            "stage_splits",
            "mining_periods",
        }
        compact = {k: run_context.get(k) for k in keep_keys if k in run_context}
        params = run_context.get("params")
        if isinstance(params, dict) and params:
            compact["params"] = params
        return cls._prune_empty(compact)

    @classmethod
    def _compact_tq_upstream(cls, tq_upstream: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(tq_upstream, dict):
            return {}
        keep_keys = {
            "status",
            "factor_name",
            "factor_path",
            "config_path",
            "profile_id",
            "metric_mode",
            "use_net_metrics",
            "exported_at",
            "error",
            "reason",
        }
        compact = {k: tq_upstream.get(k) for k in keep_keys if k in tq_upstream}
        return cls._prune_empty(compact)

    @staticmethod
    def _get_required_tq_fields(bridge) -> list[str]:
        try:
            from quantaalpha.backtest.config import (
                EXECUTION_CONSTRAINT_FIELD_REQUIREMENTS,
                REQUIRED_BACKTEST_FIELDS,
                TRANSFORM_FIELD_REQUIREMENTS,
            )

            required_fields = set(REQUIRED_BACKTEST_FIELDS)
            execution_constraints = bridge.get_execution_constraints("standalone")
            for flag, fields in EXECUTION_CONSTRAINT_FIELD_REQUIREMENTS.items():
                if execution_constraints.get(flag):
                    required_fields.update(fields)
            transform_spec = bridge.config.get("transform_spec") or {}
            for field in transform_spec.get("subuniverse", []):
                required_fields.update(TRANSFORM_FIELD_REQUIREMENTS["subuniverse"].get(field, set()))
            for field in transform_spec.get("neutralize", []):
                required_fields.update(TRANSFORM_FIELD_REQUIREMENTS["neutralize"].get(field, set()))
            return sorted(required_fields)
        except Exception:
            return []

    @staticmethod
    def _index_tq_manifest(manifest: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for item in manifest or []:
            if not isinstance(item, dict):
                continue
            factor_name = str(item.get("factor_name") or "").strip()
            if factor_name:
                indexed[factor_name] = item
        return indexed

    def _build_tq_upstream_entry(
        self,
        factor_name: str,
        factor_expression: str,
        task: Any = None,
        manifest_entry: Optional[dict[str, Any]] = None,
        factor_module_text: str = "",
        output_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        exported_at = datetime.now().isoformat()
        module_text = str(factor_module_text or "").strip()
        module_level = _extract_factor_level_from_code(module_text)
        if not factor_expression and not module_text:
            return {
                "status": "skipped",
                "factor_name": factor_name,
                "reason": "empty factor_expression and factor_module_text",
                "exported_at": exported_at,
            }

        bridge = self._get_tq_bridge()
        if bridge is None:
            return {
                "status": "unavailable",
                "factor_name": factor_name,
                "reason": self._tq_bridge_error or "TQ upstream bridge initialization failed",
                "exported_at": exported_at,
            }

        required_fields = self._get_required_tq_fields(bridge)
        runtime_check = self._jsonable(bridge.check_runtime_fields())

        try:
            if module_text and module_level == "minutes":
                out_dir = bridge.candidate_dir if output_dir is None else bridge._resolve_path(output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                file_path = out_dir / f"{factor_name}.py"
                file_path.write_text(module_text, encoding="utf-8")
                bridge.validate_factor_file(file_path)
                exported = {
                    "factor_name": factor_name,
                    "factor_path": str(file_path),
                    "spec": {
                        "factor_name": factor_name,
                        "level": module_level,
                    },
                }
            elif task is not None:
                exported = bridge.export_task(task, output_dir=output_dir)
            else:
                exported = bridge.render_factor_module(
                    expression=factor_expression,
                    factor_name=factor_name,
                    output_dir=output_dir,
                )

            factor_path = str(exported["factor_path"])
            entry = {
                "status": "exported",
                "factor_name": factor_name,
                "factor_path": factor_path,
                "candidate_dir": str(Path(factor_path).parent),
                "config_path": bridge.config.get("_config_path"),
                "profile_id": bridge.config.get("profile_id"),
                "params": self._jsonable(dict(bridge.config.get("params") or {})),
                "transform_spec": self._jsonable(dict(bridge.config.get("transform_spec") or {})),
                "metric_mode": bridge.config.get("metric_mode", "long_only"),
                "use_net_metrics": bool(bridge.config.get("use_net_metrics", False)),
                "submit_after_check": bool(bridge.config.get("submit_after_check", False)),
                "required_runtime_fields": required_fields,
                "runtime_check": runtime_check,
                "spec": self._jsonable(exported.get("spec") or {}),
                "exported_at": exported_at,
            }

            if manifest_entry:
                runner_manifest = {}
                runner_factor_path = manifest_entry.get("factor_path")
                if runner_factor_path and str(runner_factor_path) != factor_path:
                    runner_manifest["factor_path"] = str(runner_factor_path)
                if manifest_entry.get("check_summary") is not None:
                    runner_manifest["check_summary"] = self._jsonable(manifest_entry.get("check_summary"))
                if manifest_entry.get("submit_result") is not None:
                    runner_manifest["submit_result"] = self._jsonable(manifest_entry.get("submit_result"))
                if runner_manifest:
                    entry["runner_manifest"] = runner_manifest

            return entry
        except Exception as exc:
            logger.warning("Failed to export factor %s to TQ candidate module: %s", factor_name, exc)
            return {
                "status": "error",
                "factor_name": factor_name,
                "config_path": bridge.config.get("_config_path"),
                "required_runtime_fields": required_fields,
                "runtime_check": runtime_check,
                "exported_at": exported_at,
                "error": str(exc),
            }

    def add_factors_from_experiment(
        self,
        experiment,
        experiment_id: str = "unknown",
        round_number: int = 0,
        hypothesis: Optional[str] = None,
        feedback: Any = None,
        initial_direction: Optional[str] = None,
        user_initial_direction: Optional[str] = None,
        planning_direction: Optional[str] = None,
        evolution_phase: str = "original",
        trajectory_id: str = "",
        parent_trajectory_ids: Optional[list] = None,
    ):
        """Extract factors from a factor mining experiment and write to library."""
        if experiment is None:
            logger.warning("experiment is None, skip saving factors")
            return
        lock = FileLock(str(self._lock_path))
        with lock:
            self.data = self._load()
            backtest_results = self._extract_backtest_results(experiment)
            library_check_flags = self._extract_library_check_flags(experiment)
            feedback_dict = self._extract_feedback(feedback)
            sub_tasks = getattr(experiment, "sub_tasks", []) or []
            sub_workspaces = getattr(experiment, "sub_workspace_list", []) or []
            tq_manifest_by_name = self._index_tq_manifest(getattr(experiment, "tq_upstream_manifest", None))
            submission_checks = getattr(experiment, "tq_submission_checks", None)
            if not isinstance(submission_checks, dict):
                submission_checks = {}
            per_factor_backtest_results = bool(backtest_results) and any(
                isinstance(value, dict) for value in backtest_results.values()
            )
            shared_backtest_results = bool(backtest_results) and not any(
                isinstance(value, dict) for value in backtest_results.values()
            )
            saved_factor_count = 0

            for idx, task in enumerate(sub_tasks):
                factor_name = getattr(task, "factor_name", getattr(task, "name", f"factor_{idx}"))

                current_results = {}
                if factor_name in backtest_results:
                    val = backtest_results[factor_name]
                    if isinstance(val, dict):
                        current_results = val
                elif per_factor_backtest_results:
                    logger.warning(
                        f"Skip factor {factor_name} when saving library because single-factor backtest did not produce valid results."
                    )
                    continue
                elif shared_backtest_results:
                    current_results = backtest_results
                elif len(sub_tasks) == 1:
                    current_results = backtest_results

                factor_expr = canonicalize_joint_pv_daily_aliases(getattr(task, "factor_expression", ""))
                factor_desc = getattr(task, "factor_description", getattr(task, "description", ""))
                factor_form = getattr(task, "factor_formulation", "")

                factor_id = hashlib.md5(
                    f"{factor_name}_{factor_expr}".encode()
                ).hexdigest()[:16]

                code = ""
                factor_module_text = ""
                cache_location = {}
                result_h5_path = ""
                if idx < len(sub_workspaces):
                    ws = sub_workspaces[idx]
                    code_dict = getattr(ws, "code_dict", {})
                    factor_module_text = _extract_factor_module_text(code_dict)
                    if self.INCLUDE_CODE:
                        code = "\n".join(
                            f"File: {fname}\n\n{content}"
                            for fname, content in code_dict.items()
                        )
                    ws_path = getattr(ws, "workspace_path", None)
                    if ws_path:
                        ws_path = Path(ws_path)
                        h5_file = ws_path / "result.h5"
                        if h5_file.exists():
                            result_h5_path = str(h5_file)
                        else:
                            logger.warning(
                                f"result.h5 missing for {factor_name} ({h5_file}), will recompute from expression in backtest"
                            )
                manifest_module_text = _load_factor_module_text_from_manifest_entry(
                    tq_manifest_by_name.get(factor_name)
                )
                factor_module_text = _prefer_factor_module_text(
                    factor_module_text,
                    manifest_module_text,
                )
                if factor_module_text:
                    factor_module_text = canonicalize_joint_pv_daily_aliases(factor_module_text)
                if code:
                    code = canonicalize_joint_pv_daily_aliases(code)

                created_at = datetime.now().isoformat()
                tq_upstream = self._build_tq_upstream_entry(
                    factor_name=factor_name,
                    factor_expression=factor_expr,
                    task=task,
                    manifest_entry=tq_manifest_by_name.get(factor_name),
                    factor_module_text=factor_module_text,
                )

                run_context = self._extract_run_context(experiment)
                compact_check_flags = dict(library_check_flags.get(factor_name) or {})
                if not compact_check_flags:
                    check_info = submission_checks.get(factor_name) or {}
                    train_check_flags = {
                        "check_passed": bool(check_info.get("check_passed")),
                        "raw_passed": bool(check_info.get("raw_passed")),
                        "zz1000s_passed": bool(check_info.get("zz1000s_passed")),
                        "complete_passed": bool(check_info.get("complete_passed")),
                        "check_tag": str(check_info.get("tag_suggestion") or ""),
                    }
                    compact_check_flags = {
                        "check_passed": bool(check_info.get("check_passed")),
                        "train_passed": bool(check_info.get("train_passed", check_info.get("check_passed"))),
                        "test_passed": False,
                        "train_check_flags": train_check_flags,
                        "test_check_flags": {},
                    }
                check_passed = bool(compact_check_flags.get("check_passed"))
                train_passed = bool(compact_check_flags.get("train_passed"))
                test_passed = bool(compact_check_flags.get("test_passed"))
                train_check_flags = dict(compact_check_flags.get("train_check_flags") or {})
                test_check_flags = dict(compact_check_flags.get("test_check_flags") or {})
                factor_level = (
                    _extract_factor_level_from_code(factor_module_text)
                    or ("minutes" if "minutes" in (run_context.get("data_domains") or []) else None)
                    or ("minutes" if str(run_context.get("data_mode") or "").strip().lower() == "minutes" else None)
                    or "days"
                )

                factor_entry = {
                    "factor_id": factor_id,
                    "factor_name": factor_name,
                    "factor_level": factor_level,
                    "factor_expression": factor_expr,
                    "factor_description": factor_desc,
                    "factor_formulation": factor_form,
                    "added_at": created_at,
                    "metadata": {
                        "experiment_id": experiment_id,
                        "round_number": round_number,
                        "evolution_phase": evolution_phase,
                        "trajectory_id": trajectory_id,
                        "parent_trajectory_ids": parent_trajectory_ids or [],
                        "hypothesis": str(hypothesis) if hypothesis else "",
                        "initial_direction": initial_direction or "",
                        "user_initial_direction": user_initial_direction or "",
                        "planning_direction": planning_direction or "",
                        "created_at": created_at,
                    },
                    "test_backtest_metrics": current_results,
                    "train_check_flags": train_check_flags,
                    "test_check_flags": test_check_flags,
                    "feedback": feedback_dict,
                    "run_context": run_context,
                    "check_passed": check_passed,
                    "train_passed": train_passed,
                    "test_passed": test_passed,
                }

                if factor_expr and result_h5_path:
                    sync_success = self._sync_h5_to_md5_cache(factor_expr, result_h5_path)
                    if self.KEEP_RESULT_H5 or not sync_success:
                        cache_location["result_h5_path"] = result_h5_path
                    else:
                        self._cleanup_result_h5(result_h5_path)

                if code:
                    factor_entry["factor_implementation_code"] = code
                if factor_module_text:
                    factor_entry["factor_module_text"] = factor_module_text
                if cache_location:
                    factor_entry["cache_location"] = cache_location
                if self.INCLUDE_TQ_UPSTREAM:
                    factor_entry["tq_upstream"] = tq_upstream

                if self.COMPACT_MODE:
                    factor_entry["run_context"] = self._compact_run_context(factor_entry.get("run_context") or {})
                    if "tq_upstream" in factor_entry:
                        factor_entry["tq_upstream"] = self._compact_tq_upstream(factor_entry.get("tq_upstream") or {})
                    factor_entry = self._prune_empty(factor_entry)

                self.data["factors"][factor_id] = factor_entry
                saved_factor_count += 1

            self._save()
        logger.info(
            f"Saved {saved_factor_count} factors to {self.library_path} (backtest_results: {len(backtest_results)} metrics)"
        )

    def sync_tq_exports(
        self,
        factor_ids: Optional[list[str]] = None,
        output_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        selected = None
        if factor_ids:
            selected = {str(item).strip() for item in factor_ids if str(item).strip()}

        exported = 0
        skipped = 0
        failed = 0
        details = []

        lock = FileLock(str(self._lock_path))
        with lock:
            self.data = self._load()
            for factor_id, factor_info in self.data.get("factors", {}).items():
                factor_name = str(factor_info.get("factor_name") or factor_id)
                if selected and factor_id not in selected and factor_name not in selected:
                    continue

                factor_expr = str(factor_info.get("factor_expression") or "")
                entry = self._build_tq_upstream_entry(
                    factor_name=factor_name,
                    factor_expression=factor_expr,
                    output_dir=output_dir,
                )
                factor_info["tq_upstream"] = entry

                status = str(entry.get("status") or "unknown")
                if status == "exported":
                    exported += 1
                elif status in {"skipped", "unavailable"}:
                    skipped += 1
                else:
                    failed += 1

                details.append(
                    {
                        "factor_id": factor_id,
                        "factor_name": factor_name,
                        "status": status,
                        "factor_path": entry.get("factor_path"),
                        "error": entry.get("error") or entry.get("reason"),
                    }
                )

            self._save()
        return {
            "library_path": str(self.library_path),
            "selected": len(details),
            "exported": exported,
            "skipped": skipped,
            "failed": failed,
            "details": details,
        }

    @classmethod
    def export_library_to_tq_candidates(
        cls,
        library_path: str,
        factor_ids: Optional[list[str]] = None,
        output_dir: Optional[str] = None,
    ) -> dict[str, Any]:
        manager = cls(library_path)
        return manager.sync_tq_exports(factor_ids=factor_ids, output_dir=output_dir)

    @staticmethod
    def _sync_h5_to_md5_cache(factor_expression: str, h5_path: str,
                                cache_dir: Optional[str] = None) -> bool:
        """Sync factor values from result.h5 to MD5 cache dir (.pkl). Returns True on success."""
        cache_dir = resolve_factor_cache_dir(cache_dir)
        h5_file = Path(h5_path)

        if not h5_file.exists():
            return False

        pkl_file = get_factor_cache_path(factor_expression, cache_dir=cache_dir)

        if pkl_file.exists():
            return True

        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            result = pd.read_hdf(str(h5_file))
            result.to_pickle(pkl_file)
            logger.debug(f"Synced factor cache -> {pkl_file.name}")
            return True
        except Exception as e:
            logger.debug(f"Sync factor cache failed [{h5_path}]: {e}")
            return False

    @staticmethod
    def check_cache_status(library_path: str,
                           cache_dir: Optional[str] = None) -> dict:
        """Check cache status for each factor in library. Returns:
            {
                "total": int,
                "h5_cached": int,
                "md5_cached": int,
                "need_compute": int,
                "factors": [ { "factor_id", "factor_name", "status" }, ... ]
            }
        """
        cache_dir = Path(cache_dir or DEFAULT_FACTOR_CACHE_DIR)

        with open(library_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        factors = data.get("factors", {})
        total = len(factors)
        h5_cached = 0
        md5_cached = 0
        need_compute = 0
        details = []

        for fid, finfo in factors.items():
            expr = finfo.get("factor_expression", "")
            cloc = finfo.get("cache_location", {})
            h5_path = cloc.get("result_h5_path", "")

            status = "need_compute"
            if h5_path and Path(h5_path).exists():
                status = "h5_cached"
                h5_cached += 1
            elif expr:
                md5_key = hashlib.md5(expr.encode()).hexdigest()
                if (cache_dir / f"{md5_key}.pkl").exists():
                    status = "md5_cached"
                    md5_cached += 1

            if status == "need_compute":
                need_compute += 1

            details.append({
                "factor_id": fid,
                "factor_name": finfo.get("factor_name", fid),
                "status": status,
            })

        return {
            "total": total,
            "h5_cached": h5_cached,
            "md5_cached": md5_cached,
            "need_compute": need_compute,
            "factors": details,
        }

    @staticmethod
    def warm_cache_from_json(library_path: str,
                             cache_dir: Optional[str] = None) -> dict:
        """Walk factor library JSON and sync all available result.h5 to MD5 cache dir. Returns:
            { "total": int, "synced": int, "skipped": int, "failed": int,
              "already_cached": int, "no_source": int }
        """
        cache_dir_path = Path(cache_dir or DEFAULT_FACTOR_CACHE_DIR)

        with open(library_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        factors = data.get("factors", {})
        synced = 0
        skipped = 0
        failed = 0
        already_cached = 0
        no_source = 0

        for fid, finfo in factors.items():
            expr = finfo.get("factor_expression", "")
            cloc = finfo.get("cache_location", {})
            h5_path = cloc.get("result_h5_path", "")

            if not expr or not h5_path:
                no_source += 1
                skipped += 1
                continue

            md5_key = hashlib.md5(expr.encode()).hexdigest()
            pkl_file = cache_dir_path / f"{md5_key}.pkl"

            if pkl_file.exists():
                already_cached += 1
                skipped += 1
                continue

            if not Path(h5_path).exists():
                failed += 1
                continue

            try:
                cache_dir_path.mkdir(parents=True, exist_ok=True)
                result = pd.read_hdf(str(h5_path))
                result.to_pickle(pkl_file)
                synced += 1
            except Exception:
                failed += 1

        return {
            "total": len(factors),
            "synced": synced,
            "skipped": skipped,
            "failed": failed,
            "already_cached": already_cached,
            "no_source": no_source,
        }

    @staticmethod
    def _extract_selection_stage_results(experiment) -> dict[str, dict]:
        stage_payloads = getattr(experiment, "tq_stage_backtest_payloads", None)
        if isinstance(stage_payloads, dict) and stage_payloads:
            run_context = getattr(experiment, "tq_runtime_context", None)
            primary_split = ""
            if isinstance(run_context, dict):
                primary_split = str(run_context.get("primary_split") or "").strip()

            selected_results: dict[str, dict] = {}
            for factor_name, factor_stage_payload in stage_payloads.items():
                if not isinstance(factor_stage_payload, dict):
                    continue
                primary_train_payload = factor_stage_payload.get("primary_train")
                if not isinstance(primary_train_payload, dict):
                    continue

                entry: dict[str, Any] = {}
                perf = primary_train_payload.get("factor_performance")
                if isinstance(perf, dict) and perf:
                    entry = FactorLibraryManager._jsonable(perf)

                summary_df = primary_train_payload.get("summary")
                if not entry and isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
                    try:
                        factor_key = str(factor_name)
                        if factor_key in summary_df.index:
                            entry = FactorLibraryManager._jsonable(summary_df.loc[factor_key].to_dict())
                        elif factor_key in summary_df.columns:
                            entry = FactorLibraryManager._jsonable(summary_df[factor_key].to_dict())
                        else:
                            entry = FactorLibraryManager._jsonable(summary_df.iloc[0].to_dict())
                    except Exception:
                        entry = {}
                if not entry:
                    continue

                if primary_split:
                    entry["tq_primary_split"] = primary_split
                selected_results[str(factor_name)] = entry

            if selected_results:
                return selected_results

        stage_results = getattr(experiment, "tq_stage_backtest_results", None)
        if not isinstance(stage_results, dict) or not stage_results:
            return {}

        run_context = getattr(experiment, "tq_runtime_context", None)
        primary_split = ""
        if isinstance(run_context, dict):
            primary_split = str(run_context.get("primary_split") or "").strip()

        fallback_stage_name = primary_split
        selected_results: dict[str, dict] = {}
        for factor_name, factor_stage_payload in stage_results.items():
            if not isinstance(factor_stage_payload, dict):
                continue
            primary_payload = factor_stage_payload.get(fallback_stage_name)
            if not isinstance(primary_payload, dict):
                continue
            metrics = primary_payload.get("metrics")
            if not isinstance(metrics, dict) or not metrics:
                continue

            entry = dict(metrics)
            split_name = str(primary_payload.get("split") or primary_split or "").strip()
            if split_name:
                entry["tq_primary_split"] = split_name
            selected_results[str(factor_name)] = entry

        return selected_results

    @staticmethod
    def _extract_backtest_results(experiment) -> dict:
        """Extract compact per-factor library backtest metrics."""
        compact_metrics = getattr(experiment, "tq_library_backtest_metrics", None)
        if isinstance(compact_metrics, dict) and compact_metrics:
            return FactorLibraryManager._jsonable(compact_metrics)

        selection_stage = FactorLibraryManager._extract_selection_stage_results(experiment)
        if selection_stage:
            return FactorLibraryManager._jsonable(selection_stage)

        structured = getattr(experiment, "tq_structured_backtest_results", None)
        if isinstance(structured, dict) and structured:
            return FactorLibraryManager._jsonable(structured)

        result = getattr(experiment, "result", None)
        if result is None:
            return {}
        if isinstance(result, pd.Series):
            return FactorLibraryManager._jsonable(result)

        if isinstance(result, pd.DataFrame):
            try:
                return FactorLibraryManager._jsonable(result.iloc[:, 0].to_dict())
            except Exception:
                pass

        if isinstance(result, dict):
            out = {}
            for k, v in result.items():
                if isinstance(v, pd.Series):
                    out[str(k)] = FactorLibraryManager._jsonable(v)
                elif isinstance(v, pd.DataFrame):
                    try:
                        out[str(k)] = FactorLibraryManager._jsonable(v.iloc[:, 0].to_dict())
                    except Exception:
                        out[str(k)] = str(v)
                else:
                    out[str(k)] = FactorLibraryManager._jsonable(v)
            return out

        return {}

    @staticmethod
    def _extract_library_check_flags(experiment) -> dict:
        compact_flags = getattr(experiment, "tq_library_check_flags", None)
        if isinstance(compact_flags, dict) and compact_flags:
            return FactorLibraryManager._jsonable(compact_flags)

        submission_checks = getattr(experiment, "tq_submission_checks", None)
        if not isinstance(submission_checks, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for factor_name, payload in submission_checks.items():
            if not isinstance(payload, dict):
                continue
            train_check_flags = {
                "check_passed": bool(payload.get("check_passed")),
                "raw_passed": bool(payload.get("raw_passed")),
                "zz1000s_passed": bool(payload.get("zz1000s_passed")),
                "complete_passed": bool(payload.get("complete_passed")),
                "check_tag": str(payload.get("tag_suggestion") or ""),
            }
            out[str(factor_name)] = {
                "check_passed": bool(payload.get("check_passed")),
                "train_passed": bool(payload.get("train_passed", payload.get("check_passed"))),
                "test_passed": False,
                "train_check_flags": train_check_flags,
                "test_check_flags": {},
            }
        return out

    @staticmethod
    def _extract_feedback(feedback) -> dict:
        """Convert feedback object to serializable dict."""
        if feedback is None:
            return {}
        if isinstance(feedback, dict):
            return feedback

        out = {}
        for attr in ["observations", "hypothesis_evaluation", "decision", "reason",
                      "new_hypothesis", "feedback_str"]:
            val = getattr(feedback, attr, None)
            if val is not None:
                out[attr] = str(val) if not isinstance(val, (bool, int, float)) else val
        if not out:
            out["raw"] = str(feedback)
        return out

    @staticmethod
    def _extract_run_context(experiment) -> dict:
        runtime_context = getattr(experiment, "tq_runtime_context", None)
        if isinstance(runtime_context, dict):
            return FactorLibraryManager._jsonable(runtime_context)
        return {
            "backtest_engine": str(os.getenv("QUANTAALPHA_BACKTEST_ENGINE", "tq")).strip().lower(),
            "backtest_mode": str(os.getenv("FACTOR_BACKTEST_MODE", "single")).strip().lower(),
            "data_mode": str(os.getenv("FACTOR_DATA_MODE", "daily")).strip().lower(),
            "prompt_mode": str(os.getenv("FACTOR_PROMPT_MODE", "daily")).strip().lower(),
            "data_domains": list(resolve_factor_domains()),
        }



