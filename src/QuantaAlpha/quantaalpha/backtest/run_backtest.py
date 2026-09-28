from __future__ import annotations

import ast
import json
import os
import pickle
from pathlib import Path

import fire

from quantaalpha.backtest import FactorPlotter, FactorQualityAnalyzer, TQUpstreamBridge

from .notebook_report import build_factor_report_notebook
from quantaalpha.factors.library import (
    get_factor_cache_path,
    load_factor_value_from_cache,
    save_factor_value_to_cache,
)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict") and not isinstance(value, (str, bytes)):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    return value


def _to_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(value)


def _load_factor_from_library(library_path: str | Path, factor_ref: str) -> dict:
    path = Path(library_path)
    if not path.exists():
        raise FileNotFoundError(f"library_path does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    factors = payload.get("factors") or {}
    if factor_ref in factors:
        return factors[factor_ref]

    matches = []
    for factor_id, factor_info in factors.items():
        if str(factor_info.get("factor_name") or "").strip() == str(factor_ref).strip():
            entry = dict(factor_info)
            entry.setdefault("factor_id", factor_id)
            matches.append(entry)

    if not matches:
        raise KeyError(f"factor '{factor_ref}' not found in {path}")
    if len(matches) > 1:
        raise ValueError(f"factor '{factor_ref}' is ambiguous in {path}; please use factor_id.")
    return matches[0]


def _is_valid_python_module_text(module_text: str) -> bool:
    text = str(module_text or "").strip()
    if not text:
        return False
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


def _repair_common_module_syntax(module_text: str) -> str:
    text = str(module_text or "")
    if not text.strip():
        return text

    lines = text.splitlines(keepends=True)
    repaired_lines: list[str] = []
    inside_simple_dict = False

    for line in lines:
        if not inside_simple_dict and line.lstrip().startswith(("META = {", "SETTING = {")):
            inside_simple_dict = True
        elif inside_simple_dict and line.lstrip().startswith("}"):
            inside_simple_dict = False

        if inside_simple_dict and repaired_lines and line.lstrip().startswith(('"', "'")):
            previous = repaired_lines[-1]
            previous_body = previous.rstrip("\r\n")
            if previous_body.strip() and not previous_body.rstrip().endswith((",", "{")):
                newline = previous[len(previous_body) :]
                repaired_lines[-1] = previous_body + "," + newline

        repaired_lines.append(line)

    return "".join(repaired_lines)


def _resolve_library_module_text(factor_entry: dict) -> str:
    candidate_texts: list[str] = []

    module_text = str(factor_entry.get("factor_module_text") or "").strip()
    if module_text:
        candidate_texts.append(module_text)

    fallback_code = str(factor_entry.get("factor_implementation_code") or "").strip()
    if fallback_code and not fallback_code.startswith("File: "):
        candidate_texts.append(fallback_code)

    for candidate in candidate_texts:
        if _is_valid_python_module_text(candidate):
            return candidate
        repaired = _repair_common_module_syntax(candidate)
        if _is_valid_python_module_text(repaired):
            return repaired
    return ""


def _write_factor_module_from_library_entry(
    factor_entry: dict,
    factor_name: str,
    out_dir: Path,
) -> Path | None:
    module_text = _resolve_library_module_text(factor_entry)
    if not module_text:
        return None
    module_path = out_dir / f"{factor_name}.py"
    module_path.write_text(module_text, encoding="utf-8")
    return module_path


def run_single_factor_backtest(
    expression: str | None = None,
    factor_file: str | None = None,
    factor_name: str = "manual_factor",
    config_path: str | None = None,
    output_dir: str | None = None,
    plot: bool = False,
    show: bool = False,
    quality_report: bool = True,
    notebook_report: bool = False,
) -> dict[str, str | None]:
    plot = _to_bool(plot)
    show = _to_bool(show)
    quality_report = _to_bool(quality_report)
    notebook_report = _to_bool(notebook_report)

    bridge = TQUpstreamBridge(config_path)
    out_dir = Path(output_dir) if output_dir else bridge.candidate_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    standalone_params = bridge.get_eval_params("standalone")
    execution_constraints = bridge.get_execution_constraints("standalone")
    factor_path: Path | None = None
    factor_cache_path: Path | None = None
    factor_source = "file"

    if factor_file:
        factor_path = Path(factor_file)
        bridge.validate_factor_file(factor_path)
        result = bridge.evaluate(
            factor_path,
            params=standalone_params,
            context="standalone",
            execution_constraints=execution_constraints,
        )
    elif expression:
        expression = str(expression).strip()
        if not expression:
            raise ValueError("expression cannot be empty.")

        cached_factor_value = load_factor_value_from_cache(expression)
        factor_cache_path = get_factor_cache_path(expression)
        if cached_factor_value is not None:
            factor_source = "cache"
            result = bridge._get_evaluator().evaluate_factor_value(
                factor_name=factor_name,
                factor_value=cached_factor_value,
                profile_id=bridge.config["profile_id"],
                params=standalone_params,
                execution_constraints=execution_constraints,
            )
        else:
            factor_source = "expression"
            exported = bridge.render_factor_module(expression=expression, factor_name=factor_name, output_dir=out_dir)
            factor_path = Path(exported["factor_path"])
            factor_name = exported["factor_name"]
            result = bridge.evaluate(
                factor_path,
                params=standalone_params,
                context="standalone",
                execution_constraints=execution_constraints,
            )
            saved_cache_path = save_factor_value_to_cache(expression, result["factor_value"])
            if saved_cache_path is not None:
                factor_cache_path = saved_cache_path
    else:
        raise ValueError("Provide factor_file or expression.")

    summary = result["summary"].T
    summary.index.name = "factor"

    summary_path = out_dir / f"{factor_name}_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(orient="index"), ensure_ascii=False, indent=2), encoding="utf-8")
    result["factor_value"].to_pickle(out_dir / f"{factor_name}_factor_value.pkl")
    (out_dir / f"{factor_name}_factor_result.json").write_bytes(
        json.dumps(
            _jsonable(result["factor_result"]),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    )
    factor_result_pickle_path = out_dir / f"{factor_name}_factor_result.pkl"
    with factor_result_pickle_path.open("wb") as fh:
        pickle.dump(result["factor_result"], fh)

    quality_path = None
    if quality_report:
        analyzer = FactorQualityAnalyzer(bridge._get_runtime_data_provider(require_data=True))
        report = analyzer.build_report(factor_name, result["factor_value"], result["factor_result"])
        quality_path = out_dir / f"{factor_name}_quality.json"
        quality_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_path = None
    if plot:
        plot_path = out_dir / f"{factor_name}_diagnostics.png"
        plotter = FactorPlotter()
        plotter.plot_result(
            factor_name,
            result["factor_result"],
            params=standalone_params,
            output_path=plot_path,
            show=show,
        )

    notebook_path = None
    if notebook_report:
        notebook_path = out_dir / f"{factor_name}_report.ipynb"
        build_factor_report_notebook(
            factor_name=factor_name,
            summary=summary,
            output_path=notebook_path,
            summary_path=summary_path,
            factor_value_path=out_dir / f"{factor_name}_factor_value.pkl",
            factor_result_pickle_path=factor_result_pickle_path,
            factor_result_fields=sorted(str(key) for key in result["factor_result"].keys()),
            quality_path=quality_path,
            plot_path=plot_path,
        )

    return {
        "factor_name": factor_name,
        "factor_path": str(factor_path) if factor_path else None,
        "factor_source": factor_source,
        "factor_cache_path": str(factor_cache_path) if factor_cache_path else None,
        "summary_path": str(summary_path),
        "factor_result_pickle_path": str(factor_result_pickle_path),
        "quality_path": str(quality_path) if quality_path else None,
        "plot_path": str(plot_path) if plot_path else None,
        "notebook_path": str(notebook_path) if notebook_path else None,
    }


def run_library_factor_backtest(
    library_path: str | Path,
    library_factor: str,
    config_path: str | None = None,
    output_dir: str | None = None,
    plot: bool = False,
    show: bool = False,
    quality_report: bool = True,
    notebook_report: bool = False,
) -> dict[str, str | None]:
    factor_entry = _load_factor_from_library(library_path, library_factor)
    factor_name = str(factor_entry.get("factor_name") or library_factor).strip() or library_factor
    out_dir = Path(output_dir) if output_dir else None
    level = str(factor_entry.get("factor_level") or "days").strip().lower() or "days"

    module_path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        module_path = _write_factor_module_from_library_entry(factor_entry, factor_name, out_dir)
    else:
        bridge = TQUpstreamBridge(config_path)
        out_dir = bridge.candidate_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        module_path = _write_factor_module_from_library_entry(factor_entry, factor_name, out_dir)

    if module_path is not None:
        return run_single_factor_backtest(
            factor_file=str(module_path),
            factor_name=factor_name,
            config_path=config_path,
            output_dir=str(out_dir),
            plot=plot,
            show=show,
            quality_report=quality_report,
            notebook_report=notebook_report,
        )

    tq_upstream = factor_entry.get("tq_upstream") or {}
    factor_path = str(tq_upstream.get("factor_path") or "").strip()
    if level == "minutes" and factor_path:
        return run_single_factor_backtest(
            factor_file=factor_path,
            factor_name=factor_name,
            config_path=config_path,
            output_dir=str(out_dir),
            plot=plot,
            show=show,
            quality_report=quality_report,
            notebook_report=notebook_report,
        )

    expression = str(factor_entry.get("factor_expression") or "").strip()
    if expression:
        return run_single_factor_backtest(
            expression=expression,
            factor_name=factor_name,
            config_path=config_path,
            output_dir=str(out_dir),
            plot=plot,
            show=show,
            quality_report=quality_report,
            notebook_report=notebook_report,
        )

    raise ValueError(
        f"factor '{library_factor}' has no usable standalone asset; "
        "expected factor_module_text/factor_implementation_code or factor_expression."
    )


def main(
    expression: str | None = None,
    factor_file: str | None = None,
    library_path: str | None = None,
    library_factor: str | None = None,
    factor_name: str = "manual_factor",
    config_path: str | None = None,
    output_dir: str | None = None,
    plot: bool = False,
    show: bool = False,
    quality_report: bool = True,
    notebook_report: bool = False,
):
    if library_path:
        if not library_factor:
            raise ValueError("Provide library_factor together with library_path.")
        payload = run_library_factor_backtest(
            library_path=library_path,
            library_factor=library_factor,
            config_path=config_path,
            output_dir=output_dir,
            plot=plot,
            show=show,
            quality_report=quality_report,
            notebook_report=notebook_report,
        )
    else:
        payload = run_single_factor_backtest(
            expression=expression,
            factor_file=factor_file,
            factor_name=factor_name,
            config_path=config_path,
            output_dir=output_dir,
            plot=plot,
            show=show,
            quality_report=quality_report,
            notebook_report=notebook_report,
        )

    if str(os.environ.get("QUANTAALPHA_QUIET_CONSOLE", "")).strip().lower() not in {"1", "true", "yes", "y", "on"}:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    fire.Fire(main)
