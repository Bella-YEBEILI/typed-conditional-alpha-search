from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import fire

from quantaalpha.runtime import factor_library_dir

from .run_backtest import run_library_factor_backtest

FORCED_PLOT = True
FORCED_QUALITY_REPORT = True
FORCED_NOTEBOOK_REPORT = True


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "", "none", "null"}:
        return False
    return bool(value)


def _parse_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "auto", "any"}:
        return None
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean filter value: {value!r}")


def _sanitize_path_component(value: str | None, fallback: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return fallback
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", raw)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip("._ ")
    return cleaned or fallback


def _collect_truthy_check_fields(factor_info: Any) -> list[str]:
    if not isinstance(factor_info, dict):
        return []

    matched: list[str] = []
    direct_fields = (
        "check_passed",
        "train_passed",
        "test_passed",
    )
    for field_name in direct_fields:
        if _to_bool(factor_info.get(field_name)):
            matched.append(field_name)

    nested_fields = (
        "train_check_flags",
        "test_check_flags",
    )
    for field_name in nested_fields:
        nested = factor_info.get(field_name) or {}
        if isinstance(nested, dict) and _to_bool(nested.get("check_passed")):
            matched.append(f"{field_name}.check_passed")
    return matched


def _resolve_stage_check_passed(factor_info: Any, stage: str) -> bool | None:
    if not isinstance(factor_info, dict):
        return None
    stage_name = str(stage or "").strip().lower()
    if stage_name not in {"train", "test"}:
        raise ValueError(f"unsupported stage: {stage}")

    direct_key = f"{stage_name}_passed"
    if direct_key in factor_info:
        return _to_bool(factor_info.get(direct_key))

    nested = factor_info.get(f"{stage_name}_check_flags") or {}
    if isinstance(nested, dict) and "check_passed" in nested:
        return _to_bool(nested.get("check_passed"))
    return None


def _split_library_refs(library: str) -> list[str]:
    raw = str(library or "").strip()
    if not raw:
        raise ValueError("library cannot be empty")
    normalized = raw.replace("，", ",").replace("；", ";")
    parts = [item.strip() for item in re.split(r"[,;\r\n]+", normalized) if item.strip()]
    return parts or [raw]


def _resolve_library_path(library: str) -> Path:
    raw = str(library or "").strip()
    if not raw:
        raise ValueError("library cannot be empty")

    direct = Path(raw).expanduser()
    candidates: list[Path] = []
    if direct.is_absolute():
        candidates.append(direct)
    else:
        base_dir = factor_library_dir()
        candidates.extend(
            [
                direct,
                base_dir / direct,
                base_dir / f"{raw}.json",
                base_dir / f"all_factors_library_{raw}.json",
            ]
        )

    existing: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            existing.append(resolved)

    if not existing:
        raise FileNotFoundError(
            f"factor library '{library}' not found under {factor_library_dir()} or as a direct path"
        )

    if len(existing) > 1:
        raise ValueError(
            "factor library reference is ambiguous; use an exact filename or absolute path: "
            + ", ".join(str(path) for path in existing)
        )
    return existing[0]


def _load_library(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    factors = payload.get("factors")
    if not isinstance(factors, dict) or not factors:
        raise ValueError(f"factor library is empty: {path}")
    return payload


def _sanitize_output_suffix(output_suffix: str | None) -> str:
    raw = str(output_suffix or "").strip()
    if not raw:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    return cleaned.strip("._")


def resolve_default_output_dir(
    library_paths: list[Path],
    output_dir: str | None = None,
    output_suffix: str | None = None,
) -> Path:
    if output_dir:
        return Path(output_dir)
    suffix = _sanitize_output_suffix(output_suffix)
    if suffix:
        folder_name = suffix
    elif len(library_paths) == 1:
        folder_name = library_paths[0].stem
    else:
        folder_name = "multi_library_batch"
    return factor_library_dir().parent / "tq_upstream" / folder_name


def main(
    library: str,
    config_path: str | None = None,
    output_dir: str | None = None,
    output_suffix: str | None = None,
    continue_on_error: bool = True,
    limit: int | None = None,
    require_train_passed: bool | str | None = None,
    require_test_passed: bool | str | None = None,
):
    continue_on_error = bool(continue_on_error)
    require_train_passed = _parse_optional_bool(require_train_passed)
    require_test_passed = _parse_optional_bool(require_test_passed)

    library_refs = _split_library_refs(library)
    library_paths = [_resolve_library_path(ref) for ref in library_refs]
    skipped_unchecked: list[dict[str, str]] = []
    factor_items: list[tuple[Path, str, Any]] = []
    library_stats: list[dict[str, Any]] = []

    for library_path in library_paths:
        payload = _load_library(library_path)
        all_factor_items = list((payload.get("factors") or {}).items())
        eligible_count = 0
        skipped_count = 0

        for factor_id, factor_info in all_factor_items:
            matched_fields = _collect_truthy_check_fields(factor_info)
            factor_name = str((factor_info or {}).get("factor_name") or factor_id).strip() or str(factor_id)
            train_passed = _resolve_stage_check_passed(factor_info, "train")
            test_passed = _resolve_stage_check_passed(factor_info, "test")

            if not matched_fields:
                skipped_unchecked.append(
                    {
                        "library_path": str(library_path),
                        "factor_id": str(factor_id),
                        "factor_name": factor_name,
                        "reason": "no truthy check_passed/train_passed/test_passed flag found",
                    }
                )
                skipped_count += 1
                continue

            if require_train_passed is not None and train_passed != require_train_passed:
                skipped_unchecked.append(
                    {
                        "library_path": str(library_path),
                        "factor_id": str(factor_id),
                        "factor_name": factor_name,
                        "reason": f"train check_passed={train_passed} does not match required {require_train_passed}",
                    }
                )
                skipped_count += 1
                continue

            if require_test_passed is not None and test_passed != require_test_passed:
                skipped_unchecked.append(
                    {
                        "library_path": str(library_path),
                        "factor_id": str(factor_id),
                        "factor_name": factor_name,
                        "reason": f"test check_passed={test_passed} does not match required {require_test_passed}",
                    }
                )
                skipped_count += 1
                continue

            factor_items.append((library_path, factor_id, factor_info))
            eligible_count += 1

        library_stats.append(
            {
                "library_path": str(library_path),
                "library_name": library_path.name,
                "total": len(all_factor_items),
                "eligible_count": eligible_count,
                "skipped_count": skipped_count,
            }
        )

    if limit is not None:
        factor_items = factor_items[: max(0, int(limit))]

    default_output_dir = resolve_default_output_dir(
        library_paths=library_paths,
        output_dir=output_dir,
        output_suffix=output_suffix,
    )
    default_output_dir.mkdir(parents=True, exist_ok=True)

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    used_factor_dirs: set[str] = set()
    multi_library_mode = len(library_paths) > 1

    for library_path, factor_id, factor_info in factor_items:
        factor_name = str((factor_info or {}).get("factor_name") or factor_id).strip() or str(factor_id)
        factor_folder_name = _sanitize_path_component(factor_name, fallback=str(factor_id))
        library_folder_name = _sanitize_path_component(library_path.stem, fallback="library")
        dedupe_key = factor_folder_name if not multi_library_mode else f"{library_folder_name}/{factor_folder_name}"
        if dedupe_key in used_factor_dirs:
            factor_folder_name = _sanitize_path_component(f"{factor_name}_{factor_id}", fallback=str(factor_id))
            dedupe_key = factor_folder_name if not multi_library_mode else f"{library_folder_name}/{factor_folder_name}"
        used_factor_dirs.add(dedupe_key)
        factor_output_dir = (
            default_output_dir / library_folder_name / factor_folder_name
            if multi_library_mode
            else default_output_dir / factor_folder_name
        )
        factor_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = run_library_factor_backtest(
                library_path=str(library_path),
                library_factor=str(factor_id),
                config_path=config_path,
                output_dir=str(factor_output_dir),
                plot=FORCED_PLOT,
                show=False,
                quality_report=FORCED_QUALITY_REPORT,
                notebook_report=FORCED_NOTEBOOK_REPORT,
            )
            result["factor_id"] = str(factor_id)
            result["library_path"] = str(library_path)
            result["library_name"] = library_path.name
            result["factor_output_dir"] = str(factor_output_dir)
            result["eligible_check_fields"] = _collect_truthy_check_fields(factor_info)
            successes.append(result)
        except Exception as exc:
            failure = {
                "library_path": str(library_path),
                "library_name": library_path.name,
                "factor_id": str(factor_id),
                "factor_name": factor_name,
                "factor_output_dir": str(factor_output_dir),
                "eligible_check_fields": _collect_truthy_check_fields(factor_info),
                "error": str(exc),
            }
            failures.append(failure)
            if not continue_on_error:
                break

    summary = {
        "library_path": str(library_paths[0]) if len(library_paths) == 1 else None,
        "library_paths": [str(path) for path in library_paths],
        "library_count": len(library_paths),
        "output_dir": str(default_output_dir),
        "output_suffix": _sanitize_output_suffix(output_suffix) or None,
        "total": sum(int(item["total"]) for item in library_stats),
        "eligible_count": len(factor_items),
        "skipped_unchecked_count": len(skipped_unchecked),
        "success_count": len(successes),
        "failure_count": len(failures),
        "libraries": library_stats,
        "successes": successes,
        "failures": failures,
        "skipped_unchecked": skipped_unchecked,
    }
    summary_filename = (
        f"{library_paths[0].stem}_batch_summary.json"
        if len(library_paths) == 1
        else "multi_library_batch_summary.json"
    )
    summary_path = default_output_dir / summary_filename
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)

    if str(os.environ.get("QUANTAALPHA_QUIET_CONSOLE", "")).strip().lower() not in {"1", "true", "yes", "y", "on"}:
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    fire.Fire(main)
