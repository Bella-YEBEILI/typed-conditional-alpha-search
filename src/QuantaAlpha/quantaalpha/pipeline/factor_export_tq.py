from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import fire

from quantaalpha.factors.library import FactorLibraryManager
from quantaalpha.runtime import factor_library_path


def _normalize_factor_ids(factor_ids) -> list[str] | None:
    if factor_ids is None:
        return None
    if isinstance(factor_ids, str):
        return [item.strip() for item in factor_ids.split(",") if item.strip()]
    if isinstance(factor_ids, Iterable):
        normalized = [str(item).strip() for item in factor_ids if str(item).strip()]
        return normalized or None
    value = str(factor_ids).strip()
    return [value] if value else None


def main(library_path: str | None = None, factor_ids=None, output_dir: str | None = None):
    """Export factor library entries to TQ-compatible candidate factor modules."""
    resolved_library = Path(library_path) if library_path else factor_library_path()
    result = FactorLibraryManager.export_library_to_tq_candidates(
        library_path=str(resolved_library),
        factor_ids=_normalize_factor_ids(factor_ids),
        output_dir=output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    fire.Fire(main)
