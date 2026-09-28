from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _markdown_table(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "No summary metrics available."
    table = summary.reset_index().rename(columns={"index": "factor"}).fillna("")
    headers = [str(col) for col in table.columns]
    rows = [headers, ["---" for _ in headers]]
    for _, row in table.iterrows():
        rows.append([str(row[col]) for col in table.columns])
    return "\n".join("| " + " | ".join(items) + " |" for items in rows)


def _artifact_line(label: str, path: Path | None) -> str:
    return f"- {label}: `{Path(path).name}`" if path is not None else f"- {label}: not generated"


def build_factor_report_notebook(
    factor_name: str,
    summary: pd.DataFrame,
    output_path: str | Path,
    summary_path: str | Path,
    factor_value_path: str | Path,
    factor_result_pickle_path: str | Path,
    factor_result_fields: Iterable[str],
    quality_path: str | Path | None = None,
    plot_path: str | Path | None = None,
) -> Path:
    try:
        import nbformat as nbf
    except ImportError as exc:
        raise RuntimeError("nbformat is required to build notebook reports") from exc

    output_path = Path(output_path)
    summary_path = Path(summary_path)
    factor_value_path = Path(factor_value_path)
    factor_result_pickle_path = Path(factor_result_pickle_path)
    quality_path = Path(quality_path) if quality_path else None
    plot_path = Path(plot_path) if plot_path else None
    result_field_list = [str(field) for field in factor_result_fields]

    notebook = nbf.v4.new_notebook()
    summary_markdown = _markdown_table(summary)
    artifact_lines = [
        _artifact_line("summary", summary_path),
        _artifact_line("factor_value", factor_value_path),
        _artifact_line("factor_result_pickle", factor_result_pickle_path),
        _artifact_line("quality", quality_path),
        _artifact_line("diagnostics_plot", plot_path),
    ]
    intro = "\n".join(
        [
            f"# {factor_name} Standalone Backtest Report",
            "",
            "## Summary Metrics",
            summary_markdown,
            "",
            "## Factor Result Fields",
            ", ".join(result_field_list) if result_field_list else "None",
            "",
            "## Artifacts",
            *artifact_lines,
        ]
    )
    cells = [nbf.v4.new_markdown_cell(intro)]

    load_cell = "\n".join(
        [
            "import json",
            "import pickle",
            "from pathlib import Path",
            "import pandas as pd",
            "",
            "base_dir = Path('.')",
            f"summary = json.loads((base_dir / '{summary_path.name}').read_text(encoding='utf-8'))",
            f"factor_value = pd.read_pickle(base_dir / '{factor_value_path.name}')",
            f"with (base_dir / '{factor_result_pickle_path.name}').open('rb') as fh:",
            "    factor_result = pickle.load(fh)",
            "summary_df = pd.DataFrame.from_dict(summary, orient='index')",
            "summary_df",
        ]
    )
    cells.append(nbf.v4.new_code_cell(load_cell))

    result_cell = "\n".join(
        [
            "sorted(factor_result.keys())",
            "",
            "{",
            "    key: type(value).__name__",
            "    for key, value in factor_result.items()",
            "}",
        ]
    )
    cells.append(nbf.v4.new_code_cell(result_cell))

    if quality_path is not None:
        quality_cell = "\n".join(
            [
                f"quality = json.loads((Path('.') / '{quality_path.name}').read_text(encoding='utf-8'))",
                "quality.keys()",
            ]
        )
        cells.append(nbf.v4.new_code_cell(quality_cell))

    if plot_path is not None:
        cells.append(
            nbf.v4.new_markdown_cell(
                "## Diagnostics Plot\n\n" + f"![{factor_name} diagnostics]({plot_path.name})"
            )
        )

    notebook["cells"] = cells
    nbf.write(notebook, output_path)
    return output_path
