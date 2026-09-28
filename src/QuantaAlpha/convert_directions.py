#!/usr/bin/env python3
"""Convert PV and Minutes direction JSON files to Joint-like structured format.

Joint format target:
  Portfolio N, K factors: FactorName, fields=[f1, f2], expr=FORMULA, category=Cat ||
  FactorName2, fields=[f3], expr=FORMULA2, category=Cat2
"""
import json
import re
import sys
from pathlib import Path


def convert_pv_description(desc: str) -> str:
    """Convert PV flat-text direction to Joint-like structured format."""
    header_match = re.match(r"Portfolio\s+(\d+)\s+combines\s+(\d+)\s+factor ideas:\s*", desc)
    if not header_match:
        return desc
    pid = header_match.group(1)
    nfactors = header_match.group(2)
    body = desc[header_match.end():]

    factor_pattern = re.compile(
        r"(\w+)\s+from\s+\w+\.py,\s*uses\s+daily\s+fields\s+"
        r"(.*?),\s*with\s+final\s+calculation\s+"
        r"(.*?),\s*category=(\w+(?:\.\w+)*)(?:,\s*tag=\S+)?\.?"
    )

    factors = []
    for m in factor_pattern.finditer(body):
        name = m.group(1)
        fields_str = m.group(2).strip()
        fields_list = [f.strip() for f in fields_str.split(",")]
        expr = m.group(3).strip()
        category = m.group(4).strip()
        factors.append(f"{name}, fields=[{', '.join(fields_list)}], expr={expr}, category={category}")

    if not factors:
        return desc

    return f"Portfolio {pid}, {nfactors} factors: " + " || ".join(factors)


def _parse_ga_operator_expr(operator_str: str) -> str:
    """Extract a readable expression from ga_m2 operator parameters."""
    params = {}
    for key in ("op_name", "a_name", "b_name", "window", "mask_field", "mask_rule", "lag", "slice_value"):
        m = re.search(rf"'{key}':\s*'?([^',\}}]+)'?", operator_str)
        if m:
            params[key] = m.group(1).strip()

    op = params.get("op_name", "unknown_op")
    a = params.get("a_name", "?")
    b = params.get("b_name", "?")
    w = params.get("window", "")
    mask_f = params.get("mask_field", "")
    mask_r = params.get("mask_rule", "")
    lag = params.get("lag", "")

    parts = [f"{op}({a}, {b}"]
    if w:
        parts.append(f"w={w}")
    if mask_f and mask_r:
        parts.append(f"mask={mask_f}>{mask_r}")
    if lag and lag != "0":
        parts.append(f"lag={lag}")
    return ", ".join(parts) + ")"


def _collect_all_inputs(factor_text: str) -> list[str]:
    """Collect all unique input field names from a minute factor description."""
    all_inputs = []
    for m in re.finditer(r"inputs=\[([^\]]*)\]", factor_text):
        for field in m.group(1).replace("'", "").split(","):
            field = field.strip()
            if field and field not in all_inputs:
                all_inputs.append(field)
    return all_inputs


def convert_minutes_description(desc: str) -> str:
    """Convert Minutes flat-text direction to Joint-like structured format."""
    header_match = re.match(r"Portfolio\s+(\d+)\s+combines\s+(\d+)\s+factor ideas:\s*", desc)
    if not header_match:
        return desc
    pid = header_match.group(1)
    nfactors = header_match.group(2)
    body = desc[header_match.end():]

    # Split body into individual factors:
    # Each factor starts with "FACTOR_NAME from FACTOR_NAME.py"
    factor_splits = re.split(r"(?<=\.)\s+(?=\w+\s+from\s+\w+\.py)", body)

    factors = []
    for ftext in factor_splits:
        ftext = ftext.strip().rstrip(".")
        if not ftext:
            continue

        name_match = re.match(r"(\w+)\s+from\s+\w+\.py", ftext)
        if not name_match:
            continue
        name = name_match.group(1)

        all_inputs = _collect_all_inputs(ftext)

        # Extract expr from "and finishes with EXPR"
        finish_match = re.search(r"and\s+finishes\s+with\s+(.+?)(?:,\s*category=|,\s*tag=|$)", ftext)
        expr = ""
        if finish_match:
            expr = finish_match.group(1).strip().rstrip(",").strip()

        # Extract category
        cat_match = re.search(r"category=(\S+?)(?:,|$)", ftext)
        category = cat_match.group(1).rstrip(".,") if cat_match else ""

        # For ga_m2 types with minute_ctx['result'], construct expr from operator params
        sign_prefix = ""
        if "minute_ctx" in expr or not expr:
            if expr.startswith("-"):
                sign_prefix = "neg "
            op_match = re.search(r"operator=\(([^)]+\))\)", ftext)
            if not op_match:
                op_match = re.search(r"operator=\((.+?)\),\s*(?:preprocess|startminute|endminute)", ftext)
            if op_match:
                expr = sign_prefix + _parse_ga_operator_expr(op_match.group(0))
            elif not expr:
                expr = name

        fields_str = f"[{', '.join(all_inputs)}]" if all_inputs else "[]"
        parts = [f"{name}, fields={fields_str}, expr={expr}"]
        if category:
            parts[0] += f", category={category}"
        factors.append(parts[0])

    if not factors:
        return desc

    return f"Portfolio {pid}, {nfactors} factors: " + " || ".join(factors)


def convert_file(input_path: str, convert_fn, backup: bool = True):
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    portfolios = data.get("factor_portfolios", [])
    if not portfolios:
        print(f"  No portfolios found in {path.name}")
        return

    changed = 0
    for item in portfolios:
        old_desc = item.get("description", "")
        new_desc = convert_fn(old_desc)
        if new_desc != old_desc:
            item["description"] = new_desc
            changed += 1

    if changed == 0:
        print(f"  No changes needed for {path.name}")
        return

    if backup:
        backup_path = path.with_suffix(".json.bak")
        with backup_path.open("w", encoding="utf-8") as f:
            # Read original for backup
            pass
        import shutil
        shutil.copy2(path, backup_path)
        print(f"  Backed up to {backup_path.name}")

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  Converted {changed}/{len(portfolios)} portfolios in {path.name}")


def preview(input_path: str, convert_fn, n: int = 3):
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    portfolios = data.get("factor_portfolios", [])
    for item in portfolios[:n]:
        pid = item.get("portfolio_id", "?")
        old = item.get("description", "")
        new = convert_fn(old)
        print(f"=== Portfolio {pid} ===")
        print(f"NEW: {new}")
        print()


if __name__ == "__main__":
    base = Path("/home/workspace/users/liwei/tqstrategyserver/QuantaAlpha/experiment")

    mode = sys.argv[1] if len(sys.argv) > 1 else "preview"

    pv_path = str(base / "original_direction_daily.json")
    min_path = str(base / "original_direction_minutes.json")

    if mode == "preview":
        print("=== PV (daily) Preview ===")
        preview(pv_path, convert_pv_description, n=3)
        print("\n=== Minutes Preview ===")
        preview(min_path, convert_minutes_description, n=3)
    elif mode == "convert":
        print("Converting PV (daily)...")
        convert_file(pv_path, convert_pv_description)
        print("Converting Minutes...")
        convert_file(min_path, convert_minutes_description)
        print("Done!")
    else:
        print(f"Usage: {sys.argv[0]} [preview|convert]")
