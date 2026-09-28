import ast
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


OUT_DIR = Path(r"C:\Users\35503\Documents\LXW")
PROJECT_ROOT = Path(r"C:\Users\35503\Desktop\research\lxw")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from alpha_factory.config.ops_registry import OPS_REGISTRY
from alpha_factory.core.parser import FormulaParser
from alpha_factory.core.value_engine import ValueEngine
from alpha_factory.factory_data.data_manager import DataManager
from alpha_factory.factory_ops import numpy_funcs


AUDIT_JSON = OUT_DIR/"v4_condition_synthesis_audit.json"
AUDIT_CSV = OUT_DIR/"v4_condition_synthesis_audit.csv"
AUDIT_MD = OUT_DIR/"v4_condition_synthesis_audit.md"


def write_csv(path,rows):
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(v):
    if isinstance(v,(int,float)) and math.isfinite(float(v)):
        return float(v)
    return None


def parse_ops(formula):
    try:
        tree = ast.parse(formula,mode="eval")
    except SyntaxError:
        return []
    ops = []
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            ops.append(node.func.id)
    return ops


def load_candidate_rows():
    rows = []
    for name in ("v3_no_leak_candidates.json","v4_flow_role_candidates.json","v4_bucket_condition_candidates.json"):
        path = OUT_DIR/name
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for row in data:
                row = dict(row)
                row["artifact"] = name
                rows.append(row)
    return rows


def summarize_experiment_usage(rows):
    usage = {
        "eq":0,
        "le":0,
        "ge":0,
        "con_and":0,
        "con_or":0,
        "threshold_pattern":0,
        "rank":0,
        "bucket":0,
        "precompiled_condition_field":0,
    }
    condition_ids = set()
    condition_fields = set()
    for row in rows:
        formula = row.get("formula","")
        ops = set(parse_ops(formula))
        for op in ("eq","le","ge","con_and","con_or","bucket"):
            if op in ops:
                usage[op] += 1
        if {"le","ge","eq"} & ops:
            usage["threshold_pattern"] += 1
        if {"cs_rank","ts_rank","group_rank"} & ops:
            usage["rank"] += 1
        cid = row.get("condition_id") or row.get("selected_condition") or ""
        if cid:
            condition_ids.add(cid)
            usage["precompiled_condition_field"] += 1
        for token in formula.replace("("," ").replace(")"," ").replace(","," ").split():
            if token.endswith("_ts_5") or token.endswith("_ts_20") or token.endswith("_cs_mkt") or token.endswith("_cs_sector"):
                condition_fields.add(token)
    return usage,sorted(condition_ids),sorted(condition_fields)


def audit_formula(parser,value_engine,formula):
    parsed_ok,parse_error = parser.validate(formula)
    row = {
        "formula":formula,
        "parse_ok":parsed_ok,
        "parse_error":parse_error or "",
        "execute_ok":False,
        "shape":"",
        "finite_rate":None,
        "true_rate":None,
        "unique_sample":"",
        "error":"",
    }
    if not parsed_ok:
        return row
    try:
        node = parser.parse(formula)
        value = value_engine.calc(node.root)
        arr = np.asarray(value)
        row["execute_ok"] = True
        row["shape"] = "x".join(str(x) for x in arr.shape)
        finite = np.isfinite(arr)
        row["finite_rate"] = finite.mean().item() if finite.size else None
        sample = arr[finite]
        if sample.size:
            sample = sample[:min(sample.size,200000)]
            row["true_rate"] = (sample>0).mean().item()
            uniq = np.unique(sample[:5000])
            row["unique_sample"] = ",".join(str(float(x)) for x in uniq[:10])
    except Exception as e:
        row["error"] = str(e)
    finally:
        value_engine.clear_value_cache()
    return row


def main():
    dm = DataManager(mmap=True)
    parser = FormulaParser(dm)
    value_engine = ValueEngine(dm,cache_flds=True)
    candidate_rows = load_candidate_rows()
    usage,condition_ids,condition_fields = summarize_experiment_usage(candidate_rows)

    representative = {
        "eq":"eq(bucket(ret1,5),1)",
        "le":"le(ret1,0.0)",
        "ge":"ge(to1,0.0)",
        "con_and":"con_and(le(ret1,0.0),ge(to1,0.0))",
        "con_or":"con_or(le(ret1,0.0),ge(to1,0.0))",
        "threshold":"ge(cs_rank(ret1,True,True),0.8)",
        "rank":"ge(ts_rank(ret5,20),0.8)",
        "bucket":"eq(bucket(ret1,5),3)",
    }

    formula_audit = []
    for category,formula in representative.items():
        row = audit_formula(parser,value_engine,formula)
        row["category"] = category
        formula_audit.append(row)

    coverage_rows = []
    for category,formula in representative.items():
        formula_row = next(r for r in formula_audit if r["category"]==category)
        if category=="threshold":
            registry_support = "pattern via le/ge/eq scalar comparisons, not named threshold op"
            implementation_support = bool(hasattr(numpy_funcs,"le") and hasattr(numpy_funcs,"ge"))
            experiment_count = usage["threshold_pattern"]
        elif category=="rank":
            registry_support = "cs_rank, ts_rank, group_rank"
            implementation_support = all(hasattr(numpy_funcs,op) for op in ("cs_rank","ts_rank","group_rank"))
            experiment_count = usage["rank"]
        else:
            registry_support = category in OPS_REGISTRY
            implementation_support = hasattr(numpy_funcs,category)
            experiment_count = usage.get(category,0)

        if category in ("le","ge","threshold","rank"):
            used_as_precompiled = len(condition_fields)
        else:
            used_as_precompiled = 0

        if formula_row["execute_ok"] and experiment_count>0:
            support_level = "code+execution+main_experiment"
        elif formula_row["execute_ok"] and used_as_precompiled>0 and category in ("le","ge","threshold","rank"):
            support_level = "code+execution+precompiled_condition_use"
        elif formula_row["execute_ok"]:
            support_level = "code+execution_only"
        else:
            support_level = "registered_or_documented_only"

        if category=="threshold":
            gap = "No standalone threshold operator name; write as threshold-style comparison unless a named op is added."
        elif category=="bucket":
            gap = "Executable bucket-to-condition pattern exists, but current main experiments do not use bucket-derived conditions."
        elif category in ("eq","le","ge"):
            gap = "Executable comparison exists, but current main experiments mostly use precompiled condition fields rather than logging generation from raw Float fields."
        elif category=="rank":
            gap = "Rank operators are executable and used, but condition-generation logs should show rank-to-condition construction explicitly."
        else:
            gap = "Logical composition is used in v3 condition crosses; add family-level ablation if claiming composition itself improves stability."

        coverage_rows.append(
            {
                "category":category,
                "representative_formula":formula,
                "registry_support":registry_support,
                "implementation_support":implementation_support,
                "representative_parse_ok":formula_row["parse_ok"],
                "representative_execute_ok":formula_row["execute_ok"],
                "finite_rate":finite_float(formula_row["finite_rate"]),
                "true_rate":finite_float(formula_row["true_rate"]),
                "main_experiment_formula_count":experiment_count,
                "precompiled_condition_field_count":used_as_precompiled,
                "support_level":support_level,
                "paper_risk":gap,
            }
        )

    out = {
        "summary":{
            "candidate_rows_scanned":len(candidate_rows),
            "unique_condition_ids":condition_ids,
            "unique_condition_fields_detected":condition_fields,
            "usage_counts":usage,
            "interpretation":"4.3 is code-and-execution supported, but not yet fully experiment-isolated as an automatic condition-synthesis algorithm.",
        },
        "coverage_rows":coverage_rows,
        "formula_audit":formula_audit,
    }
    AUDIT_JSON.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    write_csv(AUDIT_CSV,coverage_rows)

    lines = []
    lines.append("# Condition Synthesis Audit for Section 4.3")
    lines.append("")
    lines.append(f"Candidate rows scanned: {len(candidate_rows)}.")
    lines.append(f"Unique condition ids: {len(condition_ids)}.")
    lines.append(f"Detected precompiled condition fields: {len(condition_fields)}.")
    lines.append("")
    lines.append("| 4.3 operator family | Representative formula | Support level | Main experiment formula count | Precompiled condition fields | Paper risk |")
    lines.append("|---|---|---|---:|---:|---|")
    for row in coverage_rows:
        lines.append(
            f"| {row['category']} | `{row['representative_formula']}` | {row['support_level']} | "
            f"{row['main_experiment_formula_count']} | {row['precompiled_condition_field_count']} | {row['paper_risk']} |"
        )
    lines.append("")
    lines.append("Reviewer-facing conclusion: Section 4.3 can claim that the framework supports executable condition construction from Float fields through comparison, threshold-style rules, rank, bucket, and logical composition. It should not yet claim that a fully automatic condition-synthesis generator has been exhaustively evaluated as a separate algorithmic contribution.")
    AUDIT_MD.write_text("\n".join(lines),encoding="utf-8")

    print(AUDIT_JSON)
    print(AUDIT_CSV)
    print(AUDIT_MD)


if __name__=="__main__":
    main()
