import contextlib
import csv
import io
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(r"C:\Users\35503\Desktop\research\lxw")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from alpha_factory.main import AlphaFactory


OUT_DIR = Path(r"C:\Users\35503\Documents\LXW")
RESULT_JSON = OUT_DIR / "v4_flow_role_candidates.json"
RESULT_CSV = OUT_DIR / "v4_flow_role_candidates.csv"
SUMMARY_JSON = OUT_DIR / "v4_flow_role_summary.json"
SELECTED_CSV = OUT_DIR / "v4_flow_role_selected.csv"


def finite_float(v):
    if v is None:
        return None
    if isinstance(v,(int,float)) and math.isfinite(float(v)):
        return float(v)
    return None


def mean(vals):
    vals = [float(v) for v in vals if isinstance(v,(int,float)) and math.isfinite(float(v))]
    return sum(vals)/len(vals) if vals else float("nan")


def median(vals):
    vals = [float(v) for v in vals if isinstance(v,(int,float)) and math.isfinite(float(v))]
    return statistics.median(vals) if vals else float("nan")


def bootstrap_ci(vals,n_boot=5000,seed=20260625):
    vals = [float(v) for v in vals if isinstance(v,(int,float)) and math.isfinite(float(v))]
    if not vals:
        return [float("nan"),float("nan")]
    rng = random.Random(seed)
    n = len(vals)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(vals[rng.randrange(n)] for _ in range(n))/n)
    boots.sort()
    return [boots[int(0.025*n_boot)],boots[int(0.975*n_boot)]]


def sign_test_pvalue(num_pos,n):
    if n<=0:
        return float("nan")
    k = min(num_pos,n-num_pos)
    total = 0
    for i in range(k+1):
        total += math.comb(n,i)
    return min(1.0,2.0*total/(2**n))


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")


def write_csv(path,rows):
    if not rows:
        path.write_text("",encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def seed_specs(fields,source_class):
    specs = []
    for field in fields:
        specs.append((f"{field}_neg",f"neg({field})","math",field,source_class))
        specs.append((f"{field}_tsmean_neg",f"ts_mean(neg({field}),5)","ts",field,source_class))
        specs.append((f"{field}_sector_rank",f"group_rank({field},sectors)","group",field,source_class))
    return specs


def build_candidates():
    flow_fields = [
        "to1",
        "to5",
        "to20",
        "illiq1",
        "illiq5",
        "turnover_rate",
        "cashflow_quality",
        "cashor",
    ]
    nonflow_fields = [
        "ret1",
        "ret5",
        "ret20",
        "vol5",
        "value",
        "ep",
        "bp",
        "roe",
    ]
    flow_conditions = [
        ("to1_ts5","to1_ts_5"),
        ("to5_ts20","to5_ts_20"),
        ("to1_mkt","to1_cs_mkt"),
        ("to20_mkt","to20_cs_mkt"),
        ("illiq1_ts5","illiq1_ts_5"),
        ("illiq5_ts20","illiq5_ts_20"),
        ("illiq5_mkt","illiq5_cs_mkt"),
        ("cashflow_quality_sector","cashflow_quality_cs_sector"),
        ("turnover_rate_sector","turnover_rate_cs_sector"),
    ]
    nonflow_conditions = [
        ("ret1_ts5","ret1_ts_5"),
        ("ret5_ts20","ret5_ts_20"),
        ("vol5_ts20","vol5_ts_20"),
        ("value_sector","value_cs_sector"),
        ("asset_return_sector","asset_return_cs_sector"),
        ("margin_sector","margin_cs_sector"),
    ]
    transforms = ("trade_when","adjust_by","reverse_by","reverse_rank_by")
    specs = seed_specs(flow_fields,"flow_source")+seed_specs(nonflow_fields,"nonflow_source")
    jobs = []
    for seed_id,seed_formula,seed_factory,seed_field,source_class in specs:
        jobs.append(
            {
                "seed_id":seed_id,
                "seed_formula":seed_formula,
                "seed_factory":seed_factory,
                "seed_field":seed_field,
                "source_class":source_class,
                "state_class":"static",
                "family":"static",
                "transform":"static",
                "condition_id":"",
                "formula":seed_formula,
            }
        )
        for state_class,conditions in (("flow_state",flow_conditions),("nonflow_state",nonflow_conditions)):
            for condition_id,condition_formula in conditions:
                for transform in transforms:
                    if transform=="trade_when":
                        formula = f"trade_when({seed_formula},{condition_formula},0.2)"
                    elif transform=="adjust_by":
                        formula = f"adjust_by({seed_formula},{condition_formula},0.5)"
                    elif transform=="reverse_by":
                        formula = f"reverse_by({seed_formula},{condition_formula})"
                    else:
                        formula = f"reverse_rank_by({seed_formula},{condition_formula})"
                    jobs.append(
                        {
                            "seed_id":seed_id,
                            "seed_formula":seed_formula,
                            "seed_factory":seed_factory,
                            "seed_field":seed_field,
                            "source_class":source_class,
                            "state_class":state_class,
                            "family":"conditional",
                            "transform":transform,
                            "condition_id":condition_id,
                            "formula":formula,
                        }
                    )
    return jobs


def eval_formula(af,formula):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = af.check_submission(formula)
    out = {"spec_id":result.get("spec_id"),"formula_id":result.get("formula_id")}
    for period in ("is","os","full"):
        for metric in ("ic","rankic","rankicir","longret","turnover","coverage"):
            out[f"{period}_{metric}"] = finite_float(result[period].get(metric))
    return out


def evaluate_jobs(af,jobs):
    rows = []
    if RESULT_JSON.exists():
        try:
            existing = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
            if isinstance(existing,list):
                rows = existing
                print(f"resume_flow_rows {len(rows)}",flush=True)
        except Exception as exc:
            print(f"resume_flow_failed {exc!r}",flush=True)
    start = time.time()
    for idx,job in enumerate(jobs,1):
        if idx<=len(rows):
            continue
        row = dict(job)
        row["job_index"] = idx
        row["status"] = "ok"
        try:
            row.update(eval_formula(af,job["formula"]))
            row["error"] = ""
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
        rows.append(row)
        if idx%25==0 or idx==len(jobs):
            write_json(RESULT_JSON,rows)
            write_csv(RESULT_CSV,rows)
            ok = sum(1 for r in rows if r["status"]=="ok")
            print(f"flow_progress {idx}/{len(jobs)} ok={ok} elapsed={time.time()-start:.1f}s",flush=True)
    return rows


def summarize_selected(ok_rows):
    selected = []
    seed_ids = sorted({r["seed_id"] for r in ok_rows})
    for seed_id in seed_ids:
        sr = [r for r in ok_rows if r["seed_id"]==seed_id]
        static = next((r for r in sr if r["state_class"]=="static"),None)
        if static is None:
            continue
        for state_class in ("flow_state","nonflow_state"):
            candidates = [r for r in sr if r["state_class"]==state_class]
            if not candidates:
                continue
            best = max(candidates,key=lambda r: r.get("is_rankicir") if isinstance(r.get("is_rankicir"),(int,float)) else -999)
            selected.append(
                {
                    "seed_id":seed_id,
                    "seed_formula":static["seed_formula"],
                    "seed_field":static["seed_field"],
                    "seed_factory":static["seed_factory"],
                    "source_class":static["source_class"],
                    "state_class":state_class,
                    "static_os_rankic":static.get("os_rankic"),
                    "static_os_rankicir":static.get("os_rankicir"),
                    "static_os_longret":static.get("os_longret"),
                    "static_os_turnover":static.get("os_turnover"),
                    "selected_formula":best["formula"],
                    "selected_transform":best["transform"],
                    "selected_condition":best["condition_id"],
                    "selected_is_rankic":best.get("is_rankic"),
                    "selected_is_rankicir":best.get("is_rankicir"),
                    "selected_os_rankic":best.get("os_rankic"),
                    "selected_os_rankicir":best.get("os_rankicir"),
                    "selected_os_longret":best.get("os_longret"),
                    "selected_os_turnover":best.get("os_turnover"),
                    "delta_os_rankic":(best.get("os_rankic") or 0.0)-(static.get("os_rankic") or 0.0),
                }
            )
    return selected


def summarize_group(rows,selected):
    ok_rows = [r for r in rows if r.get("status")=="ok"]
    candidate_summary = []
    for source_class in ("flow_source","nonflow_source"):
        for state_class in ("static","flow_state","nonflow_state"):
            gr = [r for r in ok_rows if r["source_class"]==source_class and r["state_class"]==state_class]
            if not gr:
                continue
            candidate_summary.append(
                {
                    "source_class":source_class,
                    "state_class":state_class,
                    "n":len(gr),
                    "mean_is_rankicir":mean([r.get("is_rankicir") for r in gr]),
                    "mean_os_rankic":mean([r.get("os_rankic") for r in gr]),
                    "median_os_rankic":median([r.get("os_rankic") for r in gr]),
                    "mean_os_rankicir":mean([r.get("os_rankicir") for r in gr]),
                    "mean_os_longret":mean([r.get("os_longret") for r in gr]),
                    "mean_os_turnover":mean([r.get("os_turnover") for r in gr]),
                    "hit_rate_os_rankic_gt_001":mean([1.0 if (r.get("os_rankic") or -999)>0.01 else 0.0 for r in gr]),
                }
            )

    selected_summary = []
    for source_class in ("flow_source","nonflow_source"):
        for state_class in ("flow_state","nonflow_state"):
            gr = [r for r in selected if r["source_class"]==source_class and r["state_class"]==state_class]
            if not gr:
                continue
            deltas = [r["delta_os_rankic"] for r in gr]
            num_pos = sum(1 for d in deltas if d>0)
            selected_summary.append(
                {
                    "source_class":source_class,
                    "state_class":state_class,
                    "n_seeds":len(gr),
                    "mean_static_os_rankic":mean([r["static_os_rankic"] for r in gr]),
                    "mean_selected_os_rankic":mean([r["selected_os_rankic"] for r in gr]),
                    "mean_delta_os_rankic":mean(deltas),
                    "median_delta_os_rankic":median(deltas),
                    "delta_bootstrap_ci95":bootstrap_ci(deltas),
                    "positive_delta_rate":num_pos/len(gr) if gr else float("nan"),
                    "sign_test_pvalue":sign_test_pvalue(num_pos,len(gr)),
                    "mean_selected_os_rankicir":mean([r["selected_os_rankicir"] for r in gr]),
                    "mean_selected_os_longret":mean([r["selected_os_longret"] for r in gr]),
                    "mean_selected_os_turnover":mean([r["selected_os_turnover"] for r in gr]),
                }
            )

    condition_summary = []
    for state_class in ("flow_state","nonflow_state"):
        for condition_id in sorted({r["selected_condition"] for r in selected if r["state_class"]==state_class}):
            gr = [r for r in selected if r["state_class"]==state_class and r["selected_condition"]==condition_id]
            if not gr:
                continue
            condition_summary.append(
                {
                    "state_class":state_class,
                    "selected_condition":condition_id,
                    "n_selected":len(gr),
                    "mean_delta_os_rankic":mean([r["delta_os_rankic"] for r in gr]),
                    "mean_selected_os_rankic":mean([r["selected_os_rankic"] for r in gr]),
                }
            )
    condition_summary.sort(key=lambda r:(r["state_class"],-r["n_selected"],-r["mean_delta_os_rankic"]))

    return {
        "job_count":len(rows),
        "ok_count":len(ok_rows),
        "error_count":len(rows)-len(ok_rows),
        "candidate_summary":candidate_summary,
        "selected_summary":selected_summary,
        "condition_summary":condition_summary,
        "selected":selected,
    }


def main():
    print("v4_flow_role_experiments_start",flush=True)
    started = time.time()
    jobs = build_candidates()
    print(f"flow_jobs {len(jobs)}",flush=True)
    af = AlphaFactory()
    rows = evaluate_jobs(af,jobs)
    ok_rows = [r for r in rows if r.get("status")=="ok"]
    selected = summarize_selected(ok_rows)
    summary = summarize_group(rows,selected)
    write_json(SUMMARY_JSON,summary)
    write_csv(SELECTED_CSV,selected)
    print("selected_summary",json.dumps(summary["selected_summary"],ensure_ascii=False),flush=True)
    print(f"v4_flow_role_experiments_done elapsed={time.time()-started:.1f}s",flush=True)


if __name__=="__main__":
    main()
