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


PROJECT_ROOT = Path(r"C:\Users\35503\Desktop\research\lxw")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from alpha_factory.main import AlphaFactory


OUT_DIR = Path(r"C:\Users\35503\Documents\LXW")
RESULT_JSON = OUT_DIR/"v4_bucket_condition_candidates.json"
RESULT_CSV = OUT_DIR/"v4_bucket_condition_candidates.csv"
SUMMARY_JSON = OUT_DIR/"v4_bucket_condition_summary.json"
SELECTED_CSV = OUT_DIR/"v4_bucket_condition_selected.csv"


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


def build_candidates():
    seed_specs = [
        ("ret1_neg","neg(ret1)","math","ret1"),
        ("ret5_neg","neg(ret5)","math","ret5"),
        ("ret20_neg","neg(ret20)","math","ret20"),
        ("vol5_neg","neg(vol5)","math","vol5"),
        ("to1_neg","neg(to1)","math","to1"),
        ("to5_neg","neg(to5)","math","to5"),
        ("value_sector_rank","group_rank(value,sectors)","group","value"),
        ("ep_sector_rank","group_rank(ep,sectors)","group","ep"),
        ("bp_sector_rank","group_rank(bp,sectors)","group","bp"),
        ("roe_sector_rank","group_rank(roe,sectors)","group","roe"),
        ("illiq1_neg","neg(illiq1)","math","illiq1"),
        ("cashflow_quality_rank","group_rank(cashflow_quality,sectors)","group","cashflow_quality"),
    ]
    bucket_conditions = [
        ("ret1_bucket_low","eq(bucket(ret1,5),0)"),
        ("ret1_bucket_high","eq(bucket(ret1,5),4)"),
        ("ret5_bucket_low","eq(bucket(ret5,5),0)"),
        ("ret5_bucket_high","eq(bucket(ret5,5),4)"),
        ("to1_bucket_high","eq(bucket(to1,5),4)"),
        ("vol5_bucket_high","eq(bucket(vol5,5),4)"),
        ("value_bucket_low","eq(bucket(value,5),0)"),
        ("ep_bucket_high","eq(bucket(ep,5),4)"),
    ]
    transforms = ("trade_when","adjust_by","reverse_by","reverse_rank_by")
    jobs = []
    for seed_id,seed_formula,seed_factory,seed_field in seed_specs:
        jobs.append(
            {
                "seed_id":seed_id,
                "seed_formula":seed_formula,
                "seed_factory":seed_factory,
                "seed_field":seed_field,
                "condition_id":"",
                "condition_formula":"",
                "family":"static",
                "transform":"static",
                "formula":seed_formula,
            }
        )
        for condition_id,condition_formula in bucket_conditions:
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
                        "condition_id":condition_id,
                        "condition_formula":condition_formula,
                        "family":"bucket_condition",
                        "transform":transform,
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
                print(f"resume_bucket_rows {len(rows)}",flush=True)
        except Exception as exc:
            print(f"resume_bucket_failed {exc!r}",flush=True)
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
        if idx%24==0 or idx==len(jobs):
            write_json(RESULT_JSON,rows)
            write_csv(RESULT_CSV,rows)
            ok = sum(1 for r in rows if r["status"]=="ok")
            print(f"bucket_progress {idx}/{len(jobs)} ok={ok} elapsed={time.time()-start:.1f}s",flush=True)
    return rows


def summarize(rows):
    ok_rows = [r for r in rows if r.get("status")=="ok"]
    selected = []
    for seed_id in sorted({r["seed_id"] for r in ok_rows}):
        sr = [r for r in ok_rows if r["seed_id"]==seed_id]
        static = next((r for r in sr if r["family"]=="static"),None)
        candidates = [r for r in sr if r["family"]=="bucket_condition"]
        if static is None or not candidates:
            continue
        best = max(candidates,key=lambda r: r.get("is_rankicir") if isinstance(r.get("is_rankicir"),(int,float)) else -999)
        selected.append(
            {
                "seed_id":seed_id,
                "seed_formula":static["formula"],
                "static_os_rankic":static.get("os_rankic"),
                "static_os_rankicir":static.get("os_rankicir"),
                "selected_condition":best["condition_id"],
                "selected_condition_formula":best["condition_formula"],
                "selected_transform":best["transform"],
                "selected_formula":best["formula"],
                "selected_is_rankicir":best.get("is_rankicir"),
                "selected_os_rankic":best.get("os_rankic"),
                "selected_os_rankicir":best.get("os_rankicir"),
                "selected_os_turnover":best.get("os_turnover"),
                "selected_os_coverage":best.get("os_coverage"),
                "delta_os_rankic":(best.get("os_rankic") or 0.0)-(static.get("os_rankic") or 0.0),
            }
        )
    deltas = [r["delta_os_rankic"] for r in selected]
    num_pos = sum(1 for d in deltas if d>0)
    family_summary = []
    for family in ("static","bucket_condition"):
        fr = [r for r in ok_rows if r["family"]==family]
        family_summary.append(
            {
                "family":family,
                "n":len(fr),
                "mean_os_rankic":mean([r.get("os_rankic") for r in fr]),
                "median_os_rankic":median([r.get("os_rankic") for r in fr]),
                "mean_os_rankicir":mean([r.get("os_rankicir") for r in fr]),
                "hit_rate_os_rankic_gt_001":mean([1.0 if (r.get("os_rankic") or -999)>0.01 else 0.0 for r in fr]),
                "mean_os_coverage":mean([r.get("os_coverage") for r in fr]),
            }
        )
    condition_summary = []
    for condition_id in sorted({r["condition_id"] for r in ok_rows if r["family"]=="bucket_condition"}):
        cr = [r for r in ok_rows if r["condition_id"]==condition_id]
        condition_summary.append(
            {
                "condition_id":condition_id,
                "n":len(cr),
                "mean_os_rankic":mean([r.get("os_rankic") for r in cr]),
                "median_os_rankic":median([r.get("os_rankic") for r in cr]),
                "hit_rate_os_rankic_gt_001":mean([1.0 if (r.get("os_rankic") or -999)>0.01 else 0.0 for r in cr]),
                "selected_count":sum(1 for r in selected if r["selected_condition"]==condition_id),
            }
        )
    condition_summary.sort(key=lambda r:(-r["selected_count"],-r["mean_os_rankic"]))
    out = {
        "candidate_count":len(rows),
        "ok_count":len(ok_rows),
        "static_seed_count":len(selected),
        "bucket_condition_count":len({r["condition_id"] for r in ok_rows if r["family"]=="bucket_condition"}),
        "selection_summary":{
            "n_seeds":len(selected),
            "mean_static_os_rankic":mean([r["static_os_rankic"] for r in selected]),
            "mean_selected_os_rankic":mean([r["selected_os_rankic"] for r in selected]),
            "mean_delta_os_rankic":mean(deltas),
            "median_delta_os_rankic":median(deltas),
            "delta_bootstrap_ci95":bootstrap_ci(deltas),
            "positive_delta_rate":num_pos/len(deltas) if deltas else float("nan"),
            "sign_test_pvalue":sign_test_pvalue(num_pos,len(deltas)),
            "mean_selected_os_rankicir":mean([r["selected_os_rankicir"] for r in selected]),
            "mean_selected_os_turnover":mean([r["selected_os_turnover"] for r in selected]),
            "mean_selected_os_coverage":mean([r["selected_os_coverage"] for r in selected]),
        },
        "family_summary":family_summary,
        "condition_summary":condition_summary,
        "selected":selected,
    }
    write_json(SUMMARY_JSON,out)
    write_csv(SELECTED_CSV,selected)
    return out


def main():
    jobs = build_candidates()
    print(f"bucket_jobs {len(jobs)}",flush=True)
    af = AlphaFactory()
    rows = evaluate_jobs(af,jobs)
    summary = summarize(rows)
    print(json.dumps(summary["selection_summary"],ensure_ascii=False,indent=2),flush=True)
    print(SUMMARY_JSON)
    print(SELECTED_CSV)


if __name__=="__main__":
    main()
