import contextlib
import csv
import io
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np

from alpha_factory.main import AlphaFactory
from alpha_factory.config.fld_registry import FLD_REGISTRY


OUT_DIR = Path(r"C:\Users\35503\Documents\LXW")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_JSON = OUT_DIR / "v3_no_leak_candidates.json"
RESULT_CSV = OUT_DIR / "v3_no_leak_candidates.csv"
SUMMARY_JSON = OUT_DIR / "v3_no_leak_summary.json"
ROLLING_JSON = OUT_DIR / "v3_no_leak_rolling.json"
ROLLING_CSV = OUT_DIR / "v3_no_leak_rolling.csv"
UNTYPED_JSON = OUT_DIR / "v3_untyped_ablation.json"
UNTYPED_CSV = OUT_DIR / "v3_untyped_ablation.csv"


def finite_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def mean(vals):
    vals = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def median(vals):
    vals = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return statistics.median(vals) if vals else float("nan")


def bootstrap_ci(vals, n_boot=5000, seed=20260625):
    vals = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    boots = []
    n = len(vals)
    for _ in range(n_boot):
        boots.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return (boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)])


def sign_test_pvalue(num_pos, n):
    if n <= 0:
        return float("nan")
    k = min(num_pos, n - num_pos)
    # Two-sided exact binomial test under p=0.5.
    total = 0
    for i in range(k + 1):
        total += math.comb(n, i)
    return min(1.0, 2.0 * total / (2 ** n))


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_candidates():
    base_fields = [
        "ret1",
        "ret5",
        "ret20",
        "vol5",
        "to1",
        "to5",
        "illiq1",
        "value",
        "ep",
        "bp",
    ]
    seed_specs = []
    for field in base_fields:
        seed_specs.append((f"{field}_neg", f"neg({field})", "math", field))
        seed_specs.append((f"{field}_tsmean_neg", f"ts_mean(neg({field}),5)", "ts", field))
        seed_specs.append((f"{field}_sector_rank", f"group_rank({field},sectors)", "group", field))

    conditions = [
        ("ret1_ts5", "ret1_ts_5"),
        ("ret5_ts20", "ret5_ts_20"),
        ("vol5_ts20", "vol5_ts_20"),
        ("to1_ts5", "to1_ts_5"),
        ("to1_mkt", "to1_cs_mkt"),
        ("value_sector", "value_cs_sector"),
        ("cashflow_quality_sector", "cashflow_quality_cs_sector"),
    ]
    cross_conditions = [
        ("ret_vol_and", "con_and(ret1_ts_5,vol5_ts_20)"),
        ("ret_vol_or", "con_or(ret1_ts_5,vol5_ts_20)"),
        ("to_vol_and", "con_and(to1_ts_5,vol5_ts_20)"),
        ("value_cashflow_and", "con_and(value_cs_sector,cashflow_quality_cs_sector)"),
    ]

    jobs = []
    for i, (seed_id, seed_formula, seed_factory, seed_field) in enumerate(seed_specs):
        jobs.append(
            {
                "seed_id": seed_id,
                "seed_formula": seed_formula,
                "seed_factory": seed_factory,
                "seed_field": seed_field,
                "family": "static",
                "transform": "static",
                "condition_id": "",
                "formula": seed_formula,
            }
        )
        for condition_id, condition_formula in conditions:
            for transform in ("trade_when", "adjust_by", "reverse_by", "reverse_rank_by"):
                if transform == "trade_when":
                    formula = f"trade_when({seed_formula},{condition_formula},0.2)"
                elif transform == "adjust_by":
                    formula = f"adjust_by({seed_formula},{condition_formula},0.5)"
                elif transform == "reverse_by":
                    formula = f"reverse_by({seed_formula},{condition_formula})"
                else:
                    formula = f"reverse_rank_by({seed_formula},{condition_formula})"
                jobs.append(
                    {
                        "seed_id": seed_id,
                        "seed_formula": seed_formula,
                        "seed_factory": seed_factory,
                        "seed_field": seed_field,
                        "family": "conditional",
                        "transform": transform,
                        "condition_id": condition_id,
                        "formula": formula,
                    }
                )
        for condition_id, condition_formula in cross_conditions:
            for transform in ("trade_when", "adjust_by"):
                if transform == "trade_when":
                    formula = f"trade_when({seed_formula},{condition_formula},0.2)"
                else:
                    formula = f"adjust_by({seed_formula},{condition_formula},0.5)"
                jobs.append(
                    {
                        "seed_id": seed_id,
                        "seed_formula": seed_formula,
                        "seed_factory": seed_factory,
                        "seed_field": seed_field,
                        "family": "condition_cross",
                        "transform": transform,
                        "condition_id": condition_id,
                        "formula": formula,
                    }
                )

        alt_1 = seed_specs[(i + 1) % len(seed_specs)][1]
        alt_2 = seed_specs[(i + 7) % len(seed_specs)][1]
        for j, (condition_id, condition_formula, alt_formula) in enumerate(
            [
                ("if_ret1_ts5", "ret1_ts_5", alt_1),
                ("if_to1_ts5", "to1_ts_5", alt_2),
            ]
        ):
            jobs.append(
                {
                    "seed_id": seed_id,
                    "seed_formula": seed_formula,
                    "seed_factory": seed_factory,
                    "seed_field": seed_field,
                    "family": "if_else",
                    "transform": "if_else",
                    "condition_id": condition_id,
                    "formula": f"if_else({seed_formula},{alt_formula},{condition_formula})",
                }
            )
    return jobs


def eval_formula(af, formula):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = af.check_submission(formula)
    out = {
        "spec_id": result.get("spec_id"),
        "formula_id": result.get("formula_id"),
    }
    for period in ("is", "os", "full"):
        for metric in ("ic", "rankic", "rankicir", "longret", "turnover", "coverage"):
            out[f"{period}_{metric}"] = finite_float(result[period].get(metric))
    return out


def evaluate_jobs(af, jobs):
    rows = []
    if RESULT_JSON.exists():
        try:
            existing = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                rows = existing
                print(f"resume_main_rows {len(rows)}", flush=True)
        except Exception as exc:
            print(f"resume_main_failed {exc!r}", flush=True)
    start = time.time()
    for idx, job in enumerate(jobs, 1):
        if idx <= len(rows):
            continue
        row = dict(job)
        row["job_index"] = idx
        row["status"] = "ok"
        try:
            row.update(eval_formula(af, job["formula"]))
            row["error"] = ""
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
        rows.append(row)
        if idx % 25 == 0 or idx == len(jobs):
            write_json(RESULT_JSON, rows)
            write_csv(RESULT_CSV, rows)
            ok = sum(1 for r in rows if r["status"] == "ok")
            print(f"main_progress {idx}/{len(jobs)} ok={ok} elapsed={time.time() - start:.1f}s", flush=True)
    return rows


def summarize_main(rows):
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    family_summary = []
    for family in ("static", "conditional", "condition_cross", "if_else"):
        fr = [r for r in ok_rows if r["family"] == family]
        family_summary.append(
            {
                "family": family,
                "n": len(fr),
                "mean_is_rankicir": mean([r.get("is_rankicir") for r in fr]),
                "mean_os_rankic": mean([r.get("os_rankic") for r in fr]),
                "median_os_rankic": median([r.get("os_rankic") for r in fr]),
                "mean_os_rankicir": mean([r.get("os_rankicir") for r in fr]),
                "mean_os_longret": mean([r.get("os_longret") for r in fr]),
                "mean_os_turnover": mean([r.get("os_turnover") for r in fr]),
                "survival_os_rankic_gt_001": mean([1.0 if (r.get("os_rankic") or -999) > 0.01 else 0.0 for r in fr]),
            }
        )

    transform_summary = []
    for transform in sorted({r["transform"] for r in ok_rows}):
        tr = [r for r in ok_rows if r["transform"] == transform]
        transform_summary.append(
            {
                "transform": transform,
                "n": len(tr),
                "mean_os_rankic": mean([r.get("os_rankic") for r in tr]),
                "median_os_rankic": median([r.get("os_rankic") for r in tr]),
                "mean_os_rankicir": mean([r.get("os_rankicir") for r in tr]),
                "mean_os_longret": mean([r.get("os_longret") for r in tr]),
                "mean_os_turnover": mean([r.get("os_turnover") for r in tr]),
                "survival_os_rankic_gt_001": mean([1.0 if (r.get("os_rankic") or -999) > 0.01 else 0.0 for r in tr]),
            }
        )

    selected = []
    for seed_id in sorted({r["seed_id"] for r in ok_rows}):
        sr = [r for r in ok_rows if r["seed_id"] == seed_id]
        static = next((r for r in sr if r["family"] == "static"), None)
        candidates = [r for r in sr if r["family"] != "static"]
        if static is None or not candidates:
            continue
        selected_is = max(
            candidates,
            key=lambda r: r.get("is_rankicir") if isinstance(r.get("is_rankicir"), (int, float)) else -999,
        )
        oracle_os = max(
            candidates,
            key=lambda r: r.get("os_rankic") if isinstance(r.get("os_rankic"), (int, float)) else -999,
        )
        selected.append(
            {
                "seed_id": seed_id,
                "seed_formula": static["seed_formula"],
                "seed_factory": static["seed_factory"],
                "static_os_rankic": static.get("os_rankic"),
                "static_os_rankicir": static.get("os_rankicir"),
                "static_os_longret": static.get("os_longret"),
                "static_os_turnover": static.get("os_turnover"),
                "selected_formula": selected_is["formula"],
                "selected_family": selected_is["family"],
                "selected_transform": selected_is["transform"],
                "selected_condition": selected_is["condition_id"],
                "selected_is_rankicir": selected_is.get("is_rankicir"),
                "selected_is_rankic": selected_is.get("is_rankic"),
                "selected_os_rankic": selected_is.get("os_rankic"),
                "selected_os_rankicir": selected_is.get("os_rankicir"),
                "selected_os_longret": selected_is.get("os_longret"),
                "selected_os_turnover": selected_is.get("os_turnover"),
                "delta_os_rankic": (selected_is.get("os_rankic") or 0.0) - (static.get("os_rankic") or 0.0),
                "oracle_os_rankic": oracle_os.get("os_rankic"),
                "oracle_formula": oracle_os["formula"],
            }
        )

    deltas = [r["delta_os_rankic"] for r in selected]
    num_pos = sum(1 for d in deltas if d > 0)
    ci_lo, ci_hi = bootstrap_ci(deltas)
    selection_summary = {
        "n_seeds": len(selected),
        "mean_static_os_rankic": mean([r["static_os_rankic"] for r in selected]),
        "mean_selected_os_rankic": mean([r["selected_os_rankic"] for r in selected]),
        "mean_delta_os_rankic": mean(deltas),
        "median_delta_os_rankic": median(deltas),
        "delta_bootstrap_ci95": [ci_lo, ci_hi],
        "positive_delta_rate": num_pos / len(selected) if selected else float("nan"),
        "sign_test_pvalue": sign_test_pvalue(num_pos, len(selected)),
        "mean_selected_os_rankicir": mean([r["selected_os_rankicir"] for r in selected]),
        "mean_selected_os_longret": mean([r["selected_os_longret"] for r in selected]),
        "mean_selected_os_turnover": mean([r["selected_os_turnover"] for r in selected]),
        "mean_oracle_os_rankic": mean([r["oracle_os_rankic"] for r in selected]),
    }

    return {
        "job_count": len(rows),
        "ok_count": len(ok_rows),
        "error_count": len(rows) - len(ok_rows),
        "family_summary": family_summary,
        "transform_summary": transform_summary,
        "selected_by_is_rankicir": selected,
        "selection_summary": selection_summary,
    }


def daily_rankics(af, formula):
    node = af.parser.parse(formula)
    value = af.value_engine.calc(node.root)
    value = af.post_processor.process(value, 0, None)
    result = af.full_result_engine.calc(value)
    return np.asarray(result.rankics, dtype=np.float64)


def compute_rolling(af, selected):
    cache = {}

    def get(formula):
        if formula not in cache:
            cache[formula] = daily_rankics(af, formula)
        return cache[formula]

    rows = []
    for idx, row in enumerate(selected, 1):
        static = get(row["seed_formula"])
        conditional = get(row["selected_formula"])
        n = min(len(static), len(conditional))
        static = static[:n]
        conditional = conditional[:n]
        edges = np.linspace(0, n, 6, dtype=int)
        for block in range(5):
            lo, hi = int(edges[block]), int(edges[block + 1])
            s = static[lo:hi]
            c = conditional[lo:hi]
            mask = np.isfinite(s) & np.isfinite(c)
            if mask.sum() == 0:
                continue
            delta = c[mask] - s[mask]
            rows.append(
                {
                    "seed_id": row["seed_id"],
                    "block": block + 1,
                    "n_days": int(mask.sum()),
                    "static_rankic": float(np.nanmean(s[mask])),
                    "selected_rankic": float(np.nanmean(c[mask])),
                    "delta_rankic": float(np.nanmean(delta)),
                    "daily_win_rate": float(np.mean(delta > 0)),
                    "selected_transform": row["selected_transform"],
                    "selected_condition": row["selected_condition"],
                    "selected_formula": row["selected_formula"],
                }
            )
        if idx % 5 == 0 or idx == len(selected):
            print(f"rolling_progress {idx}/{len(selected)} formulas_cached={len(cache)}", flush=True)

    by_block = []
    for block in range(1, 6):
        br = [r for r in rows if r["block"] == block]
        by_block.append(
            {
                "block": block,
                "n_seed_blocks": len(br),
                "mean_static_rankic": mean([r["static_rankic"] for r in br]),
                "mean_selected_rankic": mean([r["selected_rankic"] for r in br]),
                "mean_delta_rankic": mean([r["delta_rankic"] for r in br]),
                "seed_block_win_rate": mean([1.0 if r["delta_rankic"] > 0 else 0.0 for r in br]),
                "mean_daily_win_rate": mean([r["daily_win_rate"] for r in br]),
            }
        )
    deltas = [r["delta_rankic"] for r in rows]
    overall = {
        "n_blocks": len(rows),
        "mean_static_rankic": mean([r["static_rankic"] for r in rows]),
        "mean_selected_rankic": mean([r["selected_rankic"] for r in rows]),
        "mean_delta_rankic": mean(deltas),
        "median_delta_rankic": median(deltas),
        "seed_block_win_rate": mean([1.0 if d > 0 else 0.0 for d in deltas]),
        "mean_daily_win_rate": mean([r["daily_win_rate"] for r in rows]),
    }
    return {"overall": overall, "by_block": by_block, "rows": rows}


def build_untyped_formulas(n=240):
    random.seed(20260625)
    fields = [k for k in FLD_REGISTRY.keys()]
    windows = [5, 20, 60]

    def pick():
        return random.choice(fields)

    templates = [
        lambda: f"neg({pick()})",
        lambda: f"abs({pick()})",
        lambda: f"ts_mean({pick()},{random.choice(windows)})",
        lambda: f"ts_zscore({pick()},{random.choice(windows)})",
        lambda: f"ts_rank({pick()},{random.choice(windows)})",
        lambda: f"group_rank({pick()},{pick()})",
        lambda: f"group_zscore({pick()},{pick()})",
        lambda: f"add({pick()},{pick()})",
        lambda: f"sub({pick()},{pick()})",
        lambda: f"mul({pick()},{pick()})",
        lambda: f"trade_when({pick()},{pick()},0.2)",
        lambda: f"adjust_by({pick()},{pick()},0.5)",
        lambda: f"reverse_by({pick()},{pick()})",
        lambda: f"reverse_rank_by({pick()},{pick()})",
        lambda: f"if_else({pick()},{pick()},{pick()})",
    ]
    out = []
    seen = set()
    attempts = 0
    while len(out) < n and attempts < n * 30:
        attempts += 1
        formula = random.choice(templates)()
        if formula in seen:
            continue
        seen.add(formula)
        out.append(formula)
    return out


def evaluate_untyped(af):
    formulas = build_untyped_formulas()
    rows = []
    if UNTYPED_JSON.exists():
        try:
            existing = json.loads(UNTYPED_JSON.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and isinstance(existing.get("rows"), list):
                rows = existing["rows"]
                print(f"resume_untyped_rows {len(rows)}", flush=True)
        except Exception as exc:
            print(f"resume_untyped_failed {exc!r}", flush=True)
    start = time.time()
    for idx, formula in enumerate(formulas, 1):
        if idx <= len(rows):
            continue
        row = {"pool": "untyped_random", "idx": idx, "formula": formula, "status": "ok"}
        try:
            row.update(eval_formula(af, formula))
            row["error"] = ""
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
        rows.append(row)
        if idx % 25 == 0 or idx == len(formulas):
            write_json(UNTYPED_JSON, {"rows": rows})
            write_csv(UNTYPED_CSV, rows)
            ok = sum(1 for r in rows if r["status"] == "ok")
            print(f"untyped_progress {idx}/{len(formulas)} ok={ok} elapsed={time.time() - start:.1f}s", flush=True)
    return rows


def summarize_ablation(main_rows, untyped_rows):
    typed_rows = [r for r in main_rows if r.get("status") == "ok" and r.get("family") != "static"]
    untyped_ok = [r for r in untyped_rows if r.get("status") == "ok"]
    return [
        {
            "pool": "untyped_random",
            "candidate_count": len(untyped_rows),
            "valid_count": len(untyped_ok),
            "valid_rate": len(untyped_ok) / len(untyped_rows) if untyped_rows else float("nan"),
            "mean_os_rankic": mean([r.get("os_rankic") for r in untyped_ok]),
            "median_os_rankic": median([r.get("os_rankic") for r in untyped_ok]),
            "mean_os_rankicir": mean([r.get("os_rankicir") for r in untyped_ok]),
            "survival_os_rankic_gt_001": mean([1.0 if (r.get("os_rankic") or -999) > 0.01 else 0.0 for r in untyped_ok]),
        },
        {
            "pool": "typed_conditional",
            "candidate_count": len([r for r in main_rows if r.get("family") != "static"]),
            "valid_count": len(typed_rows),
            "valid_rate": len(typed_rows) / len([r for r in main_rows if r.get("family") != "static"]),
            "mean_os_rankic": mean([r.get("os_rankic") for r in typed_rows]),
            "median_os_rankic": median([r.get("os_rankic") for r in typed_rows]),
            "mean_os_rankicir": mean([r.get("os_rankicir") for r in typed_rows]),
            "survival_os_rankic_gt_001": mean([1.0 if (r.get("os_rankic") or -999) > 0.01 else 0.0 for r in typed_rows]),
        },
    ]


def main():
    print("v3_no_leak_experiments_start", flush=True)
    started = time.time()
    jobs = build_candidates()
    print(f"main_jobs {len(jobs)}", flush=True)
    af = AlphaFactory()
    rows = evaluate_jobs(af, jobs)
    summary = summarize_main(rows)
    write_json(SUMMARY_JSON, summary)
    print("main_summary", json.dumps(summary["selection_summary"], ensure_ascii=False), flush=True)

    rolling = compute_rolling(af, summary["selected_by_is_rankicir"])
    write_json(ROLLING_JSON, rolling)
    write_csv(ROLLING_CSV, rolling["rows"])
    print("rolling_summary", json.dumps(rolling["overall"], ensure_ascii=False), flush=True)

    untyped_rows = evaluate_untyped(af)
    ablation_summary = summarize_ablation(rows, untyped_rows)
    write_json(UNTYPED_JSON, {"summary": ablation_summary, "rows": untyped_rows})
    write_csv(UNTYPED_CSV, untyped_rows)
    print("ablation_summary", json.dumps(ablation_summary, ensure_ascii=False), flush=True)
    print(f"v3_no_leak_experiments_done elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
