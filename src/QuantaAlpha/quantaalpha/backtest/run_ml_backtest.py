"""
ML backtest with rolling window — LightGBM / XGBoost + Optuna.

Loads factors from a library JSON, filters by quality, builds feature panel
from cached factor values, trains a tree model with Optuna HPO, and evaluates
on a rolling-window test set.

Called by frontend backend app.py via:
    python -m quantaalpha.backtest.run_ml_backtest \
        --factor-json <library.json> --model lgbm --optuna-trials 50 ...
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

import fire
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# Force line-buffered stdout for subprocess visibility
if not sys.stdout.line_buffering:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass


def _print(*args, **kwargs):
    """Print with flush for subprocess visibility."""
    print(*args, **kwargs, flush=True)


# ── paths ────────────────────────────────────────────────────────────
PV_DIR = "/home/workspace/common/quant_data/pv"
UNIV_DIR = "/home/workspace/common/quant_data/universe"
IDX_DIR = "/home/workspace/common/quant_data/index_data"
COMMON_FACTOR_VALUES = "/home/workspace/common/quant/quant_factor/factor_values"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACTOR_CACHE_DIR = Path(
    os.environ.get("FACTOR_CACHE_DIR", str(PROJECT_ROOT / "data" / "results" / "factor_cache"))
)
FACTORLIB_DIR = PROJECT_ROOT / "data" / "factorlib"

COST_ONEWAY = 0.0012
TRADING_DAYS = 243


# ── helpers ──────────────────────────────────────────────────────────
def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_library(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_factors(
    factors: dict[str, dict], quality_filter: str
) -> list[tuple[str, dict]]:
    """Filter factors by quality: high = both train+test passed,
    medium = at least one passed, all = no filter."""
    result = []
    for fid, finfo in factors.items():
        train_ok = bool(finfo.get("train_passed"))
        test_ok = bool(finfo.get("test_passed"))
        if quality_filter == "high" and not (train_ok and test_ok):
            continue
        if quality_filter == "medium" and not (train_ok or test_ok):
            continue
        result.append((fid, finfo))
    return result


def _load_factor_value(finfo: dict) -> pd.DataFrame | None:
    """Try loading factor value from cache or common factor_values dir."""
    name = finfo.get("factor_name", "")
    expr = finfo.get("factor_expression", "")

    # 1) Try expression-based cache
    if expr:
        cache_path = FACTOR_CACHE_DIR / f"{_md5(expr)}.pkl"
        if cache_path.exists():
            try:
                val = pd.read_pickle(cache_path)
                if isinstance(val, pd.Series):
                    val = val.to_frame(name=name or "factor")
                return val
            except Exception:
                pass

    # 2) Try common factor_values by name
    if name:
        pkl_path = Path(COMMON_FACTOR_VALUES) / f"{name}.pkl"
        if pkl_path.exists():
            try:
                val = _load_pickle(pkl_path)
                if isinstance(val, pd.Series):
                    val = val.to_frame(name=name)
                if isinstance(val, pd.DataFrame):
                    return val
            except Exception:
                pass

    return None


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank normalization (vectorized)."""
    arr = X.values.astype(np.float64).copy()
    arr[~np.isfinite(arr)] = np.nan

    # ── vectorized winsorize (1st/99th percentile per row) ──
    valid_mask = ~np.isnan(arr)
    n_valid = valid_mask.sum(axis=1)
    clip_rows = n_valid >= 10
    if clip_rows.any():
        sub = arr[clip_rows]
        lo = np.nanpercentile(sub, 1, axis=1, keepdims=True)
        hi = np.nanpercentile(sub, 99, axis=1, keepdims=True)
        clipped = np.clip(sub, lo, hi)
        clipped[np.isnan(sub)] = np.nan
        arr[clip_rows] = clipped

    # ── vectorized rank normalization per row using pandas ──
    df = pd.DataFrame(arr, index=X.index, columns=X.columns)
    ranked = df.rank(axis=1, method="average", na_option="keep")
    n_valid_2d = valid_mask.sum(axis=1, keepdims=True).astype(np.float64)
    n_valid_2d[n_valid_2d == 0] = 1
    result = (ranked.values - 0.5) / n_valid_2d
    result[~valid_mask] = 0.5
    skip = n_valid < 5
    if skip.any():
        result[skip] = 0.5

    return pd.DataFrame(result, index=X.index, columns=X.columns)


def compute_rolling_window_ranges(
    dates: list[pd.Timestamp] | pd.DatetimeIndex,
    rolling_window: int,
    validation_size: int,
    gap_size: int,
    test_size: int,
    max_windows: int = 0,
) -> list[dict[str, Any]]:
    """Build non-overlapping rolling train/validation/test date ranges."""
    dates_list = list(dates)
    n_dates = len(dates_list)
    test_start_idx = rolling_window + validation_size + gap_size
    windows: list[dict[str, Any]] = []

    while test_start_idx < n_dates:
        if max_windows > 0 and len(windows) >= max_windows:
            break

        train_end_idx = test_start_idx - gap_size - validation_size
        val_start_idx = train_end_idx
        val_end_idx = test_start_idx - gap_size
        test_end_idx = min(test_start_idx + test_size, n_dates)
        train_start_idx = max(0, train_end_idx - rolling_window)

        windows.append({
            "window": len(windows) + 1,
            "train_start_idx": train_start_idx,
            "train_end_idx": train_end_idx,
            "val_start_idx": val_start_idx,
            "val_end_idx": val_end_idx,
            "test_start_idx": test_start_idx,
            "test_end_idx": test_end_idx,
            "train_start_date": dates_list[train_start_idx],
            "train_end_date": dates_list[max(train_start_idx, train_end_idx - 1)],
            "val_start_date": dates_list[val_start_idx],
            "val_end_date": dates_list[max(val_start_idx, val_end_idx - 1)],
            "test_start_date": dates_list[test_start_idx],
            "test_end_date": dates_list[test_end_idx - 1],
        })

        test_start_idx = test_end_idx

    return windows


def compute_fixed_date_range(
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> dict[str, Any]:
    """Build one fixed train/validation/test date range."""
    ts = pd.Timestamp
    window = {
        "window": 1,
        "train_start_date": ts(train_start),
        "train_end_date": ts(train_end),
        "val_start_date": ts(val_start),
        "val_end_date": ts(val_end),
        "test_start_date": ts(test_start),
        "test_end_date": ts(test_end),
        "test_end_idx": 0,
    }
    if not (
        window["train_start_date"]
        <= window["train_end_date"]
        < window["val_start_date"]
        <= window["val_end_date"]
        < window["test_start_date"]
        <= window["test_end_date"]
    ):
        raise ValueError(
            "Fixed date ranges must satisfy: "
            "train_start <= train_end < val_start <= val_end < test_start <= test_end"
        )
    return window


def compute_walk_forward_window_ranges(
    dates: list[pd.Timestamp] | pd.DatetimeIndex,
    train_lookback: int,
    validation_size: int,
    gap_size: int,
    test_size: int,
    test_start: str,
    test_end: str,
    max_windows: int = 0,
) -> list[dict[str, Any]]:
    """Build walk-forward windows using trading-day counts.

    train_lookback is the number of training trading days only; it does not
    include validation or gap days.
    """
    dates_list = list(pd.to_datetime(list(dates)))
    ts_start = pd.Timestamp(test_start)
    ts_end = pd.Timestamp(test_end)
    if ts_start > ts_end:
        raise ValueError("test_start must be <= test_end")

    windows: list[dict[str, Any]] = []
    start_idx = next((i for i, dt in enumerate(dates_list) if dt >= ts_start), None)
    if start_idx is None:
        return windows

    while start_idx < len(dates_list) and dates_list[start_idx] <= ts_end:
        if max_windows > 0 and len(windows) >= max_windows:
            break

        val_end_idx = start_idx - gap_size
        val_start_idx = val_end_idx - validation_size
        train_end_idx = val_start_idx
        train_start_idx = train_end_idx - train_lookback
        if train_start_idx < 0:
            raise ValueError(
                "not enough trading days before test_start for "
                "train_lookback + validation_size + gap_size"
            )

        end_idx = min(start_idx + max(1, test_size), len(dates_list))
        while end_idx > start_idx and dates_list[end_idx - 1] > ts_end:
            end_idx -= 1
        if end_idx <= start_idx:
            break

        windows.append({
            "window": len(windows) + 1,
            "train_start_idx": train_start_idx,
            "train_end_idx": train_end_idx,
            "val_start_idx": val_start_idx,
            "val_end_idx": val_end_idx,
            "test_start_idx": start_idx,
            "test_end_idx": end_idx,
            "train_start_date": dates_list[train_start_idx],
            "train_end_date": dates_list[train_end_idx - 1],
            "val_start_date": dates_list[val_start_idx],
            "val_end_date": dates_list[val_end_idx - 1],
            "test_start_date": dates_list[start_idx],
            "test_end_date": dates_list[end_idx - 1],
        })
        start_idx = end_idx

    return windows


def select_top_n_holding(pred_s: pd.Series, top_n: int) -> set:
    """Select the top N assets by prediction score."""
    if top_n <= 0:
        return set()
    return set(pred_s.dropna().sort_values(ascending=False).head(top_n).index)


# ── core ─────────────────────────────────────────────────────────────
UNIVERSE_MAP = {
    "hs300": ("hs300s.pkl", "hs300s_returns"),
    "zz500": ("zz500s.pkl", "zz500s_returns"),
    "zz800": ("zz800s.pkl", "zz800s_returns"),
    "zz1000": ("zz1000s.pkl", "zz1000s_returns"),
    "sz50": ("sz50s.pkl", "sz50s_returns"),
}


def main(
    factor_json: str,
    model: str = "lgbm",
    optuna_trials: int = 20,
    label_type: str = "return_regression",
    rolling_window: int = 504,
    validation_size: int = 63,
    gap_size: int = 5,
    test_size: int = 63,
    quality_filter: str = "all",
    skip_uncached: bool = True,
    max_windows: int = 0,
    v: bool = False,
    split_mode: str = "rolling",
    top_n: int = 50,
    universe: str = "hs300",
    train_start: str | None = None,
    train_end: str | None = None,
    val_start: str | None = None,
    val_end: str | None = None,
    walk_test_start: str | None = None,
    walk_test_end: str | None = None,
    train_lookback: int | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
):
    """Run ML backtest with rolling window or fixed date split."""
    import lightgbm as lgb  # noqa: delayed import
    import xgboost as xgb
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    verbose = v
    factor_json_path = Path(factor_json)
    if not factor_json_path.is_absolute():
        candidate = FACTORLIB_DIR / factor_json
        factor_json_path = candidate if candidate.exists() else factor_json_path

    _print(f"加载因子库: {factor_json_path}")
    lib = _load_library(factor_json_path)
    all_factors = lib.get("factors", {})
    filtered = _filter_factors(all_factors, quality_filter)
    _print(f"因子总数: {len(all_factors)}, 质量过滤后: {len(filtered)} (filter={quality_filter})")

    if not filtered:
        print("WARNING: 过滤后无因子可用，切换为全部因子")
        filtered = list(all_factors.items())

    # ── load factor values ───────────────────────────────────────────
    print("\n加载因子数据...")
    factor_dfs: dict[str, pd.DataFrame] = {}
    skipped = 0
    for fid, finfo in filtered:
        name = finfo.get("factor_name", fid)
        val = _load_factor_value(finfo)
        if val is None:
            skipped += 1
            if verbose:
                _print(f"  跳过 {name}: 无缓存数据")
            continue
        val.index = pd.to_datetime(val.index)
        factor_dfs[name] = val
        if verbose and len(factor_dfs) % 10 == 0:
            _print(f"  已加载 {len(factor_dfs)} 个因子...")

    _print(f"成功加载: {len(factor_dfs)} 个因子, 跳过: {skipped} (无缓存)")

    # If too few factors from library, supplement with common factor_values
    if len(factor_dfs) < 5:
        common_dir = Path(COMMON_FACTOR_VALUES)
        if common_dir.exists():
            _print(f"\n因子库缓存不足, 从公共因子池补充 ({common_dir})...")
            common_pkls = sorted(common_dir.glob("*.pkl"))
            added = 0
            for pkl in common_pkls:
                fname = pkl.stem
                if fname in factor_dfs:
                    continue
                try:
                    val = _load_pickle(pkl)
                    if isinstance(val, pd.Series):
                        val = val.to_frame(name=fname)
                    if isinstance(val, pd.DataFrame) and val.shape[0] > 100:
                        val.index = pd.to_datetime(val.index)
                        factor_dfs[fname] = val
                        added += 1
                except Exception:
                    pass
            _print(f"  从公共因子池补充: {added} 个因子")

    _print(f"最终可用因子: {len(factor_dfs)}")

    if len(factor_dfs) < 3:
        msg = f"可用因子不足 ({len(factor_dfs)}), 需要至少3个因子有缓存数据。请先运行因子挖掘生成缓存，或选择已有缓存的因子库 (如 all_factors_library.json)"
        _print(f"ERROR: {msg}")
        sys.exit(1)

    # ── load market data ─────────────────────────────────────────────
    univ_file, bm_col = UNIVERSE_MAP.get(universe, UNIVERSE_MAP["hs300"])
    _print(f"\n加载市场数据 (股票池: {universe})...")
    univ_mask = _load_pickle(f"{UNIV_DIR}/{univ_file}")
    univ_mask.index = pd.to_datetime(univ_mask.index)

    closes = _load_pickle(f"{PV_DIR}/hfq_closes.pkl")
    closes.index = pd.to_datetime(closes.index)
    fwd_ret = closes.pct_change(5).shift(-5)

    ctc_ret = _load_pickle(f"{PV_DIR}/ctc_returns.pkl")
    ctc_ret.index = pd.to_datetime(ctc_ret.index)

    idx_data = _load_pickle(f"{IDX_DIR}/index_data.pkl")
    idx_data.index = pd.to_datetime(idx_data.index)
    index_bm = idx_data[bm_col].dropna()

    all_dates = sorted(fwd_ret.index)
    stocks = fwd_ret.columns
    _print(f"市场数据: {len(all_dates)} 交易日, {len(stocks)} 只股票")

    # ── build panel ──────────────────────────────────────────────────
    print("\n构建特征面板...")
    feat_names = sorted(factor_dfs.keys())
    feat_series = []
    for name in feat_names:
        df = factor_dfs[name]
        if isinstance(df, pd.DataFrame) and df.shape[1] == 1:
            s = df.iloc[:, 0]
        elif isinstance(df, pd.DataFrame):
            s = df.reindex(columns=stocks).stack(dropna=False)
            s.name = name
            feat_series.append(s)
            continue
        else:
            s = df
        s = s.reindex(index=pd.to_datetime(fwd_ret.index), columns=stocks) if hasattr(s, "reindex") else s
        feat_series.append(s if isinstance(s, pd.Series) else s.stack(dropna=False).rename(name))

    panel_parts = []
    for name in feat_names:
        df = factor_dfs[name]
        col_data = df.reindex(index=fwd_ret.index, columns=stocks).ffill(limit=5).stack(dropna=False)
        col_data.name = name
        panel_parts.append(col_data)

    raw_fwd = fwd_ret.reindex(index=fwd_ret.index, columns=stocks).stack(dropna=False)
    raw_fwd.name = "fwd_raw"

    is_cls = label_type in ("direction_classification", "top5_classification")

    if label_type == "direction_classification":
        y = (raw_fwd > 0).astype(int)
        y.name = "fwd"
        _print(f"标签类型: 涨跌分类 (正样本比例 {y.mean():.2%})")
    elif label_type == "top5_classification":
        y = raw_fwd.groupby(level=0).rank(pct=True).gt(0.95).astype(int)
        y.name = "fwd"
        _print(f"标签类型: Top5%分类 (正样本比例 {y.mean():.2%})")
    else:
        y = raw_fwd.copy()
        y.name = "fwd"

    panel = pd.concat(panel_parts + [y, raw_fwd], axis=1)

    hs_mask = univ_mask.reindex(index=fwd_ret.index, columns=stocks).fillna(False).astype(bool).stack()
    panel = panel.loc[hs_mask[hs_mask].index]
    panel = panel.dropna(subset=["fwd", "fwd_raw"])
    panel = panel.dropna(thresh=max(1, len(feat_names) // 5))

    _print(f"面板大小: {panel.shape}, 特征数: {len(feat_names)}")

    # ── rolling window backtest ──────────────────────────────────────
    _print(
        f"\n开始滚动窗口回测 "
        f"(model={model}, train={rolling_window}, val={validation_size}, gap={gap_size}, test={test_size})"
    )
    dates_in_panel = sorted(panel.index.get_level_values(0).unique())
    n_dates = len(dates_in_panel)

    window_results = []
    all_daily_preds = {}
    prev_holding: set = set()

    daily_long_net: dict[pd.Timestamp, float] = {}
    daily_bm: dict[pd.Timestamp, float] = {}
    daily_ic: dict[pd.Timestamp, float] = {}
    daily_selected_stocks: dict[pd.Timestamp, list[str]] = {}

    if split_mode == "walk_forward" and walk_test_start and walk_test_end:
        _print(f"【Walk-Forward模式】")
        _print(f"  测试期: {walk_test_start} ~ {walk_test_end}")
        _print(f"  训练期: {train_lookback or rolling_window} 交易日 | 验证: {validation_size} 交易日 | Gap: {gap_size} 交易日 | 测试步长: {test_size} 交易日")
        try:
            windows = compute_walk_forward_window_ranges(
                dates_in_panel,
                train_lookback=train_lookback or rolling_window,
                validation_size=validation_size,
                gap_size=gap_size,
                test_size=max(1, test_size),
                test_start=walk_test_start,
                test_end=walk_test_end,
                max_windows=max_windows,
            )
        except ValueError as exc:
            _print(f"ERROR: {exc}")
            sys.exit(1)
    elif split_mode == "fixed" and train_start and train_end and val_start and val_end and test_start and test_end:
        _print(f"使用固定日期分割模式")
        _print(f"  训练: {train_start} ~ {train_end}")
        _print(f"  验证: {val_start} ~ {val_end}")
        _print(f"  测试: {test_start} ~ {test_end}")
        try:
            fixed_window = compute_fixed_date_range(
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end,
            )
        except ValueError as exc:
            _print(f"ERROR: {exc}")
            sys.exit(1)
        fixed_window["test_end_idx"] = len(dates_in_panel)
        windows = [fixed_window]
    else:
        min_start = rolling_window + validation_size + gap_size
        min_required = min_start + max(1, test_size)
        if min_required > n_dates:
            _print(f"ERROR: 数据不足 ({n_dates} 天 < 需要 {min_required} 天)")
            sys.exit(1)

        windows = compute_rolling_window_ranges(
            dates_in_panel,
            rolling_window=rolling_window,
            validation_size=validation_size,
            gap_size=gap_size,
            test_size=max(1, test_size),
            max_windows=max_windows,
        )

    for window in windows:
        window_num = window["window"]
        test_end_idx = window["test_end_idx"]

        train_start_date = window["train_start_date"]
        train_end_date = window["train_end_date"]
        val_start_date = window["val_start_date"]
        val_end_date = window["val_end_date"]
        test_start_date = window["test_start_date"]
        test_end_date = window["test_end_date"]

        idx = panel.index.get_level_values(0)
        tr = panel[(idx >= train_start_date) & (idx <= train_end_date)]
        vl = panel[(idx >= val_start_date) & (idx <= val_end_date)]
        te = panel[(idx >= test_start_date) & (idx <= test_end_date)]

        if len(tr) < 100 or len(vl) < 50 or len(te) < 10:
            continue

        X_tr = preprocess_features(tr[feat_names])
        y_tr = tr["fwd"]
        X_vl = preprocess_features(vl[feat_names])
        y_vl = vl["fwd"]
        X_te = preprocess_features(te[feat_names])
        y_te = te["fwd"]

        # Pre-convert to numpy once — avoids repeated DataFrame→array overhead in every trial
        X_tr_np = X_tr.values.astype(np.float32)
        y_tr_np = y_tr.values.astype(np.float32)
        X_vl_np = X_vl.values.astype(np.float32)
        y_vl_np = y_vl.values.astype(np.float32)

        # Pre-compute validation groupby indices for IC calculation
        vl_raw = vl["fwd_raw"].values if "fwd_raw" in vl.columns else y_vl.values
        vl_date_idx = y_vl.index.get_level_values(0)
        vl_groups = {}
        for dt in vl_date_idx.unique():
            mask = np.asarray(vl_date_idx == dt)
            if mask.sum() >= 10:
                vl_groups[dt] = (mask, vl_raw[mask])

        _print(f"\n滚动窗口 #{window_num}: 训练 {train_start_date.date()}~{train_end_date.date()} "
              f"验证 {val_start_date.date()}~{val_end_date.date()} "
              f"测试 {test_start_date.date()}~{test_end_date.date()}")
        _print(f"  样本: train={len(tr)}, val={len(vl)}, test={len(te)}")

        # ── Optuna HPO ───────────────────────────────────────────────
        if model == "lgbm":
            lgb_train = lgb.Dataset(X_tr_np, label=y_tr_np, free_raw_data=False)
            lgb_valid = lgb.Dataset(X_vl_np, label=y_vl_np, free_raw_data=False, reference=lgb_train)

        def objective(trial):
            if model == "lgbm":
                p = dict(
                    num_leaves=trial.suggest_int("num_leaves", 8, 96),
                    learning_rate=trial.suggest_float("lr", 0.008, 0.15, log=True),
                    max_depth=trial.suggest_int("max_depth", 3, 6),
                    min_child_samples=trial.suggest_int("min_child", 30, 250),
                    subsample=trial.suggest_float("subsample", 0.5, 0.95),
                    colsample_bytree=trial.suggest_float("colsample", 0.3, 0.85),
                    reg_alpha=trial.suggest_float("reg_alpha", 0.01, 3.0, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 0.05, 5.0, log=True),
                    n_estimators=trial.suggest_int("n_estimators", 50, 350),
                    random_state=42, verbosity=-1, n_jobs=-1,
                    feature_pre_filter=False,
                    objective="binary" if is_cls else "regression",
                    metric="binary_logloss" if is_cls else "l2",
                )
                booster = lgb.train(
                    p, lgb_train, num_boost_round=p.pop("n_estimators"),
                    valid_sets=[lgb_valid], callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
                )
                raw_pred = booster.predict(X_vl_np)
                pred = raw_pred
            else:
                p = dict(
                    max_depth=trial.suggest_int("max_depth", 3, 6),
                    learning_rate=trial.suggest_float("lr", 0.008, 0.15, log=True),
                    n_estimators=trial.suggest_int("n_estimators", 50, 350),
                    min_child_weight=trial.suggest_int("min_child_weight", 5, 50),
                    subsample=trial.suggest_float("subsample", 0.5, 0.95),
                    colsample_bytree=trial.suggest_float("colsample", 0.3, 0.85),
                    reg_alpha=trial.suggest_float("reg_alpha", 0.01, 3.0, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 0.05, 5.0, log=True),
                    random_state=42, verbosity=0, n_jobs=-1, tree_method="hist",
                )
                dtrain = xgb.DMatrix(X_tr_np, label=y_tr_np)
                dvalid = xgb.DMatrix(X_vl_np, label=y_vl_np)
                xgb_params = {k: v for k, v in p.items() if k not in ("n_estimators", "random_state")}
                xgb_params["seed"] = 42
                xgb_params["objective"] = "binary:logistic" if is_cls else "reg:squarederror"
                booster = xgb.train(
                    xgb_params, dtrain, num_boost_round=p["n_estimators"],
                    evals=[(dvalid, "val")], early_stopping_rounds=30, verbose_eval=False,
                )
                pred = booster.predict(dvalid)

            # Vectorized IC calculation using pre-computed groups
            ics = []
            for dt, (mask, fwd_vals) in vl_groups.items():
                p_sub = pred[mask]
                corr, _ = spearmanr(p_sub, fwd_vals)
                if not np.isnan(corr):
                    ics.append(corr)
            return float(np.mean(ics)) if ics else 0.0

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=optuna_trials, show_progress_bar=False)
        _print(f"  Optuna best val IC: {study.best_value:.4f}")

        # ── train final model ────────────────────────────────────────
        bp = study.best_params
        if model == "lgbm":
            final_params = dict(
                num_leaves=bp["num_leaves"], learning_rate=bp["lr"],
                max_depth=bp["max_depth"], min_child_samples=bp["min_child"],
                subsample=bp["subsample"], colsample_bytree=bp["colsample"],
                reg_alpha=bp["reg_alpha"], reg_lambda=bp["reg_lambda"],
                random_state=42, verbosity=-1, n_jobs=-1,
                feature_pre_filter=False,
                objective="binary" if is_cls else "regression",
                metric="binary_logloss" if is_cls else "l2",
            )
            final_booster = lgb.train(
                final_params, lgb_train, num_boost_round=bp["n_estimators"],
                valid_sets=[lgb_valid], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
            )
        else:
            xgb_params = {
                "max_depth": bp["max_depth"], "learning_rate": bp["lr"],
                "min_child_weight": bp["min_child_weight"],
                "subsample": bp["subsample"], "colsample_bytree": bp["colsample"],
                "reg_alpha": bp["reg_alpha"], "reg_lambda": bp["reg_lambda"],
                "seed": 42, "verbosity": 0, "nthread": -1, "tree_method": "hist",
                "objective": "binary:logistic" if is_cls else "reg:squarederror",
            }
            dtrain = xgb.DMatrix(X_tr_np, label=y_tr_np)
            dvalid = xgb.DMatrix(X_vl_np, label=y_vl_np)
            final_booster = xgb.train(
                xgb_params, dtrain, num_boost_round=bp["n_estimators"],
                evals=[(dvalid, "val")], early_stopping_rounds=50, verbose_eval=False,
            )

        # ── test predictions ─────────────────────────────────────────
        X_te_np = X_te.values.astype(np.float32)
        if model == "lgbm":
            pred_te = final_booster.predict(X_te_np)
        else:
            pred_te = final_booster.predict(xgb.DMatrix(X_te_np))
        te_with_pred = te[["fwd_raw"]].copy() if "fwd_raw" in te.columns else te[["fwd"]].copy()
        te_with_pred.columns = ["fwd"]
        te_with_pred["pred"] = pred_te

        for dt in sorted(te.index.get_level_values(0).unique()):
            g = te_with_pred.xs(dt, level=0)
            if len(g) < 10:
                continue

            ic_val = g["pred"].corr(g["fwd"], method="spearman")
            if not np.isnan(ic_val):
                daily_ic[dt] = float(ic_val)

            pred_s = g["pred"]
            cur_holding = select_top_n_holding(pred_s, top_n)
            daily_selected_stocks[dt] = list(pred_s.reindex(list(cur_holding)).sort_values(ascending=False).index)

            if dt in ctc_ret.index and len(cur_holding) > 0:
                gross_ret = float(ctc_ret.loc[dt].reindex(list(cur_holding)).dropna().mean())
                turnover = 1 - len(cur_holding & prev_holding) / max(len(cur_holding), 1) if prev_holding else 1.0
                cost = turnover * COST_ONEWAY * 2
                daily_long_net[dt] = gross_ret - cost
                daily_bm[dt] = float(index_bm.get(dt, 0))

            prev_holding = cur_holding

        win_ic = te_with_pred.groupby(level=0).apply(
            lambda g: g["pred"].corr(g["fwd"], method="spearman") if len(g) >= 10 else 0
        ).dropna()
        _print(f"  窗口测试 IC: {win_ic.mean():.4f} (±{win_ic.std():.4f})")

        window_results.append({
            "window": window_num,
            "test_start": str(test_start_date.date()),
            "test_end": str(test_end_date.date()),
            "avg_ic": round(float(win_ic.mean()), 4),
            "best_val_ic": round(study.best_value, 4),
        })

    # ── aggregate results ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)

    if not daily_ic:
        print("ERROR: 无有效测试结果")
        sys.exit(1)

    ic_series = pd.Series(daily_ic).sort_index()
    test_days = len(ic_series)
    avg_ic = float(ic_series.mean())
    ic_std = float(ic_series.std()) if test_days > 1 else 0.0
    icir = float(avg_ic / ic_std * math.sqrt(TRADING_DAYS)) if ic_std > 0 else 0

    long_net_s = pd.Series(daily_long_net).sort_index()
    bm_s = pd.Series(daily_bm).sort_index()
    exc = long_net_s - bm_s

    cum_long_net = (1 + long_net_s.fillna(0)).cumprod()
    cum_bm = (1 + bm_s.fillna(0)).cumprod()
    cum_exc = (1 + exc.fillna(0)).cumprod()

    if test_days >= 20:
        ann_port_net = float(long_net_s.mean() * TRADING_DAYS)
        ann_bm = float(bm_s.mean() * TRADING_DAYS)
        ann_exc = float(exc.mean() * TRADING_DAYS)
    else:
        ann_port_net = float(cum_long_net.iloc[-1] - 1)
        ann_bm = float(cum_bm.iloc[-1] - 1)
        ann_exc = float(cum_exc.iloc[-1] - 1)

    dd_exc = (cum_exc - cum_exc.cummax()) / cum_exc.cummax()
    mdd_exc = float(abs(dd_exc.min())) if len(dd_exc) > 1 else 0
    dd_port = (cum_long_net - cum_long_net.cummax()) / cum_long_net.cummax()
    mdd_port = float(abs(dd_port.min())) if len(dd_port) > 1 else 0
    ir_exc = float(exc.mean() / exc.std() * math.sqrt(TRADING_DAYS)) if test_days > 1 and exc.std() > 0 else 0
    sharpe = float(long_net_s.mean() / long_net_s.std() * math.sqrt(TRADING_DAYS)) if test_days > 1 and long_net_s.std() > 0 else 0

    # Feature importance (from last window's booster)
    if model == "lgbm":
        fi = pd.Series(
            final_booster.feature_importance(importance_type="gain"),
            index=feat_names,
        ).sort_values(ascending=False)
    else:
        score = final_booster.get_score(importance_type="gain")
        fi = pd.Series(
            {feat_names[int(k[1:])]: v for k, v in score.items()} if score else {f: 0 for f in feat_names}
        ).reindex(feat_names, fill_value=0).sort_values(ascending=False)

    _print(f"模型: {model.upper()}")
    _print(f"因子数: {len(feat_names)}")
    _print(f"滚动窗口数: {len(window_results)}")
    _print(f"测试天数: {test_days}")
    _print(f"测试期: {ic_series.index[0].date()} ~ {ic_series.index[-1].date()}")
    _print(f"平均IC: {avg_ic:.4f}  ICIR: {icir:.3f}")
    ret_label = "年化收益(扣费)" if test_days >= 20 else "累计收益(扣费)"
    _print(f"{ret_label}: {ann_port_net:.2%}")
    exc_label = "年化超额" if test_days >= 20 else "累计超额"
    _print(f"{exc_label}: {ann_exc:.2%}")
    _print(f"超额最大回撤: {mdd_exc:.3f}")
    _print(f"超额IR: {ir_exc:.3f}")
    _print(f"Sharpe: {sharpe:.3f}")

    # cumulative excess curve for frontend chart
    cum_exc_curve = []
    for dt, v in cum_exc.items():
        cum_exc_curve.append({"date": str(dt.date()), "value": round(float(v - 1), 6)})

    # win rate (days with positive excess return) — only meaningful with enough days
    win_rate = float((exc > 0).sum() / len(exc)) if test_days >= 5 else 0

    # calmar = ann_excess / mdd_excess — only meaningful when annualized and mdd > 0
    calmar = float(ann_exc / mdd_exc) if test_days >= 20 and mdd_exc > 0 else 0

    result = {
        "model": model,
        "n_factors": len(feat_names),
        "factor_names": feat_names[:50],
        "n_windows": len(window_results),
        "quality_filter": quality_filter,
        "split_mode": split_mode,
        "top_n": top_n,
        "train_lookback": train_lookback or rolling_window,
        "test_days": test_days,
        "test_period": f"{ic_series.index[0].date()} ~ {ic_series.index[-1].date()}",
        # frontend-compatible metric keys
        "IC": round(avg_ic, 4),
        "ICIR": round(icir, 3),
        "RankIC": round(avg_ic, 4),
        "RankICIR": round(icir, 3),
        "annualized_return": round(ann_port_net, 4),
        "annualized_excess": round(ann_exc, 4),
        "annualized_benchmark": round(ann_bm, 4),
        "max_drawdown": round(mdd_port, 3),
        "max_drawdown_excess": round(mdd_exc, 3),
        "sharpe_ratio": round(sharpe, 3),
        "information_ratio": round(ir_exc, 3),
        "calmar_ratio": round(calmar, 3),
        "win_rate": round(win_rate, 4),
        "cumulative_curve": cum_exc_curve,
        "daily_selected_stocks": [
            {"date": str(dt.date()), "stocks": stocks}
            for dt, stocks in sorted(daily_selected_stocks.items())
        ],
        "cost_per_side": COST_ONEWAY,
        "universe": universe,
        "rolling_window": rolling_window,
        "validation_size": validation_size,
        "gap_size": gap_size,
        "test_size": test_size,
        "optuna_trials": optuna_trials,
        "window_details": window_results,
        "feature_importance_top10": {k: int(v) for k, v in fi.head(10).items()},
    }

    out_dir = PROJECT_ROOT / "data" / "ml_backtest_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ml_backtest_{model}_{len(feat_names)}f.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _print(f"\n结果已保存: {out_path}")

    _print("__ML_RESULT__:" + json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    fire.Fire(main)
