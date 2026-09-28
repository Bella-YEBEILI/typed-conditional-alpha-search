import os
import warnings

os.environ.setdefault("NUMBA_NUM_THREADS", "12")

import numpy as np
from numba import njit, prange

from quant.quant_lib.minute_tools import MinuteFactorEngine
from quant.quant_lib import minute_ops as _minute_ops  # noqa: F401
from quant.quant_lib.gp_crossday_features import (
    CROSSDAY_ANOMALY_NAMES,
    build_crossday_anomaly_panels_from_h5,
)
from quant.quant_lib.minute_ops import (
    ops_corr,
    ops_kurt,
    ops_max,
    ops_mean,
    ops_min,
    ops_skew,
    ops_std,
    ops_sum,
    ops_wmean,
)


IMPL_VERSION = "20260505_gp_extra_ops_v9"
EPS = np.float32(1e-8)
OPERATOR_NAME = "gp_parametric_formula"
OPERATOR_PARAM_ORDER = (
    "impl_version",
    "input_names",
    "a_name",
    "b_name",
    "window",
    "slice_value",
    "mask_field",
    "mask_rule",
    "mode",
    "op_name",
    "lag",
)
OUTER_OPS = ("identity", "ts_zscore_20", "ts_diff_1", "ts_rank_60", "ts_pctchange_5", "ts_neg")

BASE_FIELDS = [
    "opens",
    "highs",
    "lows",
    "closes",
    "volumes",
    "amounts",
    "returns",
    "turnovers",
    "vwaps",
    "ratios",
    "lreturns",
]
INDICATOR_NAMES = [
    "opens",
    "highs",
    "lows",
    "closes",
    "volumes",
    "amounts",
    "returns",
    "turnovers",
    "vwaps",
    "ratios",
    "lreturns",
    "spread",
    "mid_price",
    "typical_price",
    "money_flow",
    "return_abs",
    "log_volume",
    "log_amount",
    "price_range_pct",
    "upper_shadow",
    "lower_shadow",
    "body",
    "vwap_deviation",
    "volume_intensity",
    "cum_return",
    "amt_per_trade_proxy",
    "amihud",
    "kyle_lambda",
    "signed_volume",
    "vol_accel",
    "rv_ratio",
    "order_imbalance",
    "path_high_time",
    "path_low_time",
    "path_high_to_close",
    "path_low_rebound",
    "path_efficiency",
    "gate_liq_high_amount",
    "gate_rev_high_rv",
    "gate_rev_pm_disagree",
    "gate_rev_high_oimb",
    "anom_amihud_z20",
    "anom_tail_vol_share_z20",
    "anom_high_to_close_z20",
    "anom_order_imbalance_z20",
    "anom_path_efficiency_z20",
]
INPUT_FIELD_ORDER = BASE_FIELDS + list(CROSSDAY_ANOMALY_NAMES)
MASK_RULES = [
    "none",
    "high_0.3",
    "high_0.5",
    "high_0.6",
    "high_0.7",
    "high_0.8",
    "low_0.3",
    "low_0.5",
    "low_0.7",
]
MODE1_OPS = [
    "mean",
    "std",
    "skew",
    "kurt",
    "median",
    "sum",
    "max",
    "min",
    "range",
    "first",
    "last",
    "ret",
    "autocorr",
    "entropy",
    "zero_cross",
]
MODE2_OPS = [
    "corr",
    "slope",
    "intercept",
    "r2",
    "euc_dist",
    "cov",
    "cos_sim",
    "wmean",
    "rank_corr",
    "mutual_info",
    "beta",
    "residual_std",
    "max_corr",
    "tail_corr",
    "diff_mean",
    "diff_std",
    "ratio_mean",
    "ratio_std",
]
WINDOW_CHOICES = [15, 30, 45, 60, 90, 120, 180, 238, 240]
SLICE_CHOICES = [None, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
B_SHIFT_CHOICES = [-5, -3, -2, -1, 0, 1, 2, 3, 5]

INDICATOR_DEPENDENCIES = {
    "opens": ("opens",),
    "highs": ("highs",),
    "lows": ("lows",),
    "closes": ("closes",),
    "volumes": ("volumes",),
    "amounts": ("amounts",),
    "returns": ("returns",),
    "turnovers": ("turnovers",),
    "vwaps": ("vwaps",),
    "ratios": ("ratios",),
    "lreturns": ("lreturns",),
    "spread": ("highs", "lows"),
    "mid_price": ("highs", "lows"),
    "typical_price": ("highs", "lows", "closes"),
    "money_flow": ("highs", "lows", "closes", "volumes"),
    "return_abs": ("returns",),
    "log_volume": ("volumes",),
    "log_amount": ("amounts",),
    "price_range_pct": ("opens", "highs", "lows"),
    "upper_shadow": ("highs", "opens", "closes"),
    "lower_shadow": ("lows", "opens", "closes"),
    "body": ("opens", "closes"),
    "vwap_deviation": ("closes", "vwaps"),
    "volume_intensity": ("volumes",),
    "cum_return": ("returns",),
    "amt_per_trade_proxy": ("amounts", "volumes"),
    "amihud": ("returns", "volumes", "closes"),
    "kyle_lambda": ("closes", "opens", "volumes"),
    "signed_volume": ("closes", "opens", "volumes"),
    "vol_accel": ("volumes",),
    "rv_ratio": ("returns",),
    "order_imbalance": ("returns", "volumes"),
    "path_high_time": ("highs",),
    "path_low_time": ("lows",),
    "path_high_to_close": ("opens", "highs", "closes"),
    "path_low_rebound": ("opens", "lows", "closes"),
    "path_efficiency": ("opens", "closes"),
    "gate_liq_high_amount": ("returns", "amounts"),
    "gate_rev_high_rv": ("opens", "closes", "returns"),
    "gate_rev_pm_disagree": ("opens", "closes"),
    "gate_rev_high_oimb": ("opens", "closes", "returns", "volumes"),
    "anom_amihud_z20": ("anom_amihud_z20",),
    "anom_tail_vol_share_z20": ("anom_tail_vol_share_z20",),
    "anom_high_to_close_z20": ("anom_high_to_close_z20",),
    "anom_order_imbalance_z20": ("anom_order_imbalance_z20",),
    "anom_path_efficiency_z20": ("anom_path_efficiency_z20",),
}
DIM_GROUPS = {
    "opens": "PRICE",
    "highs": "PRICE",
    "lows": "PRICE",
    "closes": "PRICE",
    "vwaps": "PRICE",
    "mid_price": "PRICE",
    "typical_price": "PRICE",
    "spread": "PRICE",
    "upper_shadow": "PRICE",
    "lower_shadow": "PRICE",
    "body": "PRICE",
    "vwap_deviation": "PRICE",
    "volumes": "VOLUME",
    "amounts": "VOLUME",
    "money_flow": "VOLUME",
    "log_volume": "VOLUME",
    "log_amount": "VOLUME",
    "volume_intensity": "VOLUME",
    "amt_per_trade_proxy": "VOLUME",
    "returns": "RETURN",
    "lreturns": "RETURN",
    "return_abs": "RETURN",
    "cum_return": "RETURN",
    "turnovers": "RATIO",
    "ratios": "RATIO",
    "price_range_pct": "RATIO",
    "path_high_time": "PATH",
    "path_low_time": "PATH",
    "path_high_to_close": "PATH",
    "path_low_rebound": "PATH",
    "path_efficiency": "PATH",
    "gate_liq_high_amount": "GATED",
    "gate_rev_high_rv": "GATED",
    "gate_rev_pm_disagree": "GATED",
    "gate_rev_high_oimb": "GATED",
    "anom_amihud_z20": "ANOMALY",
    "anom_tail_vol_share_z20": "ANOMALY",
    "anom_high_to_close_z20": "ANOMALY",
    "anom_order_imbalance_z20": "ANOMALY",
    "anom_path_efficiency_z20": "ANOMALY",
}
DAILY_STATE_GROUPS = {"PATH", "GATED", "ANOMALY"}
DAILY_STATE_INDICATORS = {
    name for name, group in DIM_GROUPS.items()
    if group in DAILY_STATE_GROUPS
}
SCALE_SENSITIVE_OPS = {
    "diff_mean",
    "diff_std",
    "slope",
    "intercept",
    "cov",
    "wmean",
    "beta",
    "residual_std",
}

_INDICATOR_SET = set(INDICATOR_NAMES)
_MODE1_SET = set(MODE1_OPS)
_MODE2_SET = set(MODE2_OPS)
_MASK_SET = set(MASK_RULES)
_CROSSDAY_PANEL_CACHE = {}


def _try_read_daily_from_h5(h5_path: str, date_end: int):
    """Fast path: read precomputed daily/<name> panels written by日更.

    Returns dict[name -> ndarray (date_end, n_stocks)] when h5 已落地 daily group, else None.
    """
    try:
        import h5py
        with h5py.File(h5_path, "r") as f:
            if "daily" not in f:
                return None
            grp = f["daily"]
            if any(name not in grp for name in CROSSDAY_ANOMALY_NAMES):
                return None
            panels = {}
            for name in CROSSDAY_ANOMALY_NAMES:
                ds = grp[name]
                n = min(int(date_end), ds.shape[0])
                panels[name] = ds[:n].astype(np.float32)
            return panels
    except Exception:
        return None


def _get_crossday_panels_for_range(h5_path: str, date_end: int) -> dict[str, np.ndarray]:
    key = (os.path.abspath(h5_path), int(date_end))
    cached = _CROSSDAY_PANEL_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_CROSSDAY_PANEL_CACHE) >= 2:
        _CROSSDAY_PANEL_CACHE.clear()
    panels = _try_read_daily_from_h5(h5_path, date_end)
    if panels is None:
        panels = build_crossday_anomaly_panels_from_h5(
            h5_path=h5_path,
            day_indices=np.arange(int(date_end), dtype=np.int64),
            stock_idx=None,
            chunk_days=32,
            window=20,
        )
    _CROSSDAY_PANEL_CACHE[key] = panels
    return panels


def _register_crossday_provider(name: str):
    @MinuteFactorEngine.register_input_provider(name)
    def _provider(*, h5_path: str, field: str, date_start: int, date_end: int, minute_start: int, minute_end: int):
        del minute_start, minute_end
        panels = _get_crossday_panels_for_range(h5_path, date_end)
        return panels[field][date_start:date_end]
    return _provider


for _crossday_name in CROSSDAY_ANOMALY_NAMES:
    _register_crossday_provider(_crossday_name)
del _crossday_name


def decode_params(params: list[int] | tuple[int, ...]) -> dict:
    if len(params) not in (10, 11):
        raise ValueError("params must have length 10 or 11")
    mode = int(params[6]) + 1
    outer_idx = int(params[10]) if len(params) >= 11 else 0
    if outer_idx < 0 or outer_idx >= len(OUTER_OPS):
        outer_idx = 0
    return {
        "a_name": INDICATOR_NAMES[int(params[0])],
        "b_name": INDICATOR_NAMES[int(params[1])] if mode == 2 else "",
        "window": int(WINDOW_CHOICES[int(params[2])]),
        "slice_value": SLICE_CHOICES[int(params[3])],
        "mask_field": INDICATOR_NAMES[int(params[4])],
        "mask_rule": MASK_RULES[int(params[5])],
        "mode": mode,
        "op_name": MODE1_OPS[int(params[7])] if mode == 1 else MODE2_OPS[int(params[8])],
        "lag": 0 if mode == 1 else int(B_SHIFT_CHOICES[int(params[9])]),
        "outer_op": OUTER_OPS[outer_idx],
    }


def parse_formula(formula: str) -> dict:
    text = formula.strip()
    if not (text.startswith("f(") and text.endswith(")")):
        raise ValueError(f"invalid formula: {formula}")
    body = text[2:-1]
    items = {}
    for part in body.split(","):
        key, value = part.split("=", 1)
        items[key.strip()] = value.strip()
    mode = int(items["mode"])
    slice_text = items["slice"]
    outer_text = items.get("outer", "identity")
    return {
        "a_name": items["A"],
        "b_name": items.get("B", "") if mode == 2 else "",
        "window": int(items["window"]),
        "slice_value": None if slice_text == "None" else float(slice_text),
        "mask_field": items["mask"].split(":", 1)[0],
        "mask_rule": items["mask"].split(":", 1)[1],
        "mode": mode,
        "op_name": items["op"],
        "lag": 0 if mode == 1 else int(items.get("lag", "0")),
        "outer_op": outer_text if outer_text in OUTER_OPS else "identity",
    }


def required_inputs_for_spec(spec: dict) -> list[str]:
    required = set(INDICATOR_DEPENDENCIES[spec["a_name"]])
    if int(spec["mode"]) == 2 and spec.get("b_name"):
        required.update(INDICATOR_DEPENDENCIES[spec["b_name"]])
    if spec["mask_rule"] != "none":
        required.update(INDICATOR_DEPENDENCIES[spec["mask_field"]])
    return [name for name in INPUT_FIELD_ORDER if name in required]


def apply_daily_outer(daily_df, outer_op: str):
    """Apply a causal time-series outer transform on a daily factor DataFrame (index=date)."""
    import pandas as pd
    if daily_df is None or outer_op == "identity" or outer_op is None:
        return daily_df
    if outer_op == "ts_neg":
        return -daily_df
    if outer_op == "ts_diff_1":
        return daily_df.diff(1)
    if outer_op == "ts_pctchange_5":
        base = daily_df.shift(5)
        safe_base = base.where(base.abs() > 1e-3)
        out = daily_df / safe_base - 1.0
        return out.replace([np.inf, -np.inf], np.nan)
    if outer_op == "ts_zscore_20":
        rolling = daily_df.rolling(window=20, min_periods=10)
        mean = rolling.mean()
        std = rolling.std(ddof=1)
        std = std.where(std > 1e-8, 1.0)
        return (daily_df - mean) / std
    if outer_op == "ts_rank_60":
        return daily_df.rolling(window=60, min_periods=20).rank(pct=True) * 2.0 - 1.0
    return daily_df


def normalize_factor_spec(formula: str | None = None, params: list[int] | tuple[int, ...] | None = None) -> dict:
    spec = decode_params(params) if params is not None else parse_formula(formula)
    if spec["a_name"] not in _INDICATOR_SET:
        raise ValueError(f"unknown indicator: {spec['a_name']}")
    if spec["b_name"] and spec["b_name"] not in _INDICATOR_SET:
        raise ValueError(f"unknown indicator: {spec['b_name']}")
    if spec["mask_field"] not in _INDICATOR_SET:
        raise ValueError(f"unknown mask indicator: {spec['mask_field']}")
    if spec["mask_rule"] not in _MASK_SET:
        raise ValueError(f"unsupported mask rule: {spec['mask_rule']}")
    if spec["mode"] == 1 and spec["op_name"] not in _MODE1_SET:
        raise ValueError(f"unsupported mode1 operator: {spec['op_name']}")
    if spec["mode"] == 2 and spec["op_name"] not in _MODE2_SET:
        raise ValueError(f"unsupported mode2 operator: {spec['op_name']}")
    spec = dict(spec)
    if spec.get("outer_op") not in OUTER_OPS:
        spec["outer_op"] = "identity"
    spec["impl_version"] = IMPL_VERSION
    spec["input_names"] = tuple(required_inputs_for_spec(spec))
    return spec


def _compute_slice_bounds(window: int, slice_value: float | None, n_minutes: int) -> tuple[int, int]:
    if window >= n_minutes:
        return 0, n_minutes
    if slice_value is None:
        return n_minutes - window, n_minutes
    center = int(slice_value * (n_minutes - 1))
    half = window // 2
    start = max(0, center - half)
    end = min(n_minutes, start + window)
    if end == n_minutes:
        start = max(0, n_minutes - window)
    return start, end


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(b) > EPS, a / b, 0.0)
    return out.astype(np.float32, copy=False)


def _shift_minute_window(x: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return x.astype(np.float32, copy=False)
    out = np.roll(x, shift=shift, axis=0).astype(np.float32, copy=False)
    if shift > 0:
        out[:shift, :] = np.nan
    else:
        out[shift:, :] = np.nan
    return out


def _zscore_window(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return ((x - mu) / std).astype(np.float32, copy=False)


def _build_cross_sectional_state_mask(field_data: np.ndarray, mask_rule: str) -> np.ndarray:
    direction, quantile_text = mask_rule.split("_", 1)
    quantile = float(quantile_text)
    values = field_data[0, :]
    valid = np.isfinite(values)
    fill_value = -1e18 if direction == "high" else 1e18
    filled = np.where(valid, values, fill_value)
    rank = np.argsort(np.argsort(filled)).astype(np.float32)
    n_valid = np.float32(valid.sum())
    rank_norm = rank / (n_valid - 1.0 + EPS)
    if direction == "high":
        stock_mask = rank_norm >= (1.0 - quantile)
    else:
        stock_mask = rank_norm <= quantile
    stock_mask = stock_mask & valid
    return np.broadcast_to(stock_mask.reshape(1, -1), field_data.shape)


def _build_mask(field_data: np.ndarray, mask_rule: str, mask_field_name: str | None = None) -> np.ndarray:
    if mask_rule == "none":
        return np.ones(field_data.shape, dtype=bool)
    if mask_field_name in DAILY_STATE_INDICATORS:
        return _build_cross_sectional_state_mask(field_data, mask_rule)
    direction, quantile_text = mask_rule.split("_", 1)
    quantile = float(quantile_text)
    valid = np.isfinite(field_data)
    fill_value = -1e18 if direction == "high" else 1e18
    filled = np.where(valid, field_data, fill_value)
    rank = np.argsort(np.argsort(filled, axis=0), axis=0).astype(np.float32)
    n_valid = np.sum(valid, axis=0, keepdims=True).astype(np.float32)
    rank_norm = rank / (n_valid - 1.0 + EPS)
    if direction == "high":
        mask = rank_norm >= (1.0 - quantile)
    else:
        mask = rank_norm <= quantile
    return mask & valid


def _derive_indicator(raw: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name in raw:
        out = np.asarray(raw[name])
        if not out.flags.writeable:
            out = out.copy()
        out = np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype(np.float32, copy=False)

    def need(field: str) -> np.ndarray:
        if field not in raw:
            raise KeyError(f"missing raw field: {field}")
        return raw[field]

    opens = need("opens") if name in {
        "price_range_pct", "upper_shadow", "lower_shadow", "body", "kyle_lambda",
        "signed_volume", "path_high_to_close", "path_low_rebound",
        "path_efficiency", "gate_rev_high_rv", "gate_rev_pm_disagree",
        "gate_rev_high_oimb",
    } else None
    highs = need("highs") if name in {
        "spread", "mid_price", "typical_price", "money_flow", "price_range_pct",
        "upper_shadow", "path_high_time", "path_high_to_close",
    } else None
    lows = need("lows") if name in {
        "spread", "mid_price", "typical_price", "money_flow", "price_range_pct",
        "lower_shadow", "path_low_time", "path_low_rebound",
    } else None
    closes = need("closes") if name in {
        "typical_price", "money_flow", "upper_shadow", "lower_shadow", "body",
        "vwap_deviation", "amihud", "kyle_lambda", "signed_volume",
        "path_high_to_close", "path_low_rebound", "path_efficiency",
        "gate_rev_high_rv", "gate_rev_pm_disagree", "gate_rev_high_oimb",
    } else None
    volumes = need("volumes") if name in {
        "money_flow", "log_volume", "volume_intensity", "amt_per_trade_proxy",
        "amihud", "kyle_lambda", "signed_volume", "vol_accel",
        "order_imbalance", "gate_rev_high_oimb",
    } else None
    amounts = need("amounts") if name in {"log_amount", "amt_per_trade_proxy", "gate_liq_high_amount"} else None
    returns = need("returns") if name in {
        "return_abs", "cum_return", "amihud", "rv_ratio", "order_imbalance",
        "gate_liq_high_amount", "gate_rev_high_rv", "gate_rev_high_oimb",
    } else None
    vwaps = need("vwaps") if name in {"vwap_deviation"} else None

    if name == "spread":
        out = highs - lows
    elif name == "mid_price":
        out = (highs + lows) * 0.5
    elif name == "typical_price":
        out = (highs + lows + closes) / 3.0
    elif name == "money_flow":
        out = ((highs + lows + closes) / 3.0) * volumes
    elif name == "return_abs":
        out = np.abs(returns)
    elif name == "log_volume":
        out = np.log(volumes + 1.0)
    elif name == "log_amount":
        out = np.log(amounts + 1.0)
    elif name == "price_range_pct":
        out = (highs - lows) / (opens + EPS)
    elif name == "upper_shadow":
        out = highs - np.maximum(opens, closes)
    elif name == "lower_shadow":
        out = np.minimum(opens, closes) - lows
    elif name == "body":
        out = np.abs(closes - opens)
    elif name == "vwap_deviation":
        out = closes - vwaps
    elif name == "volume_intensity":
        out = volumes / (np.nanmean(volumes, axis=0, keepdims=True) + EPS)
    elif name == "cum_return":
        out = np.cumprod(1.0 + np.nan_to_num(returns, nan=0.0), axis=0)
    elif name == "amt_per_trade_proxy":
        out = amounts / (volumes + EPS)
    elif name == "amihud":
        out = np.abs(returns) / (volumes * closes + EPS)
    elif name == "kyle_lambda":
        out = np.abs(closes - opens) / (np.sqrt(volumes * closes + EPS) + EPS)
    elif name == "signed_volume":
        out = np.sign(closes - opens) * volumes
    elif name == "vol_accel":
        vol_lag = np.roll(volumes, 1, axis=0)
        vol_lag[0, :] = volumes[0, :]
        out = np.sign((volumes - vol_lag) / (vol_lag + EPS)) * np.log1p(np.abs((volumes - vol_lag) / (vol_lag + EPS)))
    elif name == "rv_ratio":
        rv1 = np.nanvar(returns, axis=0, keepdims=True)
        rv5 = np.nanvar(returns[::5, :], axis=0, keepdims=True)
        out = np.broadcast_to(rv1 / (rv5 + EPS), returns.shape)
    elif name == "order_imbalance":
        sign_vol = np.sign(np.nan_to_num(returns, nan=0.0)) * np.nan_to_num(volumes, nan=0.0)
        out = np.cumsum(sign_vol, axis=0) / (np.cumsum(np.nan_to_num(volumes, nan=0.0), axis=0) + EPS)
    elif name in {
        "path_high_time", "path_low_time", "path_high_to_close",
        "path_low_rebound", "path_efficiency", "gate_liq_high_amount",
        "gate_rev_high_rv", "gate_rev_pm_disagree", "gate_rev_high_oimb",
    }:
        base = highs if highs is not None else closes if closes is not None else returns
        n_minutes, _ = base.shape

        def broadcast(values: np.ndarray) -> np.ndarray:
            return np.broadcast_to(values.reshape(1, -1), base.shape)

        if name == "path_high_time":
            fill = np.where(np.isfinite(highs), highs, -np.inf)
            values = np.argmax(fill, axis=0).astype(np.float32) / np.float32(max(n_minutes - 1, 1))
            values[~np.any(np.isfinite(highs), axis=0)] = np.nan
            out = broadcast(values)
        elif name == "path_low_time":
            fill = np.where(np.isfinite(lows), lows, np.inf)
            values = np.argmin(fill, axis=0).astype(np.float32) / np.float32(max(n_minutes - 1, 1))
            values[~np.any(np.isfinite(lows), axis=0)] = np.nan
            out = broadcast(values)
        elif name == "path_high_to_close":
            values = (np.nanmax(highs, axis=0) - closes[-1]) / (np.abs(np.nanmax(highs, axis=0) - opens[0]) + EPS)
            out = broadcast(values)
        elif name == "path_low_rebound":
            values = (closes[-1] - np.nanmin(lows, axis=0)) / (np.abs(opens[0] - np.nanmin(lows, axis=0)) + EPS)
            out = broadcast(values)
        elif name == "path_efficiency":
            path_len = np.nansum(np.abs(np.diff(closes, axis=0)), axis=0)
            out = broadcast(np.abs(closes[-1] - opens[0]) / (path_len + EPS))
        elif name == "gate_liq_high_amount":
            amount_sum = np.nansum(amounts, axis=0)
            daily_amihud = np.nanmean(np.abs(returns) / (amounts + EPS), axis=0)
            gate = amount_sum >= np.nanpercentile(amount_sum, 70)
            out = broadcast(np.where(gate, -daily_amihud, 0.0))
        elif name == "gate_rev_high_rv":
            intraday_ret = closes[-1] / (opens[0] + EPS) - 1.0
            realized_vol = np.sqrt(np.nansum(returns * returns, axis=0))
            gate = realized_vol >= np.nanpercentile(realized_vol, 70)
            out = broadcast(np.where(gate, -intraday_ret, 0.0))
        elif name == "gate_rev_pm_disagree":
            intraday_ret = closes[-1] / (opens[0] + EPS) - 1.0
            mid = closes[n_minutes // 2]
            am_pm_ret_diff = mid / (opens[0] + EPS) - 1.0 - (closes[-1] / (mid + EPS) - 1.0)
            out = broadcast(np.where((am_pm_ret_diff * intraday_ret) < 0.0, -intraday_ret, 0.0))
        else:
            intraday_ret = closes[-1] / (opens[0] + EPS) - 1.0
            volume_sum = np.nansum(volumes, axis=0)
            daily_oimb = np.nansum(np.sign(returns) * volumes, axis=0) / (volume_sum + EPS)
            gate = np.abs(daily_oimb) >= np.nanpercentile(np.abs(daily_oimb), 70)
            out = broadcast(np.where(gate, -intraday_ret, 0.0))
    else:
        raise ValueError(f"unsupported indicator: {name}")
    out = np.asarray(out)
    if not out.flags.writeable:
        out = out.copy()
    out = np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32, copy=False)


@njit(cache=True, parallel=True)
def _masked_first_last_count(arr, mask):
    nd, ns = arr.shape
    first = np.full(ns, np.nan, dtype=np.float32)
    last = np.full(ns, np.nan, dtype=np.float32)
    count = np.zeros(ns, dtype=np.int32)
    for c in prange(ns):
        seen = False
        f = 0.0
        l = 0.0
        cnt = 0
        for r in range(nd):
            v = arr[r, c]
            if mask[r, c] and np.isfinite(v):
                if not seen:
                    f = v
                    seen = True
                l = v
                cnt += 1
        if seen:
            first[c] = f
            last[c] = l
            count[c] = cnt
    return first, last, count


@njit(cache=True, parallel=True)
def _safe_div_1d(a, b):
    n = len(a)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in prange(n):
        if np.isfinite(a[i]) and np.isfinite(b[i]) and np.abs(b[i]) > 1e-12:
            out[i] = a[i] / b[i]
        else:
            out[i] = 0.0
    return out


@njit(cache=True, parallel=True)
def _autocorr_masked(arr, mask):
    nd, ns = arr.shape
    out = np.full(ns, np.nan, dtype=np.float32)
    for c in prange(ns):
        s0 = 0.0
        s1 = 0.0
        cnt = 0
        for r in range(1, nd):
            v0 = arr[r - 1, c]
            v1 = arr[r, c]
            if mask[r - 1, c] and mask[r, c] and np.isfinite(v0) and np.isfinite(v1):
                s0 += v0
                s1 += v1
                cnt += 1
        if cnt < 5:
            out[c] = 0.0
            continue
        m0 = s0 / cnt
        m1 = s1 / cnt
        cov = 0.0
        ss0 = 0.0
        ss1 = 0.0
        for r in range(1, nd):
            v0 = arr[r - 1, c]
            v1 = arr[r, c]
            if mask[r - 1, c] and mask[r, c] and np.isfinite(v0) and np.isfinite(v1):
                d0 = v0 - m0
                d1 = v1 - m1
                cov += d0 * d1
                ss0 += d0 * d0
                ss1 += d1 * d1
        if ss0 > 1e-12 and ss1 > 1e-12:
            out[c] = cov / np.sqrt(ss0 * ss1)
        else:
            out[c] = 0.0
    return out


@njit(cache=True, parallel=True)
def _entropy_masked(arr, mask):
    nd, ns = arr.shape
    out = np.zeros(ns, dtype=np.float32)
    for c in prange(ns):
        lo = np.inf
        hi = -np.inf
        cnt = 0
        for r in range(nd):
            v = arr[r, c]
            if mask[r, c] and np.isfinite(v):
                if v < lo:
                    lo = v
                if v > hi:
                    hi = v
                cnt += 1
        if cnt == 0:
            continue
        rng = hi - lo
        bins = np.zeros(10, dtype=np.int32)
        if rng < 1e-12:
            bins[0] = cnt
        else:
            for r in range(nd):
                v = arr[r, c]
                if mask[r, c] and np.isfinite(v):
                    idx = int(((v - lo) / (rng + 1e-12)) * 9.99)
                    if idx < 0:
                        idx = 0
                    elif idx > 9:
                        idx = 9
                    bins[idx] += 1
        ent = 0.0
        for k in range(10):
            if bins[k] > 0:
                p = bins[k] / cnt
                ent -= p * np.log(p + 1e-12)
        out[c] = ent
    return out


@njit(cache=True, parallel=True)
def _zero_cross_masked(arr, mask):
    nd, ns = arr.shape
    out = np.zeros(ns, dtype=np.float32)
    for c in prange(ns):
        cross = 0
        valid_pairs = 0
        for r in range(1, nd):
            v0 = arr[r - 1, c]
            v1 = arr[r, c]
            if mask[r - 1, c] and mask[r, c] and np.isfinite(v0) and np.isfinite(v1):
                valid_pairs += 1
                if np.abs(np.sign(v1) - np.sign(v0)) > 0.5:
                    cross += 1
        if valid_pairs > 0:
            out[c] = cross / valid_pairs
    return out


@njit(cache=True, parallel=True)
def _linear_stats(arr1, arr2, mask):
    nd, ns = arr1.shape
    n = np.zeros(ns, dtype=np.int32)
    ma = np.zeros(ns, dtype=np.float32)
    mb = np.zeros(ns, dtype=np.float32)
    va = np.zeros(ns, dtype=np.float32)
    vb = np.zeros(ns, dtype=np.float32)
    cov = np.zeros(ns, dtype=np.float32)
    for c in prange(ns):
        s1 = 0.0
        s2 = 0.0
        cnt = 0
        for r in range(nd):
            v1 = arr1[r, c]
            v2 = arr2[r, c]
            if mask[r, c] and np.isfinite(v1) and np.isfinite(v2):
                s1 += v1
                s2 += v2
                cnt += 1
        n[c] = cnt
        if cnt < 1:
            continue
        m1 = s1 / cnt
        m2 = s2 / cnt
        ma[c] = m1
        mb[c] = m2
        ss1 = 0.0
        ss2 = 0.0
        cc = 0.0
        for r in range(nd):
            v1 = arr1[r, c]
            v2 = arr2[r, c]
            if mask[r, c] and np.isfinite(v1) and np.isfinite(v2):
                d1 = v1 - m1
                d2 = v2 - m2
                ss1 += d1 * d1
                ss2 += d2 * d2
                cc += d1 * d2
        va[c] = ss1 / cnt
        vb[c] = ss2 / cnt
        cov[c] = cc / cnt
    return n, ma, mb, va, vb, cov


@njit(cache=True, parallel=True)
def _cos_sim_masked(arr1, arr2, mask):
    nd, ns = arr1.shape
    out = np.zeros(ns, dtype=np.float32)
    for c in prange(ns):
        dot = 0.0
        na = 0.0
        nb = 0.0
        for r in range(nd):
            v1 = arr1[r, c]
            v2 = arr2[r, c]
            if mask[r, c] and np.isfinite(v1) and np.isfinite(v2):
                dot += v1 * v2
                na += v1 * v1
                nb += v2 * v2
        if na > 1e-12 and nb > 1e-12:
            out[c] = dot / np.sqrt(na * nb)
    return out


@njit(cache=True, parallel=True)
def _euc_dist_masked(arr1, arr2, mask):
    nd, ns = arr1.shape
    out = np.zeros(ns, dtype=np.float32)
    for c in prange(ns):
        s1 = 0.0
        s2 = 0.0
        cnt = 0
        for r in range(nd):
            v1 = arr1[r, c]
            v2 = arr2[r, c]
            if mask[r, c] and np.isfinite(v1) and np.isfinite(v2):
                s1 += v1
                s2 += v2
                cnt += 1
        if cnt < 5:
            continue
        m1 = s1 / cnt
        m2 = s2 / cnt
        ss1 = 0.0
        ss2 = 0.0
        for r in range(nd):
            v1 = arr1[r, c]
            v2 = arr2[r, c]
            if mask[r, c] and np.isfinite(v1) and np.isfinite(v2):
                d1 = v1 - m1
                d2 = v2 - m2
                ss1 += d1 * d1
                ss2 += d2 * d2
        std1 = np.sqrt(ss1 / cnt + 1e-12)
        std2 = np.sqrt(ss2 / cnt + 1e-12)
        acc = 0.0
        for r in range(nd):
            v1 = arr1[r, c]
            v2 = arr2[r, c]
            if mask[r, c] and np.isfinite(v1) and np.isfinite(v2):
                z1 = (v1 - m1) / std1
                z2 = (v2 - m2) / std2
                d = z1 - z2
                acc += d * d
        out[c] = np.sqrt(acc / cnt)
    return out


@njit(cache=True, parallel=True)
def _residual_std_masked(arr1, arr2, mask):
    nd, ns = arr1.shape
    out = np.zeros(ns, dtype=np.float32)
    for c in prange(ns):
        s1 = 0.0
        s2 = 0.0
        cnt = 0
        for r in range(nd):
            v1 = arr1[r, c]
            v2 = arr2[r, c]
            if mask[r, c] and np.isfinite(v1) and np.isfinite(v2):
                s1 += v1
                s2 += v2
                cnt += 1
        if cnt < 5:
            continue
        m1 = s1 / cnt
        m2 = s2 / cnt
        ss2 = 0.0
        cov = 0.0
        for r in range(nd):
            v1 = arr1[r, c]
            v2 = arr2[r, c]
            if mask[r, c] and np.isfinite(v1) and np.isfinite(v2):
                d1 = v1 - m1
                d2 = v2 - m2
                ss2 += d2 * d2
                cov += d1 * d2
        beta = cov / (ss2 + 1e-12)
        alpha = m1 - beta * m2
        rss = 0.0
        rcnt = 0
        for r in range(nd):
            v1 = arr1[r, c]
            v2 = arr2[r, c]
            if mask[r, c] and np.isfinite(v1) and np.isfinite(v2):
                res = v1 - (alpha + beta * v2)
                rss += res * res
                rcnt += 1
        if rcnt > 0:
            out[c] = np.sqrt(rss / rcnt)
    return out


@njit(cache=True, parallel=True)
def _rank_corr_masked(arr1, arr2, mask):
    nd, ns = arr1.shape
    out = np.zeros(ns, dtype=np.float32)
    for c in prange(ns):
        cnt = 0
        for r in range(nd):
            v1 = arr1[r, c]
            v2 = arr2[r, c]
            if mask[r, c] and np.isfinite(v1) and np.isfinite(v2):
                cnt += 1
        if cnt < 5:
            continue
        vals1 = np.empty(cnt, dtype=np.float32)
        vals2 = np.empty(cnt, dtype=np.float32)
        k = 0
        for r in range(nd):
            v1 = arr1[r, c]
            v2 = arr2[r, c]
            if mask[r, c] and np.isfinite(v1) and np.isfinite(v2):
                vals1[k] = v1
                vals2[k] = v2
                k += 1
        order1 = np.argsort(vals1)
        order2 = np.argsort(vals2)
        rank1 = np.empty(cnt, dtype=np.float32)
        rank2 = np.empty(cnt, dtype=np.float32)
        for k in range(cnt):
            rank1[order1[k]] = k
            rank2[order2[k]] = k
        m1 = (cnt - 1) * 0.5
        m2 = (cnt - 1) * 0.5
        ss1 = 0.0
        ss2 = 0.0
        cov = 0.0
        for k in range(cnt):
            d1 = rank1[k] - m1
            d2 = rank2[k] - m2
            ss1 += d1 * d1
            ss2 += d2 * d2
            cov += d1 * d2
        if ss1 > 1e-12 and ss2 > 1e-12:
            out[c] = cov / np.sqrt(ss1 * ss2)
    return out


@njit(cache=True, parallel=True)
def _mutual_info_masked(arr1, arr2, mask):
    nd, ns = arr1.shape
    out = np.zeros(ns, dtype=np.float32)
    for c in prange(ns):
        cnt = 0
        lo1 = np.inf
        hi1 = -np.inf
        lo2 = np.inf
        hi2 = -np.inf
        for r in range(nd):
            v1 = arr1[r, c]
            v2 = arr2[r, c]
            if mask[r, c] and np.isfinite(v1) and np.isfinite(v2):
                cnt += 1
                if v1 < lo1:
                    lo1 = v1
                if v1 > hi1:
                    hi1 = v1
                if v2 < lo2:
                    lo2 = v2
                if v2 > hi2:
                    hi2 = v2
        if cnt < 10:
            continue
        joint = np.zeros((5, 5), dtype=np.int32)
        cnt1 = np.zeros(5, dtype=np.int32)
        cnt2 = np.zeros(5, dtype=np.int32)
        rng1 = hi1 - lo1
        rng2 = hi2 - lo2
        for r in range(nd):
            v1 = arr1[r, c]
            v2 = arr2[r, c]
            if mask[r, c] and np.isfinite(v1) and np.isfinite(v2):
                i = 0 if rng1 < 1e-12 else int(((v1 - lo1) / (rng1 + 1e-12)) * 4.99)
                j = 0 if rng2 < 1e-12 else int(((v2 - lo2) / (rng2 + 1e-12)) * 4.99)
                if i < 0:
                    i = 0
                elif i > 4:
                    i = 4
                if j < 0:
                    j = 0
                elif j > 4:
                    j = 4
                joint[i, j] += 1
                cnt1[i] += 1
                cnt2[j] += 1
        mi = 0.0
        for i in range(5):
            for j in range(5):
                if joint[i, j] > 0:
                    pxy = joint[i, j] / cnt
                    px = cnt1[i] / cnt
                    py = cnt2[j] / cnt
                    mi += pxy * np.log(pxy / (px * py + 1e-12) + 1e-12)
        out[c] = mi
    return out


def _masked_array(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.where(mask & np.isfinite(arr), arr, np.nan)
    return out.astype(np.float32, copy=False)


def _paired_stats_np(a: np.ndarray, b: np.ndarray, mask: np.ndarray):
    valid = mask & np.isfinite(a) & np.isfinite(b)
    af = np.where(valid, a, 0.0)
    bf = np.where(valid, b, 0.0)
    n = np.sum(valid, axis=0).astype(np.float32)
    sa = np.sum(af, axis=0)
    sb = np.sum(bf, axis=0)
    sa2 = np.sum(af * af, axis=0)
    sb2 = np.sum(bf * bf, axis=0)
    sab = np.sum(af * bf, axis=0)
    ma = sa / (n + EPS)
    mb = sb / (n + EPS)
    va = np.maximum(sa2 / (n + EPS) - ma * ma, 0.0)
    vb = np.maximum(sb2 / (n + EPS) - mb * mb, 0.0)
    cov = sab / (n + EPS) - ma * mb
    return n, ma, mb, va, vb, cov, valid


def _rank_by_column_np(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    filled = np.where(valid, x, 1e18)
    rank = np.argsort(np.argsort(filled, axis=0), axis=0).astype(np.float32)
    return np.where(valid, rank, np.nan)


def _apply_mode1(a: np.ndarray, mask: np.ndarray, op_name: str) -> np.ndarray:
    x = _masked_array(a, mask)
    if op_name == "mean":
        return ops_mean(x)
    if op_name == "std":
        return ops_std(x)
    if op_name == "skew":
        return ops_skew(x)
    if op_name == "kurt":
        return ops_kurt(x)
    if op_name == "median":
        return np.nanmedian(x, axis=0).astype(np.float32)
    if op_name == "sum":
        return ops_sum(x)
    if op_name == "max":
        return ops_max(x)
    if op_name == "min":
        return ops_min(x)
    if op_name == "range":
        return (ops_max(x) - ops_min(x)).astype(np.float32)
    if op_name == "first":
        first, _, _ = _masked_first_last_count(a, mask)
        return first
    if op_name == "last":
        _, last, _ = _masked_first_last_count(a, mask)
        return last
    if op_name == "ret":
        first, last, count = _masked_first_last_count(a, mask)
        out = _safe_div_1d(last - first, np.abs(first) + EPS)
        out[count < 2] = np.nan
        return out.astype(np.float32, copy=False)
    if op_name == "autocorr":
        return _autocorr_masked(a, mask)
    if op_name == "entropy":
        return _entropy_masked(a, mask)
    if op_name == "zero_cross":
        return _zero_cross_masked(a, mask)
    raise ValueError(f"unsupported mode1 operator: {op_name}")


def _apply_mode2(a: np.ndarray, b: np.ndarray, mask: np.ndarray, op_name: str) -> np.ndarray:
    pair_mask = mask & np.isfinite(a) & np.isfinite(b)
    xa = np.where(pair_mask, a, np.nan).astype(np.float32, copy=False)
    xb = np.where(pair_mask, b, np.nan).astype(np.float32, copy=False)
    if op_name == "corr":
        return ops_corr(xa, xb, 0)
    if op_name == "wmean":
        weights = np.where(np.isfinite(xb) & (xb > 0), xb, np.nan).astype(np.float32, copy=False)
        return ops_wmean(xa, weights)
    if op_name == "euc_dist":
        valid = mask & np.isfinite(a) & np.isfinite(b)
        n = np.sum(valid, axis=0).astype(np.float32)
        xa_full = np.where(valid, a, np.nan)
        xb_full = np.where(valid, b, np.nan)
        ma = np.nanmean(xa_full, axis=0, keepdims=True)
        mb = np.nanmean(xb_full, axis=0, keepdims=True)
        sa = np.nanstd(xa_full, axis=0, keepdims=True) + EPS
        sb = np.nanstd(xb_full, axis=0, keepdims=True) + EPS
        za = (a - ma) / sa
        zb = (b - mb) / sb
        diff_sq = np.where(valid, (za - zb) ** 2, 0.0)
        dist = np.sqrt(np.sum(diff_sq, axis=0) / (n + EPS))
        return np.where(n >= 5, dist, 0.0).astype(np.float32)
    if op_name == "cos_sim":
        return _cos_sim_masked(a, b, pair_mask)
    if op_name == "rank_corr":
        valid = mask & np.isfinite(a) & np.isfinite(b)
        a_rank = _rank_by_column_np(a, valid)
        b_rank = _rank_by_column_np(b, valid)
        n, _, _, va, vb, cov, _ = _paired_stats_np(a_rank, b_rank, valid)
        corr = cov / (np.sqrt(va * vb) + EPS)
        return np.where(n >= 5, np.clip(corr, -1.0, 1.0), 0.0).astype(np.float32)
    if op_name == "mutual_info":
        valid = mask & np.isfinite(a) & np.isfinite(b)
        n = np.sum(valid, axis=0).astype(np.float32)
        xa_full = np.where(valid, a, np.nan)
        xb_full = np.where(valid, b, np.nan)
        a_lo = np.nanmin(xa_full, axis=0, keepdims=True)
        a_hi = np.nanmax(xa_full, axis=0, keepdims=True)
        b_lo = np.nanmin(xb_full, axis=0, keepdims=True)
        b_hi = np.nanmax(xb_full, axis=0, keepdims=True)
        a_bin = np.clip(((a - a_lo) / (a_hi - a_lo + EPS) * 4.99).astype(np.int32), 0, 4)
        b_bin = np.clip(((b - b_lo) / (b_hi - b_lo + EPS) * 4.99).astype(np.int32), 0, 4)
        mi = np.zeros(a.shape[1], dtype=np.float32)
        for bucket_a in range(5):
            p_a = np.sum((a_bin == bucket_a) & valid, axis=0).astype(np.float32) / (n + EPS)
            for bucket_b in range(5):
                joint = ((a_bin == bucket_a) & (b_bin == bucket_b) & valid)
                p_joint = np.sum(joint, axis=0).astype(np.float32) / (n + EPS)
                p_b = np.sum((b_bin == bucket_b) & valid, axis=0).astype(np.float32) / (n + EPS)
                ratio = p_joint / (p_a * p_b + EPS)
                mi += np.where(p_joint > EPS, p_joint * np.log(ratio + EPS), 0.0)
        return np.where(n >= 10, mi, 0.0).astype(np.float32)
    if op_name == "max_corr":
        window_len = a.shape[0]
        part = max(window_len // 3, 5)
        best = np.zeros(a.shape[1], dtype=np.float32)
        for start in range(0, window_len - part + 1, part):
            end = min(start + part, window_len)
            sub = _apply_mode2(a[start:end, :], b[start:end, :], mask[start:end, :], "corr")
            best = np.maximum(best, np.abs(sub))
        return best
    if op_name == "tail_corr":
        lo = np.nanpercentile(xa, 30, axis=0, keepdims=True)
        hi = np.nanpercentile(xa, 70, axis=0, keepdims=True)
        tail_mask = mask & ((a <= lo) | (a >= hi))
        return _apply_mode2(a, b, tail_mask, "corr")

    n, ma, mb, va, vb, cov = _linear_stats(a, b, mask)
    valid = n >= 5
    if op_name == "slope":
        out = cov / (vb + EPS)
        return np.where(valid, out, 0.0).astype(np.float32)
    if op_name == "intercept":
        slope = cov / (vb + EPS)
        out = ma - slope * mb
        return np.where(valid, out, 0.0).astype(np.float32)
    if op_name == "r2":
        corr = cov / (np.sqrt(va * vb) + EPS)
        out = np.clip(corr ** 2, 0.0, 1.0)
        return np.where(valid, out, 0.0).astype(np.float32)
    if op_name == "cov":
        return np.where(valid, cov, 0.0).astype(np.float32)
    if op_name == "beta":
        out = cov / (vb + EPS)
        return np.where(valid, out, 0.0).astype(np.float32)
    if op_name == "residual_std":
        return _residual_std_masked(a, b, pair_mask)
    if op_name == "diff_mean":
        return (ops_mean(xa) - ops_mean(xb)).astype(np.float32)
    if op_name == "diff_std":
        return (ops_std(xa) - ops_std(xb)).astype(np.float32)
    if op_name == "ratio_mean":
        return _safe_div_1d(ops_mean(xa), ops_mean(xb))
    if op_name == "ratio_std":
        return _safe_div_1d(ops_std(xa), ops_std(xb))
    raise ValueError(f"unsupported mode2 operator: {op_name}")


def compute_parametric_day(
    raw: dict[str, np.ndarray],
    a_name: str,
    b_name: str,
    window: int,
    slice_value: float | None,
    mask_field: str,
    mask_rule: str,
    mode: int,
    op_name: str,
    lag: int,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        a_full = _derive_indicator(raw, a_name)
        start, end = _compute_slice_bounds(int(window), slice_value, a_full.shape[0])
        a = a_full[start:end, :]
        if mask_rule == "none":
            mask = np.ones(a.shape, dtype=bool)
        else:
            mask_data = _derive_indicator(raw, mask_field)[start:end, :]
            mask = _build_mask(mask_data, mask_rule, mask_field)
        if int(mode) == 1:
            return _apply_mode1(a, mask, op_name)
        b_full = _derive_indicator(raw, b_name)[start:end, :]
        b = _shift_minute_window(b_full, int(lag))
        if op_name in SCALE_SENSITIVE_OPS:
            if DIM_GROUPS.get(a_name, "OTHER") != DIM_GROUPS.get(b_name, "OTHER"):
                a = _zscore_window(a)
                b = _zscore_window(b)
        return _apply_mode2(a, b, mask, op_name)


@MinuteFactorEngine.register(OPERATOR_NAME, param_order=OPERATOR_PARAM_ORDER)
def gp_parametric_formula(*args):
    n_meta = len(OPERATOR_PARAM_ORDER)
    if len(args) < n_meta:
        raise ValueError("invalid operator args")
    input_arrays = args[:-n_meta]
    (
        _impl_version,
        input_names,
        a_name,
        b_name,
        window,
        slice_value,
        mask_field,
        mask_rule,
        mode,
        op_name,
        lag,
    ) = args[-n_meta:]
    if len(input_arrays) != len(input_names):
        raise ValueError("input_names and arrays length mismatch")
    raw = {name: arr for name, arr in zip(input_names, input_arrays)}
    return compute_parametric_day(
        raw=raw,
        a_name=a_name,
        b_name=b_name,
        window=int(window),
        slice_value=slice_value,
        mask_field=mask_field,
        mask_rule=mask_rule,
        mode=int(mode),
        op_name=op_name,
        lag=int(lag),
    )
