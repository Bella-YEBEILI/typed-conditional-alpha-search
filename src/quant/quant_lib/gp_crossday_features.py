import h5py
import numpy as np


EPS = np.float32(1e-8)
CROSSDAY_ANOMALY_NAMES = (
    "anom_amihud_z20",
    "anom_tail_vol_share_z20",
    "anom_high_to_close_z20",
    "anom_order_imbalance_z20",
    "anom_path_efficiency_z20",
)


def rolling_zscore_past(x: np.ndarray, window: int = 20, min_periods: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    valid = np.isfinite(x)
    x0 = np.where(valid, x, 0.0).astype(np.float32, copy=False)
    x2 = (x0 * x0).astype(np.float32, copy=False)
    zero_f = np.zeros((1, x.shape[1]), dtype=np.float32)
    zero_i = np.zeros((1, x.shape[1]), dtype=np.int32)
    cs = np.vstack((zero_f, np.cumsum(x0, axis=0, dtype=np.float32)))
    cs2 = np.vstack((zero_f, np.cumsum(x2, axis=0, dtype=np.float32)))
    cc = np.vstack((zero_i, np.cumsum(valid.astype(np.int32), axis=0, dtype=np.int32)))
    end = np.arange(x.shape[0], dtype=np.int32)
    start = np.maximum(end - int(window), 0)
    cnt = cc[end] - cc[start]
    s1 = cs[end] - cs[start]
    s2 = cs2[end] - cs2[start]
    mean = np.divide(s1, cnt, out=np.zeros_like(s1), where=cnt > 0)
    var = np.divide(s2, cnt, out=np.zeros_like(s2), where=cnt > 0) - mean * mean
    std = np.sqrt(np.maximum(var, EPS))
    z = (x - mean) / std
    z[(cnt < int(min_periods)) | ~valid] = 0.0
    return np.nan_to_num(z, copy=False, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _read_h5_field(ds, day_indices: np.ndarray, stock_idx):
    if day_indices.size == 0:
        return np.empty((0, ds.shape[1], len(stock_idx) if stock_idx is not None else ds.shape[2]), dtype=np.float32)
    contiguous = day_indices.size > 1 and day_indices[-1] - day_indices[0] + 1 == day_indices.size
    if contiguous:
        if stock_idx is None:
            return ds[day_indices[0]:day_indices[-1] + 1].astype(np.float32)
        try:
            return ds[day_indices[0]:day_indices[-1] + 1, :, stock_idx].astype(np.float32)
        except TypeError:
            return ds[day_indices[0]:day_indices[-1] + 1][:, :, stock_idx].astype(np.float32)
    n_stocks = len(stock_idx) if stock_idx is not None else ds.shape[2]
    out = np.empty((day_indices.size, ds.shape[1], n_stocks), dtype=np.float32)
    for i, di in enumerate(day_indices):
        row = ds[int(di)]
        out[i] = row if stock_idx is None else row[:, stock_idx]
    return out


def _read_h5_minute(ds, day_indices: np.ndarray, stock_idx, minute_idx: int):
    if day_indices.size == 0:
        return np.empty((0, len(stock_idx) if stock_idx is not None else ds.shape[2]), dtype=np.float32)
    contiguous = day_indices.size > 1 and day_indices[-1] - day_indices[0] + 1 == day_indices.size
    if contiguous:
        if stock_idx is None:
            return ds[day_indices[0]:day_indices[-1] + 1, minute_idx, :].astype(np.float32)
        try:
            return ds[day_indices[0]:day_indices[-1] + 1, minute_idx, stock_idx].astype(np.float32)
        except TypeError:
            return ds[day_indices[0]:day_indices[-1] + 1, minute_idx, :][:, stock_idx].astype(np.float32)
    n_stocks = len(stock_idx) if stock_idx is not None else ds.shape[2]
    out = np.empty((day_indices.size, n_stocks), dtype=np.float32)
    for i, di in enumerate(day_indices):
        row = ds[int(di), minute_idx, :]
        out[i] = row if stock_idx is None else row[stock_idx]
    return out


def build_crossday_anomaly_panels_from_h5(
    h5_path: str,
    day_indices: np.ndarray,
    stock_idx=None,
    *,
    chunk_days: int = 96,
    window: int = 20,
) -> dict[str, np.ndarray]:
    day_indices = np.asarray(day_indices, dtype=np.int64)
    if stock_idx is not None:
        stock_idx = np.asarray(stock_idx, dtype=np.int64)
    with h5py.File(h5_path, "r") as f:
        n_stocks = len(stock_idx) if stock_idx is not None else f["data/opens"].shape[2]
        n_days = day_indices.size
        daily = {
            "amihud": np.empty((n_days, n_stocks), dtype=np.float32),
            "tail_vol_share": np.empty((n_days, n_stocks), dtype=np.float32),
            "high_to_close": np.empty((n_days, n_stocks), dtype=np.float32),
            "order_imbalance": np.empty((n_days, n_stocks), dtype=np.float32),
            "path_efficiency": np.empty((n_days, n_stocks), dtype=np.float32),
        }
        for start in range(0, n_days, int(chunk_days)):
            end = min(start + int(chunk_days), n_days)
            di = day_indices[start:end]
            open0 = _read_h5_minute(f["data/opens"], di, stock_idx, 0)
            closes = _read_h5_field(f["data/closes"], di, stock_idx)
            close_last = closes[:, -1, :]
            path_len = np.nansum(np.abs(np.diff(closes, axis=1)), axis=1)
            del closes

            highs = _read_h5_field(f["data/highs"], di, stock_idx)
            high_max = np.nanmax(highs, axis=1)
            del highs

            returns = _read_h5_field(f["data/returns"], di, stock_idx)
            amounts = _read_h5_field(f["data/amounts"], di, stock_idx)
            daily["amihud"][start:end] = np.nanmean(np.abs(returns) / (amounts + EPS), axis=1)
            del amounts

            volumes = _read_h5_field(f["data/volumes"], di, stock_idx)
            vol_sum = np.nansum(volumes, axis=1)
            daily["tail_vol_share"][start:end] = np.nansum(volumes[:, -30:, :], axis=1) / (vol_sum + EPS)
            daily["high_to_close"][start:end] = (high_max - close_last) / (np.abs(high_max - open0) + EPS)
            daily["order_imbalance"][start:end] = np.nansum(np.sign(returns) * volumes, axis=1) / (vol_sum + EPS)
            daily["path_efficiency"][start:end] = np.abs(close_last - open0) / (path_len + EPS)

            del open0, close_last, high_max, path_len, returns, volumes, vol_sum

    panels = {
        "anom_amihud_z20": rolling_zscore_past(daily["amihud"], window),
        "anom_tail_vol_share_z20": rolling_zscore_past(daily["tail_vol_share"], window),
        "anom_high_to_close_z20": rolling_zscore_past(daily["high_to_close"], window),
        "anom_order_imbalance_z20": rolling_zscore_past(daily["order_imbalance"], window),
        "anom_path_efficiency_z20": rolling_zscore_past(daily["path_efficiency"], window),
    }
    return panels
