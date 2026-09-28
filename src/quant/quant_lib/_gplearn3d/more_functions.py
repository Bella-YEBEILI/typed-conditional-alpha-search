"""
Unified function library for 3D genetic programming
All operators for gplearn 3D version are defined here.

This module provides:
- Basic arithmetic operators (add, sub, mul, div, neg, abs, sqrt, log, exp)
- Cross-sectional operators (rank, cs_zscore, scale)
- Time series operators (delta, delay, rolling stats)
- Dynamic window operators (ts_mean, ts_std, ts_max, ts_min, ts_rank, ts_corr, ts_cov)
"""
import copy

import numpy as np
import pandas as pd

from .functions import _Function


def error_state_decorator(func):
    """Decorator to suppress numpy overflow/underflow warnings"""
    def wrapper(A, *args, **kwargs):
        with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
            return func(A, *args, **kwargs)
    return wrapper


# ============================================================================
# Basic arithmetic operators
# ============================================================================

@error_state_decorator
def abs_op(A):
    """Absolute value"""
    return np.abs(A)


@error_state_decorator
def sqrt_op(A):
    """Square root (negative values set to NaN)"""
    result = np.sqrt(np.where(A >= 0, A, np.nan))
    return np.nan_to_num(result, nan=0.0)


@error_state_decorator
def log_op(A):
    """Natural logarithm (non-positive values set to NaN)"""
    result = np.log(np.where(A > 0, A, np.nan))
    return np.nan_to_num(result, nan=0.0)


@error_state_decorator
def exp_op(A):
    """Exponential (clamped to prevent overflow)"""
    clamped = np.clip(A, -10, 10)  # Prevent exp overflow
    return np.exp(clamped)


@error_state_decorator
def sign_op(A):
    """Sign function: -1, 0, or 1"""
    return np.sign(A)


# ============================================================================
# Cross-sectional operators
# ============================================================================

@error_state_decorator
def scale(A, scaler=1):
    """Scale to sum to scaler (default 1) across stocks"""
    ret = pd.DataFrame(A)
    factor = scaler * ret.div(ret.sum(axis=1), axis=0).to_numpy(dtype=np.double)
    return factor


@error_state_decorator
def rank(A):
    """Cross-sectional rank (ascending, normalized to [1, n_stocks])"""
    ret = pd.DataFrame(A)
    factor = ret.rank(axis=1).to_numpy(dtype=np.double)
    return factor


# ============================================================================
# Time series operators
# ============================================================================

@error_state_decorator
def delay(A, window=1):
    """Shift time series backward by window periods"""
    ret = pd.DataFrame(A)
    factor = ret.shift(window).to_numpy(dtype=np.double)
    return factor


@error_state_decorator
def delta(A, window=1):
    """Difference: A(t) - A(t-window)"""
    ret = pd.DataFrame(A)
    factor = ret.diff(window).to_numpy(dtype=np.double)
    return factor


# ============================================================================
# Rolling window operators (fixed window size)
# ============================================================================

@error_state_decorator
def rolling_nanmean(A, window=5):
    """Rolling mean over window periods"""
    ret = pd.DataFrame(A)
    factor = ret.rolling(window, min_periods=1).mean().to_numpy(dtype=np.double)
    return factor


@error_state_decorator
def rolling_nanstd(A, window=5):
    """Rolling standard deviation over window periods"""
    ret = pd.DataFrame(A)
    factor = ret.rolling(window, min_periods=2).std().to_numpy(dtype=np.double)
    return factor


@error_state_decorator
def rolling_max(A, window=5):
    """Rolling maximum over window periods"""
    ret = pd.DataFrame(A)
    factor = ret.rolling(window, min_periods=1).max().to_numpy(dtype=np.double)
    return factor


@error_state_decorator
def rolling_min(A, window=5):
    """Rolling minimum over window periods"""
    ret = pd.DataFrame(A)
    factor = ret.rolling(window, min_periods=1).min().to_numpy(dtype=np.double)
    return factor


@error_state_decorator
def rolling_sum(A, window=5):
    """Rolling sum over window periods"""
    ret = pd.DataFrame(A)
    factor = ret.rolling(window, min_periods=1).sum().to_numpy(dtype=np.double)
    return factor


@error_state_decorator
def rolling_covariance(A, B, window=5):
    """Rolling covariance between A and B"""
    ret1 = pd.DataFrame(A)
    ret2 = pd.DataFrame(B)
    factor = ret1.rolling(window, min_periods=2).cov(ret2).to_numpy(dtype=np.double)
    return factor


@error_state_decorator
def rolling_correlation(A, B, window=5):
    """Rolling correlation between A and B"""
    ret1 = pd.DataFrame(A)
    ret2 = pd.DataFrame(B)
    factor = ret1.rolling(window, min_periods=2).corr(ret2).to_numpy(dtype=np.double)
    return factor


@error_state_decorator
def rolling_argmin(A, window=5):
    """Index of minimum value in rolling window (0 to window-1)"""
    ret = pd.DataFrame(A)
    factor = ret.rolling(window, min_periods=1).apply(lambda x: np.argmin(x), raw=True).to_numpy(dtype=np.double)
    return factor


@error_state_decorator
def rolling_argmax(A, window=5):
    """Index of maximum value in rolling window (0 to window-1)"""
    ret = pd.DataFrame(A)
    factor = ret.rolling(window, min_periods=1).apply(lambda x: np.argmax(x), raw=True).to_numpy(dtype=np.double)
    return factor


@error_state_decorator
def rolling_rank(A, window=5):
    """Time-series rank: rank of current value within rolling window"""
    ret = pd.DataFrame(A)
    factor = ret.rolling(window, min_periods=1).rank().to_numpy(dtype=np.double)
    return factor


@error_state_decorator
def rolling_skew(A, window=20):
    """Rolling skewness"""
    ret = pd.DataFrame(A)
    min_periods = min(3, window)  # min_periods 不能超过 window
    factor = ret.rolling(window, min_periods=min_periods).skew().to_numpy(dtype=np.double)
    return factor


@error_state_decorator
def rolling_kurt(A, window=20):
    """Rolling kurtosis"""
    ret = pd.DataFrame(A)
    min_periods = min(4, window)  # min_periods 不能超过 window
    factor = ret.rolling(window, min_periods=min_periods).kurt().to_numpy(dtype=np.double)
    return factor


@error_state_decorator
def pow(A, pow=2):
    """Power function: A^pow"""
    return np.power(A, pow)


@error_state_decorator
def tanh_op(A):
    """Hyperbolic tangent non-linearity"""
    return np.tanh(A)


@error_state_decorator
def pos_op(A):
    """Positive part: max(A, 0)"""
    return np.maximum(A, 0.0)


@error_state_decorator
def neg_part_op(A):
    """Negative part: min(A, 0)"""
    return np.minimum(A, 0.0)


@error_state_decorator
def ts_zscore_op(A, window=60, eps=1e-9):
    """
    Time-series z-score normalization:
    (A_t - rolling_mean) / rolling_std over a window.
    """
    ret = pd.DataFrame(A)
    mean = ret.rolling(window, min_periods=5).mean()
    std = ret.rolling(window, min_periods=5).std()
    factor = ret.sub(mean, axis=0).divide(std.replace(0, np.nan) + eps, axis=0)
    return factor.to_numpy(dtype=np.double)


@error_state_decorator
def ts_rank_op(A, window=60):
    """
    Time-series percentile rank within rolling window.
    Rank is normalized to [0, 1].
    """
    ret = pd.DataFrame(A)
    def _rank_pct(x):
        r = x.rank()
        return (r.iloc[-1] - 1) / max(len(x) - 1, 1)
    factor = ret.rolling(window, min_periods=5).apply(_rank_pct, raw=False)
    return factor.to_numpy(dtype=np.double)


@error_state_decorator
def ts_ema_op(A, span=20):
    """
    Exponential moving average over time (per stock).
    Using pandas ewm with a span parameter.
    """
    ret = pd.DataFrame(A)
    factor = ret.ewm(span=span, min_periods=5, adjust=False).mean().to_numpy(dtype=np.double)
    return factor


@error_state_decorator
def cs_zscore(A, eps=1e-9):
    """Cross-sectional z-score normalization"""
    ret = pd.DataFrame(A)
    mean = ret.mean(axis=1)
    std = ret.std(axis=1)
    # Prevent zero-variance rows from exploding
    factor = ret.sub(mean, axis=0).divide(std.replace(0, np.nan) + eps, axis=0)
    return factor.to_numpy(dtype=np.double)


@error_state_decorator
def safe_div(A, B, min_abs=5e-2):
    """Safe division with clamping to avoid divide-by-zero"""
    retA = pd.DataFrame(A)
    retB = pd.DataFrame(B)
    numer = retA.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.double)
    denom = retB.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.double)
    # Clamp very small denominators
    denom = np.where(np.abs(denom) < min_abs, np.where(denom >= 0, min_abs, -min_abs), denom)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = numer / denom
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


# ============================================================================
# Function map for gplearn 3D
# All operators registered here can be used in genetic programming
# ============================================================================

_extra_function_map = {
    # Basic arithmetic (unary)
    'abs': _Function(function=abs_op, name='abs', arity=1,
                     role='arith', is_ts_op=False, complexity_weight=1.5),
    'sqrt': _Function(function=sqrt_op, name='sqrt', arity=1,
                      role='arith', is_ts_op=False, complexity_weight=1.5),
    'log': _Function(function=log_op, name='log', arity=1,
                     role='arith', is_ts_op=False, complexity_weight=1.5),
    'exp': _Function(function=exp_op, name='exp', arity=1,
                     role='arith', is_ts_op=False, complexity_weight=2.0),
    'sign': _Function(function=sign_op, name='sign', arity=1,
                      role='arith', is_ts_op=False, complexity_weight=1.0),
    'pow': _Function(function=pow, name='pow', arity=1, isRandom=(True, (2, 3)),
                     role='arith', is_ts_op=False, complexity_weight=2.0),
    'tanh': _Function(function=tanh_op, name='tanh', arity=1,
                      role='arith', is_ts_op=False, complexity_weight=1.5),
    
    # Cross-sectional operators
    'scale': _Function(function=scale, name='scale', arity=1,
                       role='cs', is_ts_op=False, complexity_weight=1.5),
    'rank': _Function(function=rank, name='rank', arity=1,
                      role='cs', is_ts_op=False, complexity_weight=2.0),
    'cs_zscore': _Function(function=cs_zscore, name='cs_zscore', arity=1,
                           role='cs', is_ts_op=False, complexity_weight=2.0),
    
    # Safe division
    'safe_div': _Function(function=safe_div, name='safe_div', arity=2,
                          role='arith', is_ts_op=False, complexity_weight=2.5),
    
    # Time series operators (fixed window)
    'delay': _Function(function=delay, name='delay', arity=1,
                       role='lag', is_ts_op=False, complexity_weight=0.8),
    'delta': _Function(function=delta, name='delta', arity=1,
                       role='lag', is_ts_op=False, complexity_weight=1.0),
    
    # Rolling window operators (dynamic window size: 5-60)
    'dynamic_ts_mean': _Function(function=rolling_nanmean, name='dynamic_ts_mean', arity=1,
                                 isRandom=(True, (5, 60)),
                                 role='ts_agg', is_ts_op=True, complexity_weight=3.0),
    'dynamic_ts_std': _Function(function=rolling_nanstd, name='dynamic_ts_std', arity=1,
                                isRandom=(True, (5, 60)),
                                role='ts_agg', is_ts_op=True, complexity_weight=3.5),
    'dynamic_ts_max': _Function(function=rolling_max, name='dynamic_ts_max', arity=1,
                                isRandom=(True, (5, 60)),
                                role='ts_agg', is_ts_op=True, complexity_weight=3.5),
    'dynamic_ts_min': _Function(function=rolling_min, name='dynamic_ts_min', arity=1,
                                isRandom=(True, (5, 60)),
                                role='ts_agg', is_ts_op=True, complexity_weight=3.0),
    'dynamic_ts_sum': _Function(function=rolling_sum, name='dynamic_ts_sum', arity=1,
                                isRandom=(True, (5, 60)),
                                role='ts_agg', is_ts_op=True, complexity_weight=3.0),
    'dynamic_ts_rank': _Function(function=rolling_rank, name='dynamic_ts_rank', arity=1,
                                 isRandom=(True, (5, 60)),
                                 role='ts_norm', is_ts_op=True, complexity_weight=3.0),
    'dynamic_ts_argmax': _Function(function=rolling_argmax, name='dynamic_ts_argmax', arity=1,
                                   isRandom=(True, (15, 60)),
                                   role='ts_agg', is_ts_op=True, complexity_weight=3.5),
    'dynamic_ts_argmin': _Function(function=rolling_argmin, name='dynamic_ts_argmin', arity=1,
                                   isRandom=(True, (15, 60)),
                                   role='ts_agg', is_ts_op=True, complexity_weight=3.5),
    
    # Rolling correlation/covariance (binary operators with dynamic window)
    'dynamic_ts_cov': _Function(function=rolling_covariance, name='dynamic_ts_cov', arity=2,
                                isRandom=(True, (5, 60)),
                                role='ts_agg', is_ts_op=True, complexity_weight=4.0),
    'dynamic_ts_corr': _Function(function=rolling_correlation, name='dynamic_ts_corr', arity=2,
                                 isRandom=(True, (5, 60)),
                                 role='ts_agg', is_ts_op=True, complexity_weight=4.0),
    
    # Higher-order statistics
    'dynamic_ts_skew': _Function(function=rolling_skew, name='dynamic_ts_skew', arity=1,
                                 isRandom=(True, (10, 60)),
                                 role='ts_agg', is_ts_op=True, complexity_weight=3.5),
    'dynamic_ts_kurt': _Function(function=rolling_kurt, name='dynamic_ts_kurt', arity=1,
                                 isRandom=(True, (10, 60)),
                                 role='ts_agg', is_ts_op=True, complexity_weight=3.5),
    
    # Time-series normalization / sign-split helpers
    'ts_zscore': _Function(function=ts_zscore_op, name='ts_zscore', arity=1,
                           isRandom=(True, (20, 120)),
                           role='ts_norm', is_ts_op=True, complexity_weight=3.0),
    'ts_rank': _Function(function=ts_rank_op, name='ts_rank', arity=1,
                         isRandom=(True, (20, 120)),
                         role='ts_norm', is_ts_op=True, complexity_weight=3.0),
    'pos': _Function(function=pos_op, name='pos', arity=1,
                     role='lag', is_ts_op=False, complexity_weight=1.5),
    'neg_part': _Function(function=neg_part_op, name='neg_part', arity=1,
                          role='lag', is_ts_op=False, complexity_weight=1.8),
    'ts_ema': _Function(function=ts_ema_op, name='ts_ema', arity=1,
                        isRandom=(True, (10, 120)),
                        role='ts_agg', is_ts_op=True, complexity_weight=3.0),
}