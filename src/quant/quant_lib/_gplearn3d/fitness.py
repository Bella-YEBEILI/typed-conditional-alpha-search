"""Metrics to evaluate the fitness of a program.

The :mod:`gplearn.fitness` module contains some metric with which to evaluate
the computer programs created by the :mod:`gplearn.genetic` module.
"""

import numbers
import copy
import numpy as np
import pandas as pd
from joblib import wrap_non_picklable_objects
from scipy.stats import rankdata

__all__ = ['make_fitness']


class _Fitness(object):

    """A metric to measure the fitness of a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        return self.function(*args)


def make_fitness(*, function, greater_is_better, wrap=True):
    """Make a fitness measure, a metric scoring the quality of a program's fit.

    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic
    operations into the next generation. The resulting object is able to be
    called with NumPy vectorized arguments and return a resulting floating
    point score quantifying the quality of the program's representation of the
    true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom metrics is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))
    if function.__code__.co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.' % function.__code__.co_argcount)
    if not isinstance(function(np.array([1, 1]),
                      np.array([2, 2]),
                      np.array([1, 1])), numbers.Number):
        raise ValueError('function must return a numeric.')

    if wrap:
        return _Fitness(function=wrap_non_picklable_objects(function),
                        greater_is_better=greater_is_better)
    return _Fitness(function=function,
                    greater_is_better=greater_is_better)


def _weighted_pearson(y, y_pred, w):
    """Calculate the weighted Pearson correlation coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=w)
        y_demean = y - np.average(y, weights=w)
        corr = ((np.sum(w * y_pred_demean * y_demean) / np.sum(w)) /
                np.sqrt((np.sum(w * y_pred_demean ** 2) *
                         np.sum(w * y_demean ** 2)) /
                        (np.sum(w) ** 2)))
    if np.isfinite(corr):
        return np.abs(corr)
    return 0.


def _weighted_spearman(y, y_pred, w):
    """Calculate the weighted Spearman correlation coefficient."""
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    return _weighted_pearson(y_pred_ranked, y_ranked, w)


def _mean_absolute_error(y, y_pred, w):
    """Calculate the mean absolute error."""
    return np.average(np.average(np.abs(y_pred - y), weights=w, axis=0))


def _mean_square_error(y, y_pred, w):
    """Calculate the mean square error."""
    return np.average(np.average(((y_pred - y) ** 2), weights=w, axis=0))


def _root_mean_square_error(y, y_pred, w):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(np.average(((y_pred - y) ** 2), weights=w, axis=0)))


def _log_loss(y, y_pred, w):
    """Calculate the log loss."""
    eps = 1e-15
    inv_y_pred = np.clip(1 - y_pred, eps, 1 - eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    score = y * np.log(y_pred) + (1 - y) * np.log(inv_y_pred)
    return np.average(np.average(-score, weights=w, axis=0))


weighted_pearson = _Fitness(function=_weighted_pearson,
                            greater_is_better=True)
weighted_spearman = _Fitness(function=_weighted_spearman,
                             greater_is_better=True)
mean_absolute_error = _Fitness(function=_mean_absolute_error,
                               greater_is_better=False)
mean_square_error = _Fitness(function=_mean_square_error,
                             greater_is_better=False)
root_mean_square_error = _Fitness(function=_root_mean_square_error,
                                  greater_is_better=False)
log_loss = _Fitness(function=_log_loss,
                    greater_is_better=False)

_fitness_map = {'pearson': weighted_pearson,
                'spearman': weighted_spearman,
                'mean absolute error': mean_absolute_error,
                'mse': mean_square_error,
                'rmse': root_mean_square_error,
                'log loss': log_loss
}


def compute_ic(y, y_pred, w, rank_ic=True):
    y = y[w.astype(bool)]
    y_pred = y_pred[w.astype(bool)]
    if rank_ic:
        ic = pd.DataFrame(y_pred).corrwith(pd.DataFrame(y), axis=1, method="spearman")
    else:
        ic = pd.DataFrame(y_pred).corrwith(pd.DataFrame(y), axis=1, method="pearson")
    return ic 


def _normal_ic(y, y_pred, w):
    ic = compute_ic(y, y_pred, w, rank_ic=False).mean()
    if np.isnan(ic):
        return 0
    else:
        return abs(ic)


def _rank_ic(y, y_pred, w):
    ic = compute_ic(y, y_pred, w).mean()
    if np.isnan(ic):
        return 0
    else:
        return ic


def _rank_icir(y, y_pred, w):
    ics = compute_ic(y, y_pred, w)
    ic = ics.mean()
    ic_std = ics.std()
    icir = ic / ic_std
    if np.isnan(icir):
        return 0
    else:
        return icir


def compute_quantile10_rets(y, y_pred, w):
    y_pred = y_pred[w.astype(bool)]
    y = y[w.astype(bool)]
    if np.all(np.isnan(y_pred)):
        return None
    
    quantiles = 10
    annulization = 252  # default annual return on a daily basis
    groups = np.array(range(quantiles)) + 1

    factor_quantiles = pd.DataFrame(y_pred).rank(axis=1,method='first').dropna(axis=0, how='all').apply(pd.qcut, q=quantiles, labels=groups, axis=1)

    rets = pd.DataFrame(y)
    return_series = {}
    for group in groups:
        returns_group = rets[factor_quantiles == group]
        return_series[group] = (returns_group.sum(axis=1) / returns_group.count(axis=1)).mean() * annulization  # scale holding to 1; equal weights
    return return_series


def _quantile10_max(y, y_pred, w):
    res = compute_quantile10_rets(y, y_pred, w)
    if res is None:
        return 0
    else:
        return max(res.values())


def measure_monotonicity(data):
    ranks = [sorted(data).index(x) + 1 for x in data]
    rank_differences = [ranks[i] - ranks[i-1] for i in range(1, len(ranks))]
    positive_differences = sum(1 for diff in rank_differences if diff > 0)
    negative_differences = sum(1 for diff in rank_differences if diff < 0)
    monotonicity_score = abs(positive_differences - negative_differences) / len(data)
    return monotonicity_score


def _quantile10_monotonicity(y, y_pred, w):
    res = compute_quantile10_rets(y, y_pred, w)
    if res is None:
        return 0
    else:
        return measure_monotonicity(res.values())


def make_fitness_ir(quantile=0.2, n_trading_days=252, abs_ret_clip=None, horizons=None, horizon_weights=None, neutralize_fn=None):
    """
    Create a custom IR (Information Ratio) fitness function for long-short strategy.
    
    This fitness measures the annualized IR of a quantile-based long-short portfolio,
    with automatic direction alignment (returns max of IR and IR of negated factor).
    Supports multi-horizon IR blending for robustness.
    
    Parameters
    ----------
    quantile : float, default=0.2
        The quantile threshold for top/bottom groups (e.g., 0.2 for top/bottom 20%)
    
    n_trading_days : int, default=252
        Number of trading days per year for annualization
    
    abs_ret_clip : float, optional
        Clip extreme daily returns to [-abs_ret_clip, +abs_ret_clip] to reduce outlier impact
    
    horizons : list of int, optional
        Multiple forward-return horizons to blend (e.g., [5, 20]). If None, uses single horizon from y.
    
    horizon_weights : list of float, optional
        Weights for each horizon (must sum to 1). If None, uses equal weights.
    
    neutralize_fn : callable, optional
        Function to neutralize y_pred (industry/style de-meaning). Signature: neutralize_fn(y_pred_2d) -> y_pred_2d_neutralized
    
    Returns
    -------
    fitness : _Fitness
        A fitness function compatible with gplearn 3D
    """
    def _fitness_ir(y, y_pred, w):
        try:
            y = y[w.astype(bool)]
            y_pred = y_pred[w.astype(bool)]
            if y.size == 0 or y_pred.size == 0:
                return 0.0
            
            # 中性化处理（行业/风格去暴露）
            if neutralize_fn is not None:
                try:
                    y_pred = neutralize_fn(y_pred)
                except Exception:
                    pass  # 中性化失败则使用原始信号
            
            # 多视野合成逻辑
            # 只要指定了 horizons（即使长度为1），就使用滚动累计口径
            if horizons is None:
                # 单视野模式（回退到原逻辑）
                r = pd.DataFrame(y)
                f = pd.DataFrame(y_pred)
                if abs_ret_clip is not None:
                    try:
                        r = r.clip(lower=-float(abs_ret_clip), upper=float(abs_ret_clip))
                    except Exception:
                        pass

                def _ir_from_factor(F: pd.DataFrame) -> float:
                    ranking = F.rank(axis=1, pct=True)
                    top = r[ranking < quantile].mean(axis=1)
                    bot = r[ranking > 1 - quantile].mean(axis=1)
                    ls = (bot - top) / 2.0
                    mu = ls.mean()
                    sigma = ls.std()
                    if not np.isfinite(sigma) or sigma <= 0:
                        return 0.0
                    ann_ret = n_trading_days * mu
                    ann_vol = np.sqrt(n_trading_days) * sigma
                    ir = ann_ret / ann_vol
                    if not np.isfinite(ir):
                        return 0.0
                    return float(ir)

                ir_pos = _ir_from_factor(pd.DataFrame(y_pred))
                ir_neg = _ir_from_factor(pd.DataFrame(-y_pred))
                return max(ir_pos, ir_neg)
            else:
                # 多视野模式：对每个 horizon 计算 IR 并加权合成
                ws = horizon_weights if horizon_weights is not None else [1.0/len(horizons)] * len(horizons)
                if abs(sum(ws) - 1.0) > 1e-6:
                    ws = [w / sum(ws) for w in ws]  # 归一化权重
                
                r_orig = pd.DataFrame(y)
                if abs_ret_clip is not None:
                    try:
                        r_orig = r_orig.clip(lower=-float(abs_ret_clip), upper=float(abs_ret_clip))
                    except Exception:
                        pass
                
                def _ir_multi_horizon(F: pd.DataFrame) -> float:
                    ir_vals = []
                    for h in horizons:
                        # 前瞻 h 期收益（简化：累加 h 期单日收益）
                        r_h = r_orig.rolling(window=h, min_periods=1).sum()
                        ranking = F.rank(axis=1, pct=True)
                        top = r_h[ranking < quantile].mean(axis=1)
                        bot = r_h[ranking > 1 - quantile].mean(axis=1)
                        ls = (bot - top) / 2.0
                        mu = ls.mean()
                        sigma = ls.std()
                        if not np.isfinite(sigma) or sigma <= 0:
                            ir_vals.append(0.0)
                        else:
                            # 年化：h期累计收益的IR需按 sqrt(252/h) 调整
                            scale = np.sqrt(n_trading_days / h)
                            ann_ret = (n_trading_days / h) * mu
                            ann_vol = scale * sigma
                            ir = ann_ret / ann_vol if ann_vol > 0 else 0.0
                            ir_vals.append(float(ir) if np.isfinite(ir) else 0.0)
                    # 加权合成
                    blended_ir = sum(w * ir for w, ir in zip(ws, ir_vals))
                    return blended_ir
                
                f = pd.DataFrame(y_pred)
                ir_pos = _ir_multi_horizon(f)
                ir_neg = _ir_multi_horizon(-f)
                return max(ir_pos, ir_neg)
        except Exception:
            return 0.0
    
    return _Fitness(function=_fitness_ir, greater_is_better=True)


def make_fitness_icir_ir(quantile=0.2,
                         n_trading_days=252,
                         abs_ret_clip=None,
                         horizons=None,
                         horizon_weights=None,
                         neutralize_fn=None,
                         w_icir=0.7):
    """
    Create a combined fitness = w_icir * ICIR + (1-w_icir) * IR (annualized),
    with automatic direction alignment and optional multi-horizon blending.
    ICIR component is scaled by sqrt(n_trading_days / h) for horizon h.
    """
    w_icir = float(w_icir)
    if w_icir < 0.0: w_icir = 0.0
    if w_icir > 1.0: w_icir = 1.0
    def _fitness_icir_ir(y, y_pred, w):
        try:
            y = y[w.astype(bool)]
            y_pred = y_pred[w.astype(bool)]
            if y.size == 0 or y_pred.size == 0:
                return 0.0
            # 中性化
            if neutralize_fn is not None:
                try:
                    y_pred = neutralize_fn(y_pred)
                except Exception:
                    pass
            r_orig = pd.DataFrame(y)
            if abs_ret_clip is not None:
                try:
                    r_orig = r_orig.clip(lower=-float(abs_ret_clip), upper=float(abs_ret_clip))
                except Exception:
                    pass
            ws = None
            if horizons is not None and isinstance(horizons, list) and len(horizons) > 0:
                ws = horizon_weights if horizon_weights is not None else [1.0/len(horizons)] * len(horizons)
                s = sum(ws)
                if s <= 0: ws = [1.0/len(horizons)] * len(horizons)
                else: ws = [w_i / s for w_i in ws]
            def _blend_metrics(F: pd.DataFrame):
                # 单/多视野 ICIR 与 IR
                if not horizons or len(horizons) == 1:
                    # 使用原始 y（已是前瞻收益）
                    # IR
                    ranking = F.rank(axis=1, pct=True)
                    top = r_orig[ranking < quantile].mean(axis=1)
                    bot = r_orig[ranking > 1 - quantile].mean(axis=1)
                    ls = (bot - top) / 2.0
                    mu = ls.mean()
                    sigma = ls.std()
                    ir = 0.0
                    if np.isfinite(sigma) and sigma > 0:
                        ann_ret = n_trading_days * mu
                        ann_vol = np.sqrt(n_trading_days) * sigma
                        ir = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
                    # ICIR（rank IC）
                    ics = compute_ic(r_orig.values, F.values, np.ones((r_orig.shape[0],), dtype=float), rank_ic=True)
                    ic = ics.mean()
                    ic_std = ics.std()
                    icir = float(ic / ic_std) if np.isfinite(ic_std) and ic_std > 0 else 0.0
                    # 年化尺度（与 horizon=1 对齐）：sqrt(n_trading_days)
                    icir_ann = float(icir) * np.sqrt(n_trading_days)
                    return w_icir * icir_ann + (1.0 - w_icir) * ir
                else:
                    icir_vals = []
                    ir_vals = []
                    for h, w_h in zip(horizons, ws):
                        try:
                            h_int = int(h)
                        except Exception:
                            h_int = None
                        if not h_int or h_int <= 0:
                            continue
                        # 前瞻h期合成（与IR一致采用滚动和口径，因 y 已是前瞻）
                        r_h = r_orig.rolling(window=h_int, min_periods=1).sum()
                        ranking = F.rank(axis=1, pct=True)
                        top = r_h[ranking < quantile].mean(axis=1)
                        bot = r_h[ranking > 1 - quantile].mean(axis=1)
                        ls = (bot - top) / 2.0
                        mu = ls.mean()
                        sigma = ls.std()
                        if np.isfinite(sigma) and sigma > 0:
                            scale = np.sqrt(n_trading_days / h_int)
                            ann_ret = (n_trading_days / h_int) * mu
                            ann_vol = scale * sigma
                            ir_h = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
                        else:
                            ir_h = 0.0
                        ir_vals.append(w_h * ir_h)
                        # ICIR_h
                        ics = compute_ic(r_h.values, F.values, np.ones((r_h.shape[0],), dtype=float), rank_ic=True)
                        ic = ics.mean()
                        ic_std = ics.std()
                        icir_h = float(ic / ic_std) if np.isfinite(ic_std) and ic_std > 0 else 0.0
                        icir_h_ann = icir_h * np.sqrt(n_trading_days / h_int)
                        icir_vals.append(w_h * icir_h_ann)
                    ir_blend = sum(ir_vals) if len(ir_vals) > 0 else 0.0
                    icir_blend = sum(icir_vals) if len(icir_vals) > 0 else 0.0
                    return w_icir * icir_blend + (1.0 - w_icir) * ir_blend
            f = pd.DataFrame(y_pred)
            score_pos = _blend_metrics(f)
            score_neg = _blend_metrics(-f)
            return max(score_pos, score_neg)
        except Exception:
            return 0.0
    return _Fitness(function=_fitness_icir_ir, greater_is_better=True)

weighted_normal_ic = _Fitness(function=_normal_ic, greater_is_better=True)
weighted_rank_ic = _Fitness(function=_rank_ic, greater_is_better=True)
weighted_rank_icir = _Fitness(function=_rank_icir, greater_is_better=True)
weighted_quantile_max = _Fitness(function=_quantile10_max, greater_is_better=True)
weighted_quantile_mono = _Fitness(function=_quantile10_monotonicity, greater_is_better=True)

_extra_map = {
    "normal_ic": weighted_normal_ic,
    "rank_ic": weighted_rank_ic,
    "rank_icir": weighted_rank_icir,
    "quantile_max": weighted_quantile_max,
    "quantile_mono": weighted_quantile_mono,
}

_fitness_map = dict(_fitness_map, **_extra_map)


