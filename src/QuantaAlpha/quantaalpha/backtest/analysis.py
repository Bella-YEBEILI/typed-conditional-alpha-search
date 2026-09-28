import operator

import numpy as np
import pandas as pd
from scipy.stats import norm
from .numbafunc import *

# math 
def neg(df):
    return -df

def abs(df):
    return df.abs()

def signal(df):
    return np.sign(df)

def sign(df):
    return np.sign(df)

def power(df,power=2):
    return df**power

def sqrt(df):
    with np.errstate(invalid='ignore'):
        return np.sqrt(df)

def log(df):
    with np.errstate(divide='ignore',invalid='ignore'):
        log_df = np.log(df)
        return log_df.replace([np.inf,-np.inf], np.nan)

def diff(df):
    return df-df.shift(1)

def diff_q(df):
    return (df/df.shift(1)-1).replace([np.inf,-np.inf],np.nan)

def tanh(df):
    return np.tanh(df)

def pos(df): 
    return np.maximum(df, 0.0)

def neg_part(df):
    return np.minimum(df,0.0)

def add(*args):
    if len(args) == 2:
        return _apply_aligned_binary_op(args[0], args[1], np.add)
    from functools import reduce
    return reduce(lambda a, b: _apply_aligned_binary_op(a, b, np.add), args)

def sub(df1,df2):
    return _apply_aligned_binary_op(df1, df2, np.subtract)

def mul(*args):
    if len(args) == 2:
        return _apply_aligned_binary_op(args[0], args[1], np.multiply)
    from functools import reduce
    return reduce(lambda a, b: _apply_aligned_binary_op(a, b, np.multiply), args)

def div(df1,df2):
    return _apply_aligned_binary_op(df1, df2, np.divide)

def pct(df1,df2):
    return div(df1,df2)-1

def GT(df1,df2):
    return _apply_aligned_comparison_op(df1, df2, operator.gt)

def LT(df1,df2):
    return _apply_aligned_comparison_op(df1, df2, operator.lt)

def GE(df1,df2):
    return _apply_aligned_comparison_op(df1, df2, operator.ge)

def LE(df1,df2):
    return _apply_aligned_comparison_op(df1, df2, operator.le)

def EQ(df1,df2):
    return _apply_aligned_comparison_op(df1, df2, operator.eq)

def NE(df1,df2):
    return _apply_aligned_comparison_op(df1, df2, operator.ne)

def gt(df1,df2):
    return GT(df1,df2)

def lt(df1,df2):
    return LT(df1,df2)

def ge(df1,df2):
    return GE(df1,df2)

def le(df1,df2):
    return LE(df1,df2)

def eq(df1,df2):
    return EQ(df1,df2)

def ne(df1,df2):
    return NE(df1,df2)

def AND(df1,df2):
    return _apply_aligned_logical_op(df1, df2, operator.and_)

def OR(df1,df2):
    return _apply_aligned_logical_op(df1, df2, operator.or_)

def and_(df1,df2):
    return AND(df1,df2)

def or_(df1,df2):
    return OR(df1,df2)

def WHERE(condition,true_value,false_value):
    if isinstance(true_value, pd.DataFrame):
        ref = true_value
    elif isinstance(false_value, pd.DataFrame):
        ref = false_value
    elif isinstance(condition, pd.DataFrame):
        ref = condition
    else:
        ref = true_value if isinstance(true_value, pd.Series) else (
            false_value if isinstance(false_value, pd.Series) else (
                condition if isinstance(condition, pd.Series) else None
            )
        )
    cond = _coerce_condition_like_frame(condition, reference=ref) if ref is not None else condition
    lhs = _coerce_like_frame(true_value, reference=ref) if ref is not None else true_value
    rhs = _coerce_like_frame(false_value, reference=ref) if ref is not None else false_value
    if isinstance(ref, pd.DataFrame):
        if isinstance(cond, pd.DataFrame):
            cond = _broadcast_frame_to_reference(cond, ref)
        if isinstance(lhs, pd.DataFrame):
            lhs = _broadcast_frame_to_reference(lhs, ref)
        if isinstance(rhs, pd.DataFrame):
            rhs = _broadcast_frame_to_reference(rhs, ref)
    if isinstance(lhs, (pd.Series, pd.DataFrame)):
        return lhs.where(cond, rhs).replace([np.inf, -np.inf], np.nan)
    out = np.where(cond, lhs, rhs)
    if np.isscalar(out):
        return np.nan if not np.isfinite(out) else out
    out = np.asarray(out, dtype=float)
    out[~np.isfinite(out)] = np.nan
    return out

def where(condition,true_value,false_value):
    return WHERE(condition,true_value,false_value)

def IF_ELSE(condition,true_value,false_value):
    return WHERE(condition,true_value,false_value)

def if_else(condition,true_value,false_value):
    return WHERE(condition,true_value,false_value)

def ifelse(condition,true_value,false_value):
    return WHERE(condition,true_value,false_value)

def last(value):
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        return arr.copy()
    if arr.shape[0] == 0:
        return np.full(arr.shape[1:], np.nan)
    return arr[-1].copy()
    
def _coerce_like_frame(value, reference=None):
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        if isinstance(reference, pd.DataFrame):
            if value.index.equals(reference.index):
                return pd.DataFrame(
                    np.repeat(value.to_numpy(dtype=float)[:, None], len(reference.columns), axis=1),
                    index=reference.index,
                    columns=reference.columns,
                )
            if value.index.equals(reference.columns):
                return pd.DataFrame(
                    np.repeat(value.to_numpy(dtype=float)[None, :], len(reference.index), axis=0),
                    index=reference.index,
                    columns=reference.columns,
                )
        return value.to_frame()
    if np.isscalar(value):
        if isinstance(reference, pd.DataFrame):
            return pd.DataFrame(float(value), index=reference.index, columns=reference.columns)
        if isinstance(reference, pd.Series):
            return pd.Series(float(value), index=reference.index)
        raise TypeError("scalar operands require a DataFrame reference for broadcasting")
    raise TypeError(f"unsupported operand type: {type(value)}")


def _coerce_condition_like_frame(value, reference=None):
    if isinstance(value, pd.DataFrame):
        return value.fillna(False).astype(bool)
    if isinstance(value, pd.Series):
        series = value.fillna(False).astype(bool)
        if isinstance(reference, pd.DataFrame):
            if series.index.equals(reference.index):
                return pd.DataFrame(
                    np.repeat(series.to_numpy(dtype=bool)[:, None], len(reference.columns), axis=1),
                    index=reference.index,
                    columns=reference.columns,
                )
            if series.index.equals(reference.columns):
                return pd.DataFrame(
                    np.repeat(series.to_numpy(dtype=bool)[None, :], len(reference.index), axis=0),
                    index=reference.index,
                    columns=reference.columns,
                )
        return series.to_frame()
    if np.isscalar(value):
        cond = bool(value) if pd.notna(value) else False
        if isinstance(reference, pd.DataFrame):
            return pd.DataFrame(cond, index=reference.index, columns=reference.columns)
        if isinstance(reference, pd.Series):
            return pd.Series(cond, index=reference.index)
        return cond
    arr = np.asarray(value)
    if arr.dtype == bool:
        return arr
    return np.nan_to_num(arr, nan=0.0).astype(bool)

def _broadcast_frame_to_reference(value, reference):
    if not isinstance(value, pd.DataFrame) or not isinstance(reference, pd.DataFrame):
        return value
    if value.index.equals(reference.index) and value.columns.equals(reference.columns):
        return value
    if value.index.equals(reference.index) and value.shape[1] == 1 and reference.shape[1] > 1:
        return pd.DataFrame(
            np.repeat(value.to_numpy(dtype=float), len(reference.columns), axis=1),
            index=reference.index,
            columns=reference.columns,
        )
    if value.columns.equals(reference.columns) and value.shape[0] == 1 and reference.shape[0] > 1:
        return pd.DataFrame(
            np.repeat(value.to_numpy(dtype=float), len(reference.index), axis=0),
            index=reference.index,
            columns=reference.columns,
        )
    return value.reindex(index=reference.index, columns=reference.columns)

def _align_frame_pair(df1, df2):
    ref = df1 if isinstance(df1, pd.DataFrame) else (df2 if isinstance(df2, pd.DataFrame) else None)
    if ref is None:
        ref = df1 if isinstance(df1, pd.Series) else (df2 if isinstance(df2, pd.Series) else None)
    if ref is None:
        return df1, df2
    lhs = _coerce_like_frame(df1, reference=ref)
    rhs = _coerce_like_frame(df2, reference=lhs if isinstance(lhs, pd.DataFrame) else ref)
    if isinstance(lhs, pd.DataFrame) and isinstance(rhs, pd.DataFrame):
        rhs = _broadcast_frame_to_reference(rhs, lhs)
    return lhs, rhs

def _apply_aligned_binary_op(df1, df2, op):
    lhs, rhs = _align_frame_pair(df1, df2)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = op(lhs, rhs)
    if isinstance(out, (pd.Series, pd.DataFrame)):
        return out.replace([np.inf, -np.inf], np.nan)
    if np.isscalar(out):
        return np.nan if not np.isfinite(out) else out
    out = np.asarray(out, dtype=float)
    out[~np.isfinite(out)] = np.nan
    return out

def _apply_aligned_comparison_op(df1, df2, op):
    lhs, rhs = _align_frame_pair(df1, df2)
    return op(lhs, rhs)

def _apply_aligned_logical_op(df1, df2, op):
    lhs, rhs = _align_frame_pair(df1, df2)
    return op(lhs.astype(bool) if hasattr(lhs, "astype") else bool(lhs),
              rhs.astype(bool) if hasattr(rhs, "astype") else bool(rhs))

def safe_div(df1,df2,eps=5e-2):
    lhs, rhs = _align_frame_pair(df1, df2)
    if not isinstance(lhs, pd.DataFrame) or not isinstance(rhs, pd.DataFrame):
        raise TypeError("safe_div expects DataFrame-like operands after broadcasting")
    v1,v2 = lhs.to_numpy(dtype=float),rhs.to_numpy(dtype=float)
    v2 = np.where(np.abs(v2)<eps,np.where(v2>=0,eps,-eps),v2)
    out = v1/v2
    out = np.where(np.isinf(out),np.nan,out)
    return pd.DataFrame(out,index=lhs.index,columns=lhs.columns)

def midmap(df):
    return -abs(cs_zscore(df)-1)

def leftmap(df):
    r = cs_rank(df)
    return -r*log(r)

def rightmap(df):
    r = cs_rank(df)
    return -(1-r)*log(1-r) 

def extreme_rightmap(df):
    r = cs_rank(df)
    return np.exp(-(r-0.85)**2)

def extreme_leftmap(df):
    r = cs_rank(df)
    return np.exp(-(r-0.15)**2)

# time-series
def _coerce_integral_arg(value, name, minimum=None):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer-compatible value, got {value!r}")
    if isinstance(value, (int, np.integer)):
        out = int(value)
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"{name} must be an integer-compatible value, got {value!r}")
        out = int(value)
    else:
        out = int(value)
        if out != value:
            raise ValueError(f"{name} must be an integer-compatible value, got {value!r}")
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {out}")
    return out


def _coerce_window(value, name="window"):
    return _coerce_integral_arg(value, name, minimum=1)


def _coerce_min_periods(value, window):
    if value is None:
        return int(window / 2)
    return _coerce_integral_arg(value, "min_periods", minimum=0)


def ts_delay(df,n):
    n = _coerce_integral_arg(n, "n")
    return df.shift(n)

def ts_delta(df,n):
    n = _coerce_integral_arg(n, "n")
    return df-df.shift(n)

def ts_pct(df,n):
    n = _coerce_integral_arg(n, "n")
    return (df/ts_delay(df,n)-1).replace([np.inf,-np.inf],np.nan)

def ts_rank(df,n,min_periods=None,pct=True):
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    out=ts_rank_2d(values,n,min_periods,bool(pct))
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ts_decay_linear(df,n):
    n = _coerce_window(n, "n")
    values=df.to_numpy(dtype=float)
    out=ts_decay_linear_2d(values,n)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ts_sum(df,n,min_periods=None):
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    out=ts_sum_2d(values,n,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ts_prod(df,n,min_periods=None):
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    out=ts_prod_2d(values,n,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ts_mean(df,n,min_periods=None):
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    out=ts_mean_2d(values,n,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ts_var(df,n,min_periods=None):
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    out=ts_var_2d(values,n,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ts_std(df,n,min_periods=None):
    return sqrt(ts_var(df,n,min_periods))

def ts_ir(df,n,min_periods=None):
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    out=ts_ir_2d(values,n,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ts_skew(df,n,min_periods=None):
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    out=ts_skew_2d(values,n,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ts_kur(df,n,min_periods=None):
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    out=ts_kur_2d(values,n,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ts_min(df,n,min_periods=None):
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    out=ts_min_2d(values,n,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ts_max(df,n,min_periods=None):
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    out=ts_max_2d(values,n,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ts_range(df,n,min_periods=None):
    return ts_max(df,n,min_periods)-ts_min(df,n,min_periods)

def ts_argmin(df,n,min_periods=None):
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    out=ts_argmin_2d(values,n,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ts_argmax(df,n,min_periods=None):
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    out=ts_argmax_2d(values,n,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ts_median(df,n,min_periods=None):
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    out=ts_median_2d(values,n,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ts_zscore(df,n,min_periods=None):
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    out=ts_zscore_2d(values,n,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ema1(df,n,min_periods=None):   # alpha=2/(1+n)
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    alpha=2.0/(1+n)
    out=ema_2d(values,alpha,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ema2(df,n,min_periods=None):   # alpha=1/(1+n)
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    values=df.to_numpy(dtype=float)
    alpha=1.0/(1+n)
    out=ema_2d(values,alpha,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ema3(df,alpha,min_periods=None):   # alpha is given by a (0<a<1)
    if min_periods is None:
        min_periods=2
    else:
        min_periods = _coerce_integral_arg(min_periods, "min_periods", minimum=0)
    values=df.to_numpy(dtype=float)
    out=ema_2d(values,alpha,min_periods)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def wma(df,w):
    values=df.to_numpy(dtype=float)
    out=wma_2d(values,w)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def ts_corr(df1,df2,n,min_periods=None,hold_nan=True):
    """
    hold_nan=True: 当日df1或df2为nan值时,返回nan值
    hold_nan=False: 仍利用窗口内的有效值计算相关系数
    """
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    df1, df2 = _align_frame_pair(df1, df2)
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        raise TypeError("ts_corr expects DataFrame-like operands after broadcasting")
    values1=df1.to_numpy(dtype=np.float64)
    values2=df2.to_numpy(dtype=np.float64)
    out=ts_corr_2d(values1,values2,n,min_periods,hold_nan)
    return pd.DataFrame(out,index=df1.index,columns=df1.columns)

def ts_reg(x,y,n,min_periods=None,rettype=0):
    """
    rettype
    0: 残差
    1: 斜率
    2: 截距
    3: 决定系数
    """
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    x, y = _align_frame_pair(x, y)
    if not isinstance(x, pd.DataFrame) or not isinstance(y, pd.DataFrame):
        raise TypeError("ts_reg expects DataFrame-like operands after broadcasting")
    xv=x.to_numpy(dtype=float)
    yv=y.to_numpy(dtype=float)
    out=ts_reg_2d(xv,yv,n,min_periods,int(rettype))
    return pd.DataFrame(out,index=y.index,columns=y.columns)

def ts_multireg(X_list,y_df,n,min_periods=None,rettype=0):
    """
    rettype
    0: 残差
    3: 决定系数
    """
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    yv = y_df.to_numpy(dtype=np.float64)
    nd,ns = yv.shape
    K = len(X_list)
    Xv = np.empty((nd,ns,K),dtype=np.float64)
    for k in range(K):
        Xv[:,:,k] = X_list[k].to_numpy(dtype=np.float64)
    is_valid = np.isfinite(yv)&np.isfinite(Xv).all(axis=2)
    out = ts_multireg_2d(Xv=Xv,yv=yv,is_valid=is_valid,n=n,min_periods=min_periods,rettype=rettype)
    return pd.DataFrame(out,index=y_df.index,columns=y_df.columns)

# cross-section 
def cs_rank(df,pct=True,ascending=True):
    values=df.to_numpy(dtype=np.float64)
    out=cs_rank_2d(values,pct,ascending)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def cs_neutralize(df):
    values=df.to_numpy(dtype=np.float64)
    out=cs_neutralize_2d(values)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def cs_zscore(df):
    values=df.to_numpy(dtype=np.float64)
    out=cs_zscore_2d(values)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def cs_weighted_zscore(df,weights):
    values=df.to_numpy(dtype=np.float64)
    w=weights.to_numpy(dtype=np.float64)
    out=cs_weighted_zscore_2d(values,w)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def cs_scale(df):
    return df.sub(df.min(axis=1),axis=0).div(df.max(axis=1)-df.min(axis=1),axis=0)

def cs_sum(df):
    return df.sum(axis=1)

def cs_mean(df):
    return df.mean(axis=1)

def cs_std(df):
    return df.std(axis=1)

def cs_median(df):
    return df.median(axis=1)

def cs_corr(df1,df2):
    return df1.T.corrwith(df2.T)

def cs_multireg(X_list,y_df,min_cs=30):
    y = y_df.to_numpy(dtype=np.float64)
    nd,ns = y.shape
    K = len(X_list)
    X = np.empty((nd,ns,K),dtype=np.float64)
    for k in range(K):
        X[:,:,k] = X_list[k].to_numpy(dtype=np.float64)
    out = cs_multireg_2d(X=X,y=y,min_cs=min_cs)
    return pd.DataFrame(out,index=y_df.index,columns=y_df.columns)

def cs_pc1(df_list,min_cs=30):
    """
    每日截面对因子列表中所有因子做pca, 取第一主成分
    """
    base = df_list[0]
    nd,ns = base.shape
    K = len(df_list)
    X = np.empty((nd,ns,K),dtype=np.float64)
    for k in range(K):
        X[:,:,k] = df_list[k].to_numpy(dtype=np.float64)
    out = cs_pc1_2d(X,min_cs)
    return pd.DataFrame(out,index=base.index,columns=base.columns)

def eqw(df_list,method="zscore"):
    result = pd.DataFrame()
    cnt = pd.DataFrame()
    if method=="zscore":
        for df in df_list:
            if result.empty:
                result = cs_zscore(df).fillna(0.0)
                cnt = df.notna()
            else:
                result += cs_zscore(df).fillna(0.0)
                cnt += df.notna()
    elif method=="rank":
        for df in df_list:
            if result.empty:
                result = cs_rank(df).fillna(0.5)
                cnt = df.notna()
            else:
                result += cs_rank(df).fillna(0.5)
                cnt += df.notna()
    return div(result,cnt)

# ts&cs 
def fd(df1,df2,n,weights,method='decay'):
    v1=df1.to_numpy(dtype=float)
    v2=df2.to_numpy(dtype=float)
    z1=cs_zscore_2d(v1)
    z2=cs_zscore_2d(v2)
    w1,w2,w3,w4=float(weights[0]),float(weights[1]),float(weights[2]),float(weights[3])
    sumv=fd_sum_2d(z1,z2,w1,w2,w3,w4)
    if method=='decay':
        out=ts_decay_linear_2d(sumv,n)
    elif method=='mean':
        out=ts_mean_2d(sumv,n)
    else:
        raise ValueError("invalid method")
    return pd.DataFrame(out,index=df1.index,columns=df1.columns)

# group 
def bucket(df,n):
    """
    每一行按截面数值将df打包为0~n-1的n组
    """
    values=df.to_numpy(dtype=float)
    out=bucket_2d(values,n)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def group_rank(df,group):
    """
    每一行按group内掩码分组排序,返回0<=x<=1的浮点数;如果只有一个唯一值,则排序值设为0.5
    group:DataFrame(同shape),元素为组别或NaN;NaN位置输出NaN
    group为字符串market时,等价于截面排序cs_rank
    """
    if isinstance(group,str):
        if group=="market":
            return cs_rank(df)
    values=df.to_numpy(dtype=np.float64)
    garr=group.to_numpy()
    flat=garr.reshape(-1)
    mask=pd.isna(flat)
    s=pd.Series(flat,copy=False)
    codes,uniques=pd.factorize(s[~mask],sort=False)
    gcodes=np.empty(flat.shape[0],dtype=np.int32)
    gcodes[mask]=-1
    gcodes[~mask]=codes.astype(np.int32)
    gcodes=gcodes.reshape(garr.shape)
    gnum=int(uniques.shape[0])
    out=group_rank_2d(values,gcodes,gnum)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def group_neutralize(df,group):
    """
    每一行,每组内数据均减去组内均值
    参数为market时,退化为截面中性化
    """
    if isinstance(group,str):
        if group=="market":
            return cs_neutralize(df)
    values=df.to_numpy(dtype=np.float64)
    garr=group.to_numpy()
    flat=garr.reshape(-1)
    mask=pd.isna(flat)
    s=pd.Series(flat,copy=False)
    codes,uniques=pd.factorize(s[~mask],sort=False)
    gcodes=np.empty(flat.shape[0],dtype=np.int32)
    gcodes[mask]=-1
    gcodes[~mask]=codes.astype(np.int32)
    gcodes=gcodes.reshape(garr.shape)
    gnum=int(uniques.shape[0])
    out=group_neutralize_2d(values,gcodes,gnum)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def group_zscore(df,group):
    """
    每一行做组内zscore标准化
    参数为market时,退化为截面标准化
    """
    if isinstance(group,str):
        if group=="market":
            return cs_zscore(df)
    values=df.to_numpy(dtype=np.float64)
    garr=group.to_numpy()
    flat=garr.reshape(-1)
    mask=pd.isna(flat)
    s=pd.Series(flat,copy=False)
    codes,uniques=pd.factorize(s[~mask],sort=False)
    gcodes=np.empty(flat.shape[0],dtype=np.int32)
    gcodes[mask]=-1
    gcodes[~mask]=codes.astype(np.int32)
    gcodes=gcodes.reshape(garr.shape)
    gnum=int(uniques.shape[0])
    out=group_zscore_2d(values,gcodes,gnum)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def group_scale(df,group):
    if isinstance(group,str):
        if group=="market":
            return cs_scale(df)
    values=df.to_numpy(dtype=np.float64)
    garr=group.to_numpy()
    flat=garr.reshape(-1)
    mask=pd.isna(flat)
    s=pd.Series(flat,copy=False)
    codes,uniques=pd.factorize(s[~mask],sort=False)
    gcodes=np.empty(flat.shape[0],dtype=np.int32)
    gcodes[mask]=-1
    gcodes[~mask]=codes.astype(np.int32)
    gcodes=gcodes.reshape(garr.shape)
    gnum=int(uniques.shape[0])
    out=group_scale_2d(values,gcodes,gnum)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

def group_mean(df,group):
    if isinstance(group,str):
        if group=="market":
            return pd.DataFrame(0.0,index=df.index,columns=df.columns).add(cs_mean(df),axis=0)
    values=df.to_numpy(dtype=np.float64)
    garr=group.to_numpy()
    flat=garr.reshape(-1)
    mask=pd.isna(flat)
    s=pd.Series(flat,copy=False)
    codes,uniques=pd.factorize(s[~mask],sort=False)
    gcodes=np.empty(flat.shape[0],dtype=np.int32)
    gcodes[mask]=-1
    gcodes[~mask]=codes.astype(np.int32)
    gcodes=gcodes.reshape(garr.shape)
    gnum=int(uniques.shape[0])
    out=group_mean_2d(values,gcodes,gnum)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

# timing 
def trade_when(df,con1,con2=-1):
    if not isinstance(con2, pd.DataFrame):
        con2 = pd.DataFrame(False,index=con1.index,columns=con1.columns)
    con1 = con1.fillna(False)
    con2 = con2.fillna(False)
    values = df.to_numpy(dtype=np.float64)
    con1 = con1.to_numpy(dtype=np.bool_)
    con2 = con2.to_numpy(dtype=np.bool_)
    out = trade_when_2d(values,con1,con2)
    return pd.DataFrame(out,index=df.index,columns=df.columns)

# transform
def winsorize1(df,n=4):
    """
    在截面上把偏离均值超过n个标准差以外的值截断为n个标准差
    """
    df_mean = df.mean(axis=1)
    df_std = df.std(axis=1)
    lower = df_mean.sub(n*df_std)
    upper = df_mean.add(n*df_std)
    df_winsorized = df.clip(lower=lower,upper=upper,axis=0)
    return df_winsorized

def winsorize2(df,lower_bound=0.01,upper_bound=0.99):
    lower = df.quantile(lower_bound,axis=1)
    upper = df.quantile(upper_bound,axis=1)
    df_winsorized = df.clip(lower=lower,upper=upper,axis=0)
    return df_winsorized

def gaussian_rank(df):
    """
    在截面上把分布调整为标准正态分布
    """
    ranking = df.rank(axis=1,method='average')
    n_counts = df.count(axis=1)
    ranking = ranking.sub(0.5,axis=0).div(n_counts,axis=0).replace([np.inf,-np.inf],np.nan)
    gaussian_values = norm.ppf(ranking.values)
    gaussian_df = pd.DataFrame(gaussian_values,index=ranking.index,columns=ranking.columns)
    return gaussian_df

# combo
def regime_switch(df):
    pass

# gplearn operator reference
def delay(df,window=1):
    return ts_delay(df,window)

def delta(df,window=1):
    return ts_delta(df,window)

def ts_return(df,n):
    return ts_pct(df,n)

def rank(df):
    return cs_rank(df,pct=False)

# other
def fast_corr(df):
    """
    计算热力矩阵: nxs->sxs
    *近似计算*: 做列zscore时只剔除单列缺失值而非两列缺失值的并集
    """
    values = df.to_numpy(dtype=np.float64)
    values = np.asfortranarray(values)
    out = fast_corr_2d(values)
    return pd.DataFrame(out,index=df.columns,columns=df.columns)

def precise_corr(df):
    """
    精确计算热力矩阵, 与df.corr()完全一致
    """
    values = df.to_numpy(dtype=np.float64)
    values = np.asfortranarray(values)
    out = precise_corr_2d(values)
    return pd.DataFrame(out,index=df.columns,columns=df.columns)

# old
def ts_corr1(df1,df2,n,min_periods=None):  # df1 and df2 have the same shape
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    return df1.rolling(n,min_periods=min_periods).corr(df2)

def ts_corr2(df1,df2,n,min_periods=None):  # df2 is a DataFrame with 1 column
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    df=df2.values+np.zeros(df1.shape)
    df=pd.DataFrame(df,index=df1.index,columns=df1.columns)   
    return df1.rolling(n,min_periods=min_periods).corr(df)

def ts_regression(x,y,n): # x is a DataFrame with 1 column
    n = _coerce_window(n, "n")
    nd,ns=y.shape
    xv=x.values
    yv=y.values

    slopes=np.full((nd,ns),np.nan)
    intcepts=np.full((nd,ns),np.nan)
    eps=np.full((nd,ns),np.nan)
    for i in range(n-1,nd):
        xs=xv[i-n+1:i+1,:]
        ys=yv[i-n+1:i+1,:]
        valid_mask=np.isfinite(xs)&np.isfinite(ys)
        valid_counts=np.sum(valid_mask,axis=0)
        calc_mask=valid_counts>=2
        if not np.any(calc_mask):
            continue

        xs_valid=np.where(valid_mask,xs,np.nan)
        ys_valid=np.where(valid_mask,ys,np.nan)

        xm=np.full(ns,np.nan)
        ym=np.full(ns,np.nan)
        xm[calc_mask]=np.nanmean(xs_valid[:,calc_mask],axis=0)
        ym[calc_mask]=np.nanmean(ys_valid[:,calc_mask],axis=0)

        x_bar=xs_valid[:,calc_mask]-xm[calc_mask]
        y_bar=ys_valid[:,calc_mask]-ym[calc_mask]
        b_num=np.nanmean(x_bar*y_bar,axis=0)
        b_den=np.nanmean(x_bar*x_bar,axis=0)

        b=np.full(ns,np.nan)
        finite_mask=np.isfinite(b_num)&np.isfinite(b_den)
        nonzero_mask=finite_mask&(b_den!=0)
        if np.any(nonzero_mask):
            b_subset=np.full(np.sum(calc_mask),np.nan)
            b_subset[nonzero_mask]=b_num[nonzero_mask]/b_den[nonzero_mask]
            b[calc_mask]=b_subset

        a=np.full(ns,np.nan)
        coeff_mask=calc_mask&np.isfinite(b)&np.isfinite(xm)&np.isfinite(ym)
        if np.any(coeff_mask):
            a[coeff_mask]=ym[coeff_mask]-b[coeff_mask]*xm[coeff_mask]

        ep=np.full(ns,np.nan)
        stable_mask=np.isfinite(a[calc_mask])&np.isfinite(b[calc_mask])
        if np.any(stable_mask):
            residual=a[calc_mask][stable_mask]+b[calc_mask][stable_mask]*xs_valid[:,calc_mask][:,stable_mask]-ys_valid[:,calc_mask][:,stable_mask]
            ep_subset=np.full(np.sum(calc_mask),np.nan)
            ep_subset[stable_mask]=np.nanstd(residual,axis=0)
            ep[calc_mask]=ep_subset

        slopes[i,:]=b
        intcepts[i,:]=a
        eps[i,:]=ep
    
    slopes=pd.DataFrame(slopes,index=y.index,columns=y.columns)
    intcepts=pd.DataFrame(intcepts,index=y.index,columns=y.columns)
    eps=pd.DataFrame(eps,index=y.index,columns=y.columns)
    return (slopes,intcepts,eps)

def ts_regression2(x,y,n,min_periods=None,rettype=0):
    """
    计算两个DataFrame之间逐列的、滚动的一元线性回归
    rettype参数：
    0表示返回残差
    1表示返回斜率
    2表示返回截距
    """
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)

    nd,ns = y.shape
    xv = x.values
    yv = y.values
    index = y.index
    columns = y.columns
    residuals = np.full((nd,ns),np.nan)
    slopes = np.full((nd,ns),np.nan)
    intercepts = np.full((nd,ns),np.nan)

    for i in range(min_periods-1,nd):
        start_index = max(0,i-n+1)
        xs = xv[start_index:i+1,:]
        ys = yv[start_index:i+1,:]
        valid_mask = (~np.isnan(xs))&(~np.isnan(ys))
        valid_counts = np.sum(valid_mask,axis=0)
        calc_mask = (valid_counts>=min_periods)
        if not np.any(calc_mask):
            continue
        xs_valid = np.where(valid_mask,xs,np.nan)
        ys_valid = np.where(valid_mask,ys,np.nan)
        xm = np.nanmean(xs_valid,axis=0)
        ym = np.nanmean(ys_valid,axis=0)
        x_bar = xs_valid-xm
        y_bar = ys_valid-ym
        b_num = np.nanmean(x_bar*y_bar,axis=0)
        b_den = np.nanmean(x_bar*x_bar,axis=0)
        b_den[b_den==0] = np.nan
        b = b_num/b_den
        a = ym-b*xm
        residual = ys_valid[-1,:]-(a+b*xs_valid[-1,:])
        residuals[i,calc_mask] = residual[calc_mask]
        slopes[i,calc_mask] = b[calc_mask]
        intercepts[i,calc_mask] = a[calc_mask]
        
    if rettype==0:
        return pd.DataFrame(residuals,index=index,columns=columns)
    elif rettype==1:
        return pd.DataFrame(slopes,index=index,columns=columns)
    elif rettype==2:
        return pd.DataFrame(intercepts,index=index,columns=columns)
    
def ts_beta(x,y,n,min_periods=None): # x is a DataFrame with 1 column
    n = _coerce_window(n, "n")
    min_periods = _coerce_min_periods(min_periods, n)
    xx=x*x
    xy=pd.DataFrame(x.values*y.values,index=y.index,columns=y.columns)
    x_bar=x.rolling(n,min_periods=min_periods).mean()
    y_bar=y.rolling(n,min_periods=min_periods).mean()
    xx_bar=xx.rolling(n,min_periods=min_periods).mean()
    xy_bar=xy.rolling(n,min_periods=min_periods).mean()
    xy_bar2=pd.DataFrame(x_bar.values*y_bar.values,index=y.index,columns=y.columns)
    z=xx_bar-x_bar*x_bar
    z=z.where(z>0,np.nan)
    b=(xy_bar-xy_bar2).values/z.values
    return pd.DataFrame(b,index=y.index,columns=y.columns)


if __name__ == '__main__':
    import time

    nd,ns=2000,5000
    df1=pd.DataFrame(np.random.randn(nd,ns))
    df2=pd.DataFrame(np.random.randn(nd,ns))
    df3=pd.DataFrame(np.random.randn(nd,ns))
    con1=pd.DataFrame(np.random.choice([True,False],size=(nd,ns)))
    con2=pd.DataFrame(np.random.choice([True,False],size=(nd,ns)))

    n=120
    m=10

    t1=time.perf_counter()
    z=neg(df1)
    t2=time.perf_counter()
    print('neg time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=abs(df1)
    t2=time.perf_counter()
    print('abs time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=signal(df1)
    t2=time.perf_counter()
    print('signal time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=power(df1)
    t2=time.perf_counter()
    print('power time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=sqrt(df1)
    t2=time.perf_counter()
    print('sqrt time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=log(df1)
    t2=time.perf_counter()
    print('log time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=diff(df1)
    t2=time.perf_counter()
    print('diff time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=diff_q(df1)
    t2=time.perf_counter()
    print('diff_q time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=tanh(df1)
    t2=time.perf_counter()
    print('tanh time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=pos(df1)
    t2=time.perf_counter()
    print('pos_op time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=neg_part(df1)
    t2=time.perf_counter()
    print('neg_part time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=add(df1,df2)
    t2=time.perf_counter()
    print('add time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=sub(df1,df2)
    t2=time.perf_counter()
    print('sub time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=mul(df1,df2)
    t2=time.perf_counter()
    print('mul time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=div(df1,df2)
    t2=time.perf_counter()
    print('div time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=safe_div(df1,df2)
    t2=time.perf_counter()
    print('safe_div time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_delay(df1,n)
    t2=time.perf_counter()
    print('ts_delay time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_delta(df1,n)
    t2=time.perf_counter()
    print('ts_delta time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_pct(df1,n)
    t2=time.perf_counter()
    print('ts_pct time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_rank(df1,n)
    t2=time.perf_counter()
    print('ts_rank time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_decay_linear(df1,n)
    t2=time.perf_counter()
    print('ts_decay_linear time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_sum(df1,n)
    t2=time.perf_counter()
    print('ts_sum time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_mean(df1,n)
    t2=time.perf_counter()
    print('ts_mean time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_var(df1,n)
    t2=time.perf_counter()
    print('ts_var time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_std(df1,n)
    t2=time.perf_counter()
    print('ts_std time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_ir(df1,n)
    t2=time.perf_counter()
    print('ts_ir time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_skew(df1,n)
    t2=time.perf_counter()
    print('ts_skew time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_kur(df1,n)
    t2=time.perf_counter()
    print('ts_kur time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_min(df1,n)
    t2=time.perf_counter()
    print('ts_min time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_max(df1,n)
    t2=time.perf_counter()
    print('ts_max time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_range(df1,n)
    t2=time.perf_counter()
    print('ts_range time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_argmin(df1,n)
    t2=time.perf_counter()
    print('ts_argmin time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_argmax(df1,n)
    t2=time.perf_counter()
    print('ts_argmax time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_median(df1,n)
    t2=time.perf_counter()
    print('ts_median time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_zscore(df1,n)
    t2=time.perf_counter()
    print('ts_zscore time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ema1(df1,n)
    t2=time.perf_counter()
    print('ema1 time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ema2(df1,n)
    t2=time.perf_counter()
    print('ema2 time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ema3(df1,alpha=0.5)
    t2=time.perf_counter()
    print('ema3 time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=wma(df1,w=np.arange(1,n+1))
    t2=time.perf_counter()
    print('wma time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_corr(df1,df2,120)
    t2=time.perf_counter()
    print('ts_corr time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_reg(df1,df2,120)
    t2=time.perf_counter()
    print('ts_reg time(res):',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_reg(df1,df2,120,rettype=3)
    t2=time.perf_counter()
    print('ts_reg time(r2):',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_multireg([df1,df2],df3,120)
    t2=time.perf_counter()
    print('ts_multireg time(res):',t2-t1,"sec")

    t1=time.perf_counter()
    z=ts_multireg([df1,df2],df3,120,rettype=3)
    t2=time.perf_counter()
    print('ts_multireg time(r2):',t2-t1,"sec")

    t1=time.perf_counter()
    gp=cs_rank(df1)
    t2=time.perf_counter()
    print('cs_rank time:',t2-t1,"sec")

    t1=time.perf_counter()
    gp=cs_neutralize(df1)
    t2=time.perf_counter()
    print('cs_neutralize time:',t2-t1,"sec")

    t1=time.perf_counter()
    gp=cs_zscore(df1)
    t2=time.perf_counter()
    print('cs_zscore time:',t2-t1,"sec")

    t1=time.perf_counter()
    gp=cs_scale(df1)
    t2=time.perf_counter()
    print('cs_scale time:',t2-t1,"sec")

    t1=time.perf_counter()
    gp=cs_sum(df1)
    t2=time.perf_counter()
    print('cs_sum time:',t2-t1,"sec")

    t1=time.perf_counter()
    gp=cs_mean(df1)
    t2=time.perf_counter()
    print('cs_mean time:',t2-t1,"sec")

    t1=time.perf_counter()
    gp=cs_std(df1)
    t2=time.perf_counter()
    print('cs_std time:',t2-t1,"sec")

    t1=time.perf_counter()
    gp=cs_median(df1)
    t2=time.perf_counter()
    print('cs_median time:',t2-t1,"sec")

    t1=time.perf_counter()
    gp=cs_corr(df1,df2)
    t2=time.perf_counter()
    print('cs_corr time:',t2-t1,"sec")

    t1=time.perf_counter()
    gp=cs_multireg([df1,df2],df3)
    t2=time.perf_counter()
    print('cs_multireg time:',t2-t1,"sec")

    t1=time.perf_counter()
    gp=fd(df1,df2,n,[-1,1,-1,1])
    t2=time.perf_counter()
    print('fd time:',t2-t1,"sec")

    t1=time.perf_counter()
    gp=bucket(df2,m)
    t2=time.perf_counter()
    print('bucket time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=group_rank(df1,gp)
    t2=time.perf_counter()
    print('group_rank time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=group_neutralize(df1,gp)
    t2=time.perf_counter()
    print('group_neutralize time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=group_zscore(df1,gp)
    t2=time.perf_counter()
    print('group_zscore time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=group_scale(df1,gp)
    t2=time.perf_counter()
    print('group_scale time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=trade_when(df1,con1,con2)
    t2=time.perf_counter()
    print('trade_when time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=winsorize1(df1)
    t2=time.perf_counter()
    print('winsorize1 time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=winsorize2(df1)
    t2=time.perf_counter()
    print('winsorize2 time:',t2-t1,"sec")

    t1=time.perf_counter()
    z=gaussian_rank(df1)
    t2=time.perf_counter()
    print('gaussian_rank time:',t2-t1,"sec")



