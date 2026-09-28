import time

import numpy as np
from .numba_kernels import *
from typing import Union

"""
数据契约:
浮点型数据: np.float64类型, np.nan表示缺失
分组型数据: np.int32类型, 仅取值为-1,0,1,2,...的连续整数, -1表缺失, 0,1,2,...表分组组别
条件型数据: np.int8类型, 仅取值为-1,0,1, -1表缺失, 0表False, 1表True
"""

"""
check
"""
def math_self(x:np.ndarray)->np.ndarray:
    return x

def ts_self(x:np.ndarray)->np.ndarray:
    return x

def group_self(x:np.ndarray)->np.ndarray:
    return x

def is_float(x)->bool:
    return (
        isinstance(x,np.ndarray)
        and x.ndim==2
        and x.dtype==np.float64
    )

def is_condition(x)->bool:
    if not (
        isinstance(x,np.ndarray)
        and x.ndim==2
        and x.dtype==np.int8
    ):
        return False
    vals = np.unique(x)
    if vals.size>3:
        return False
    return bool(np.isin(vals,[-1,0,1]).all())

def is_group(x)->bool:
    if not (
        isinstance(x,np.ndarray)
        and x.ndim==2
        and x.dtype==np.int32
    ):
        return False
    vals = np.unique(x)
    if vals.size==0:
        return False
    if vals[0]<-1:
        return False
    valid = vals[vals>=0]
    if valid.size==0:
        return True
    expected = np.arange(valid[-1]+1,dtype=np.int32)
    return bool(np.array_equal(valid,expected))

"""
construct condition
"""
def eq(x:np.ndarray,y:Union[np.ndarray,float])->np.ndarray:
    if isinstance(y,np.ndarray):
        return eq_array_kernel(x,y)
    return eq_scalar_kernel(x,float(y))

def le(x:np.ndarray,y:Union[np.ndarray,float])->np.ndarray:
    if isinstance(y,np.ndarray):
        return le_array_kernel(x,y)
    return le_scalar_kernel(x,float(y))

def ge(x:np.ndarray,y:Union[np.ndarray,float])->np.ndarray:
    if isinstance(y,np.ndarray):
        return ge_array_kernel(x,y)
    return ge_scalar_kernel(x,float(y))

def con_and(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return con_and_kernel(x,y)

def con_or(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return con_or_kernel(x,y)

def con_not(x:np.ndarray)->np.ndarray:
    return con_not_kernel(x)

"""
construct group
"""
def bucket(x:np.ndarray,n:int)->np.ndarray:
    return bucket_kernel(x,n)

def group_merge(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return group_merge_kernel(x,y)

"""
math
"""
def neg(x:np.ndarray)->np.ndarray:
    return -x

def abs(x:np.ndarray)->np.ndarray:
    return np.abs(x)

def signal(x:np.ndarray)->np.ndarray:
    return np.sign(x)

def power(x:np.ndarray,p:float=2)->np.ndarray:
    return x**p

def signed_power(x:np.ndarray,p:float=2)->np.ndarray:
    return signal(x)*power(abs(x),p)

def sqrt(x:np.ndarray)->np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.sqrt(x)
    
def log(x:np.ndarray)->np.ndarray:
    with np.errstate(divide="ignore",invalid="ignore"):
        out = np.log(x)
    out[~np.isfinite(out)] = np.nan
    return out

def diff(x:np.ndarray)->np.ndarray:
    out = np.full(x.shape,np.nan,dtype=np.float64)
    out[1:,:] = x[1:,:]-x[:-1,:]
    return out

def diff_q(x:np.ndarray)->np.ndarray:
    out = np.full(x.shape,np.nan,dtype=np.float64)
    with np.errstate(divide="ignore",invalid="ignore"):
        out[1:,:] = x[1:,:]/x[:-1,:]-1.0
    out[~np.isfinite(out)] = np.nan
    return out

def tanh(x:np.ndarray)->np.ndarray:
    return np.tanh(x)

def pos_part(x:np.ndarray)->np.ndarray:
    return np.maximum(x,0.0)

def neg_part(x:np.ndarray)->np.ndarray:
    return np.minimum(x,0.0)

def midmap(x:np.ndarray)->np.ndarray:
    r = cs_rank(x)
    return 1.0-2.0*abs(r-0.5)

def xlogx(x:np.ndarray)->np.ndarray:
    out = np.zeros(x.shape,dtype=np.float64)
    m = x>0
    out[m] = x[m]*np.log(x[m])
    out[~np.isfinite(x)] = np.nan
    return out

def leftmap(x:np.ndarray)->np.ndarray:
    r = cs_rank(x)
    return -xlogx(r)

def rightmap(x:np.ndarray)->np.ndarray:
    r = cs_rank(x)
    return -xlogx(1.0-r)

def extreme_leftmap(x:np.ndarray)->np.ndarray:
    r = cs_rank(x)
    return np.exp(-((r-0.15)**2))

def extreme_rightmap(x:np.ndarray)->np.ndarray:
    r = cs_rank(x)
    return np.exp(-((r-0.85)**2))

"""
cross section
"""
def cs_rank(x:np.ndarray,pct:bool=True,ascending:bool=True)->np.ndarray:
    return cs_rank_kernel(x,bool(pct),bool(ascending))

def cs_neutralize(x:np.ndarray)->np.ndarray:
    return cs_neutralize_kernel(x)

def cs_zscore(x:np.ndarray)->np.ndarray:
    return cs_zscore_kernel(x)

def cs_weighted_zscore(x:np.ndarray,w:np.ndarray)->np.ndarray:
    return cs_weighted_zscore_kernel(x,w)

def cs_scale(x:np.ndarray)->np.ndarray:
    return cs_scale_kernel(x)

"""
group
"""
def group_rank(x:np.ndarray,g:np.ndarray)->np.ndarray:
    return group_rank_kernel(x,g)

def group_neutralize(x:np.ndarray,g:np.ndarray)->np.ndarray:
    return group_neutralize_kernel(x,g)

def group_zscore(x:np.ndarray,g:np.ndarray)->np.ndarray:
    return group_zscore_kernel(x,g)

def group_scale(x:np.ndarray,g:np.ndarray)->np.ndarray:
    return group_scale_kernel(x,g)

def group_midmap(x:np.ndarray,g:np.ndarray)->np.ndarray:
    gr = group_rank(x,g)
    return 1-2*abs(gr-0.5)

def group_leftmap(x:np.ndarray,g:np.ndarray)->np.ndarray:
    gr = group_rank(x,g)
    return -xlogx(gr)

def group_rightmap(x:np.ndarray,g:np.ndarray)->np.ndarray:
    gr = group_rank(x,g)
    return -xlogx(1.0-gr)

"""
ts
"""
def ts_delay(x:np.ndarray,n:int)->np.ndarray:
    out = np.full(x.shape,np.nan,dtype=np.float64)
    if n>0:
        out[n:,:] = x[:-n,:]
    elif n<0:
        out[:n,:] = x[-n:,:]
    else:
        out[:,:] = x
    return out

def ts_delta(x:np.ndarray,n:int)->np.ndarray:
    return x-ts_delay(x,n)

def ts_pct(x:np.ndarray,n:int)->np.ndarray:
    with np.errstate(divide="ignore",invalid="ignore"):
        out = x/ts_delay(x,n)-1.0
    out[~np.isfinite(out)] = np.nan
    return out

def ts_rank(x:np.ndarray,n:int,min_periods=None,pct:bool=True)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_rank_kernel(x,n,min_periods,pct)

def ts_decay_linear(x:np.ndarray,n:int)->np.ndarray:
    return ts_decay_linear_kernel(x,n)

def ts_sum(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_sum_kernel(x,n,min_periods)

def ts_prod(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_prod_kernel(x,n,min_periods)

def ts_mean(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_mean_kernel(x,n,min_periods)

def ts_var(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_var_kernel(x,n,min_periods)

def ts_std(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.sqrt(ts_var(x,n,min_periods))

def ts_ir(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_ir_kernel(x,n,min_periods)

def ts_skew(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_skew_kernel(x,n,min_periods)

def ts_kur(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_kur_kernel(x,n,min_periods)

def ts_min(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_min_kernel(x,n,min_periods)

def ts_max(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_max_kernel(x,n,min_periods)

def ts_range(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    return ts_max(x,n,min_periods)-ts_min(x,n,min_periods)

def ts_argmin(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_argmin_kernel(x,n,min_periods)

def ts_argmax(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_argmax_kernel(x,n,min_periods)

def ts_median(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_median_kernel(x,n,min_periods)

def ts_zscore(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_zscore_kernel(x,n,min_periods)

def ema1(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ema_kernel(x,2.0/(1.0+n),min_periods)

def ema2(x:np.ndarray,n:int,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ema_kernel(x,1.0/(1.0+n),min_periods)

def ema3(x:np.ndarray,alpha:float,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = 2
    return ema_kernel(x,alpha,min_periods)

def wma(x:np.ndarray,w:np.ndarray)->np.ndarray:
    return wma_kernel(x,w)

"""
condition
"""
def ts_cut(x:np.ndarray,y:np.ndarray,n:int,agg:str="diff",hq:float=0.5,min_periods=None)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    h,l = ts_cut_kernel(x,y,n,hq,min_periods)
    if agg=="high":
        return h
    if agg=="low":
        return l
    if agg=="diff":
        return h-l
    if agg=="normdiff":
        return div(h-l,abs(h)+abs(l))
    if agg=="div":
        return div(h,l)
    raise ValueError("invalid agg")

def ts_cut_high(x:np.ndarray,y:np.ndarray,n:int,hq:float=0.5,min_periods=None)->np.ndarray:
    return ts_cut(x,y,n,"high",hq,min_periods)

def ts_cut_low(x:np.ndarray,y:np.ndarray,n:int,hq:float=0.5,min_periods=None)->np.ndarray:
    return ts_cut(x,y,n,"low",hq,min_periods)

def ts_cut_diff(x:np.ndarray,y:np.ndarray,n:int,hq:float=0.5,min_periods=None)->np.ndarray:
    return ts_cut(x,y,n,"diff",hq,min_periods)

def ts_cut_normdiff(x:np.ndarray,y:np.ndarray,n:int,hq:float=0.5,min_periods=None)->np.ndarray:
    return ts_cut(x,y,n,"normdiff",hq,min_periods)

def ts_cut_div(x:np.ndarray,y:np.ndarray,n:int,hq:float=0.5,min_periods=None)->np.ndarray:
    return ts_cut(x,y,n,"div",hq,min_periods)

def adjust_by(x:np.ndarray,con:np.ndarray,d:float=0.5)->np.ndarray:
    return adjust_by_kernel(x,con,d)

def reverse_by(x:np.ndarray,con:np.ndarray)->np.ndarray:
    return reverse_by_kernel(x,con)

def reverse_rank_by(x:np.ndarray,con:np.ndarray)->np.ndarray:
    return reverse_rank_by_kernel(x,con)

def trade_when(x:np.ndarray,con:np.ndarray,q:float=0.2)->np.ndarray:
    return trade_when_kernel(x,con,q)

# cross

def add(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return x+y

def addr(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return addr_kernel(x,y)

def sub(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return x-y

def subr(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return subr_kernel(x,y)

def mul(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return x*y

def mulr(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return mulr_kernel(x,y)

def mulz(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return mulz_kernel(x,y)

def div(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return div_kernel(x,y)

def safe_div(x:np.ndarray,y:np.ndarray,eps:float=5e-2)->np.ndarray:
    z = np.where(np.abs(y)<eps,np.where(y>=0,eps,-eps),y)
    out = x/z
    out[~np.isfinite(out)] = np.nan
    return out

def maximum(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return maximum_kernel(x,y)

def maximumr(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return maximumr_kernel(x,y)

def minimum(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return minimum_kernel(x,y)

def minimumr(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return minimumr_kernel(x,y)

def mix(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return mulz(x,y)

def fd(x:np.ndarray,y:np.ndarray,n:int,w1:int,w2:int,w3:int,w4:int,method="decay")->np.ndarray:
    z1 = cs_zscore(x)
    z2 = cs_zscore(y)
    z = fd_sum_kernel(z1,z2,w1,w2,w3,w4)
    if method=="decay":
        return ts_decay_linear(z,n)
    if method=="mean":
        return ts_mean(z,n)
    raise ValueError("invalid method")

def ts_corr(x:np.ndarray,y:np.ndarray,n:int,min_periods=None,hold_nan:bool=True)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_corr_kernel(x,y,n,min_periods,hold_nan)

def ts_reg(x:np.ndarray,y:np.ndarray,n:int,rettype:int=0,min_periods=None,)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    return ts_reg_kernel(x,y,n,rettype,min_periods)

def ts_multireg(xs:list[np.ndarray],y:np.ndarray,n:int,rettype:int=0,min_periods=None,)->np.ndarray:
    if min_periods is None:
        min_periods = n//2
    nd,ns = y.shape
    k = len(xs)
    x = np.empty((nd,ns,k),dtype=np.float64)
    for i in range(k):
        x[:,:,i] = xs[i]
    return ts_multireg_kernel(x,y,n,rettype,min_periods)

def cs_corr(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return cs_corr_kernel(x,y)

def cs_reg(x:np.ndarray,y:np.ndarray,rettype:int=0,min_cs:int=30,)->np.ndarray:
    return cs_reg_kernel(x,y,rettype,min_cs)

def cs_multireg(xs:list[np.ndarray],y:np.ndarray,min_cs:int=30)->np.ndarray:
    nd,ns = y.shape
    k = len(xs)
    x = np.empty((nd,ns,k),dtype=np.float64)
    for i in range(k):
        x[:,:,i] = xs[i]
    return cs_multireg_kernel(x,y,min_cs)

def if_else(a:np.ndarray,b:np.ndarray,con:np.ndarray)->np.ndarray:
    return if_else_kernel(a,b,con)

"""
other funcs
"""
def mean(x:np.ndarray)->float:
    return mean_kernel(x)

def std(x:np.ndarray)->float:
    return std_kernel(x)

def ir(x:np.ndarray)->float:
    return ir_kernel(x)

def mask(x:np.ndarray,y:np.ndarray)->np.ndarray:
    return mask_kernel(x,y)

def p2p_corr(x:np.ndarray,y:np.ndarray)->float:
    return p2p_corr_kernel(x,y)



if __name__=="__main__":
    rng = np.random.default_rng(42)
    nd = 2000
    ns = 5000
    nan_rate = 0.05
    group_n = 30
    n = 20

    x = rng.standard_normal((nd,ns)).astype(np.float64)
    y = rng.standard_normal((nd,ns)).astype(np.float64)
    z = rng.standard_normal((nd,ns)).astype(np.float64)
    w = np.abs(rng.standard_normal((nd,ns))).astype(np.float64)+0.1
    xpos = np.abs(x)+0.1
    ypos = np.abs(y)+0.1

    m = rng.random((nd,ns))<nan_rate
    x[m] = np.nan
    m = rng.random((nd,ns))<nan_rate
    y[m] = np.nan
    m = rng.random((nd,ns))<nan_rate
    z[m] = np.nan
    m = rng.random((nd,ns))<nan_rate
    w[m] = np.nan
    m = rng.random((nd,ns))<nan_rate
    xpos[m] = np.nan
    m = rng.random((nd,ns))<nan_rate
    ypos[m] = np.nan

    g = rng.integers(0,group_n,size=(nd,ns),dtype=np.int32)
    m = rng.random((nd,ns))<nan_rate
    g[m] = -1

    con = rng.choice(np.array([-1,0,1],dtype=np.int8),size=(nd,ns),p=[0.05,0.475,0.475])
    con2 = rng.choice(np.array([-1,0,1],dtype=np.int8),size=(nd,ns),p=[0.05,0.475,0.475])

    wma_w = np.array([0.1,0.2,0.3,0.4],dtype=np.float64)
    fd_w = [0.25,0.25,0.25,0.25]

    ops = [
        ("is_float",lambda:is_float(x)),
        ("is_condition",lambda:is_condition(con)),
        ("is_group",lambda:is_group(g)),

        ("eq_array",lambda:eq(x,y)),
        ("eq_scalar",lambda:eq(x,0.0)),
        ("le_array",lambda:le(x,y)),
        ("le_scalar",lambda:le(x,0.0)),
        ("ge_array",lambda:ge(x,y)),
        ("ge_scalar",lambda:ge(x,0.0)),
        ("con_and",lambda:con_and(con,con2)),
        ("con_or",lambda:con_or(con,con2)),
        ("con_not",lambda:con_not(con)),

        ("bucket",lambda:bucket(x,10)),

        ("neg",lambda:neg(x)),
        ("abs",lambda:abs(x)),
        ("signal",lambda:signal(x)),
        ("power",lambda:power(x,2.0)),
        ("signed_power",lambda:signed_power(x,2.0)),
        ("sqrt",lambda:sqrt(xpos)),
        ("log",lambda:log(xpos)),
        ("diff",lambda:diff(x)),
        ("diff_q",lambda:diff_q(xpos)),
        ("tanh",lambda:tanh(x)),
        ("pos_part",lambda:pos_part(x)),
        ("neg_part",lambda:neg_part(x)),
        ("midmap",lambda:midmap(x)),
        ("xlogx",lambda:xlogx(xpos)),
        ("leftmap",lambda:leftmap(x)),
        ("rightmap",lambda:rightmap(x)),
        ("extreme_leftmap",lambda:extreme_leftmap(x)),
        ("extreme_rightmap",lambda:extreme_rightmap(x)),

        ("cs_rank",lambda:cs_rank(x,True,True)),
        ("cs_neutralize",lambda:cs_neutralize(x)),
        ("cs_zscore",lambda:cs_zscore(x)),
        ("cs_weighted_zscore",lambda:cs_weighted_zscore(x,w)),
        ("cs_scale",lambda:cs_scale(x)),

        ("group_rank",lambda:group_rank(x,g)),
        ("group_neutralize",lambda:group_neutralize(x,g)),
        ("group_zscore",lambda:group_zscore(x,g)),
        ("group_scale",lambda:group_scale(x,g)),
        ("group_midmap",lambda:group_midmap(x,g)),
        ("group_leftmap",lambda:group_leftmap(x,g)),
        ("group_rightmap",lambda:group_rightmap(x,g)),

        ("ts_delay_pos",lambda:ts_delay(x,5)),
        ("ts_delay_neg",lambda:ts_delay(x,-5)),
        ("ts_delta",lambda:ts_delta(x,5)),
        ("ts_pct",lambda:ts_pct(xpos,5)),
        ("ts_rank",lambda:ts_rank(x,n)),
        ("ts_decay_linear",lambda:ts_decay_linear(x,n)),
        ("ts_sum",lambda:ts_sum(x,n)),
        ("ts_prod",lambda:ts_prod(xpos,n)),
        ("ts_mean",lambda:ts_mean(x,n)),
        ("ts_var",lambda:ts_var(x,n)),
        ("ts_std",lambda:ts_std(x,n)),
        ("ts_ir",lambda:ts_ir(x,n)),
        ("ts_skew",lambda:ts_skew(x,n)),
        ("ts_kur",lambda:ts_kur(x,n)),
        ("ts_min",lambda:ts_min(x,n)),
        ("ts_max",lambda:ts_max(x,n)),
        ("ts_range",lambda:ts_range(x,n)),
        ("ts_argmin",lambda:ts_argmin(x,n)),
        ("ts_argmax",lambda:ts_argmax(x,n)),
        ("ts_median",lambda:ts_median(x,n)),
        ("ts_zscore",lambda:ts_zscore(x,n)),
        ("ema1",lambda:ema1(x,n)),
        ("ema2",lambda:ema2(x,n)),
        ("ema3",lambda:ema3(x,0.1)),
        ("wma",lambda:wma(x,wma_w)),

        ("add",lambda:add(x,y)),
        ("sub",lambda:sub(x,y)),
        ("mul",lambda:mul(x,y)),
        ("div",lambda:div(x,y)),
        ("safe_div",lambda:safe_div(x,y)),
        ("mix",lambda:mix(x,y)),
        ("fd_decay",lambda:fd(x,y,n,fd_w,"decay")),
        ("fd_mean",lambda:fd(x,y,n,fd_w,"mean")),
        ("ts_corr",lambda:ts_corr(x,y,n)),
        ("ts_reg_0",lambda:ts_reg(x,y,n,rettype=0)),
        ("ts_reg_1",lambda:ts_reg(x,y,n,rettype=1)),
        ("ts_multireg",lambda:ts_multireg([x,y],z,n)),
        ("cs_corr",lambda:cs_corr(x,y)),
        ("cs_multireg",lambda:cs_multireg([x,y],z)),

        ("ts_cut_high",lambda:ts_cut_high(x,y,n)),
        ("ts_cut_low",lambda:ts_cut_low(x,y,n)),
        ("ts_cut_diff",lambda:ts_cut_diff(x,y,n)),
        ("ts_cut_normdiff",lambda:ts_cut_normdiff(x,y,n)),
        ("ts_cut_div",lambda:ts_cut_div(x,y,n)),
        ("adjust_by",lambda:adjust_by(x,con,0.5)),
        ("reverse_by",lambda:reverse_by(x,con)),
        ("reverse_rank_by",lambda:reverse_rank_by(x,con)),
        ("if_else",lambda:if_else(con,x,y)),
        ("trade_when",lambda:trade_when(x,con,0.2)),

        ("mean",lambda:mean(x[:,0])),
        ("std",lambda:std(x[:,0])),
        ("ir",lambda:ir(x[:,0])),
        ("mask",lambda:mask(x,con)),
    ]

    for name,fn in ops:
        t1 = time.perf_counter()
        fn()
        t2 = time.perf_counter()
        print(name,"time:",round(t2-t1,4))
