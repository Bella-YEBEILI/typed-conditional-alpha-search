import numpy as np
import pandas as pd
from scipy.stats import norm
from quant.quant_lib.numbafunc import *

# math 
def neg(target:pd.DataFrame|pd.Series):
    return -target

def abs(target:pd.DataFrame|pd.Series):
    return target.abs()

def signal(target:pd.DataFrame|pd.Series):
    return np.sign(target)

def power(target:pd.DataFrame|pd.Series,p=2):
    return target**p

def signed_power(target:pd.DataFrame|pd.Series,p=2):
    return signal(target)*power(abs(target),p)

def sqrt(target:pd.DataFrame|pd.Series):
    with np.errstate(invalid='ignore'):
        return np.sqrt(target)

def log(target:pd.DataFrame|pd.Series):
    with np.errstate(divide='ignore',invalid='ignore'):
        log_df = np.log(target)
        return log_df.replace([np.inf,-np.inf], np.nan)

def diff(target:pd.DataFrame|pd.Series):
    return target-target.shift(1)

def diff_q(target:pd.DataFrame|pd.Series):
    return (target/target.shift(1)-1).replace([np.inf,-np.inf],np.nan)

def tanh(target:pd.DataFrame|pd.Series):
    return np.tanh(target)

def pos_part(target:pd.DataFrame|pd.Series): 
    return np.maximum(target, 0.0)

def neg_part(target:pd.DataFrame|pd.Series):
    return np.minimum(target,0.0)

def midmap(df):
    return 1-2*abs(cs_rank(df)-0.5)

def leftmap(df):
    r = cs_rank(df)
    return -r*log(r)

def rightmap(df):
    r = cs_rank(df)
    return -(1-r)*log(1-r) 

def extreme_leftmap(df):
    r = cs_rank(df)
    return np.exp(-(r-0.15)**2)

def extreme_rightmap(df):
    r = cs_rank(df)
    return np.exp(-(r-0.85)**2)

# time-series 
def ts_delay(target:pd.DataFrame|pd.Series,n):
    return target.shift(n)

def ts_delta(target:pd.DataFrame|pd.Series,n):
    return target-target.shift(n)

def ts_pct(target:pd.DataFrame|pd.Series,n):
    return (target/ts_delay(target,n)-1).replace([np.inf,-np.inf],np.nan)

def ts_rank(target:pd.DataFrame|pd.Series,n,min_periods=None,pct=True):
    if min_periods is None:
        min_periods=int(n/2)
    if isinstance(target,pd.Series):
        return target.rolling(window=n,min_periods=min_periods).rank(pct=pct)
    values=target.to_numpy(dtype=float)
    out=ts_rank_2d(values,n,min_periods,bool(pct))
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ts_decay_linear(target:pd.DataFrame|pd.Series,n):
    if isinstance(target,pd.Series):
        def _decay(x):
            if not np.isfinite(x[-1]):
                return np.nan
            weights = np.arange(n-x.shape[0]+1,n+1,dtype=float)
            mask = np.isfinite(x)
            return np.average(x[mask],weights=weights[mask])
        return target.rolling(window=n,min_periods=1).apply(_decay,raw=True)
    values = target.to_numpy(dtype=float)
    out = ts_decay_linear_2d(values,n)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ts_sum(target:pd.DataFrame|pd.Series,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    if isinstance(target,pd.Series):
        return target.rolling(window=n,min_periods=min_periods).sum()
    values = target.to_numpy(dtype=float)
    out = ts_sum_2d(values,n,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ts_prod(target:pd.DataFrame|pd.Series,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    if isinstance(target,pd.Series):
        return target.rolling(window=n,min_periods=min_periods).apply(np.nanprod,raw=True)
    values = target.to_numpy(dtype=float)
    out = ts_prod_2d(values,n,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ts_mean(target:pd.DataFrame|pd.Series,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    if isinstance(target,pd.Series):
        return target.rolling(window=n,min_periods=min_periods).mean()
    values = target.to_numpy(dtype=float)
    out = ts_mean_2d(values,n,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ts_var(target:pd.DataFrame|pd.Series,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    if isinstance(target,pd.Series):
        return target.rolling(window=n,min_periods=min_periods).var(ddof=1)
    values = target.to_numpy(dtype=float)
    out = ts_var_2d(values,n,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ts_std(target:pd.DataFrame|pd.Series,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    if isinstance(target,pd.Series):
        return target.rolling(window=n,min_periods=min_periods).std(ddof=1)
    return sqrt(ts_var(target,n,min_periods))

def ts_ir(target:pd.DataFrame|pd.Series,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    if isinstance(target,pd.Series):
        rolling = target.rolling(window=n,min_periods=min_periods)
        ts_std = rolling.std(ddof=1).where(lambda x: x>1e-10)
        return rolling.mean()/ts_std
    values = target.to_numpy(dtype=float)
    out = ts_ir_2d(values,n,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ts_skew(target:pd.DataFrame|pd.Series,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    if isinstance(target,pd.Series):
        return target.rolling(window=n,min_periods=min_periods).skew()
    values = target.to_numpy(dtype=float)
    out = ts_skew_2d(values,n,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ts_kur(target:pd.DataFrame|pd.Series,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    if isinstance(target,pd.Series):
        return target.rolling(window=n,min_periods=min_periods).kurt()
    values = target.to_numpy(dtype=float)
    out = ts_kur_2d(values,n,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ts_min(target:pd.DataFrame|pd.Series,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    if isinstance(target,pd.Series):
        return target.rolling(window=n,min_periods=min_periods).min()
    values = target.to_numpy(dtype=float)
    out = ts_min_2d(values,n,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ts_max(target:pd.DataFrame|pd.Series,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    if isinstance(target,pd.Series):
        return target.rolling(window=n,min_periods=min_periods).max()
    values = target.to_numpy(dtype=float)
    out = ts_max_2d(values,n,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ts_range(target:pd.DataFrame|pd.Series,n,min_periods=None):
    return ts_max(target,n,min_periods)-ts_min(target,n,min_periods)

def ts_argmin(target:pd.DataFrame|pd.Series,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    if isinstance(target,pd.Series):
        def _argmin(x):
            mask = np.isfinite(x)
            if not mask.any():
                return np.nan
            pos = np.flatnonzero(mask)
            vals = x[mask]
            min_val = vals.min()
            return x.shape[0]-pos[np.where(vals==min_val)[0][-1]]
        return target.rolling(window=n,min_periods=min_periods).apply(_argmin,raw=True)
    values = target.to_numpy(dtype=float)
    out = ts_argmin_2d(values,n,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ts_argmax(target:pd.DataFrame|pd.Series,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    if isinstance(target,pd.Series):
        def _argmax(x):
            mask = np.isfinite(x)
            if not mask.any():
                return np.nan
            pos = np.flatnonzero(mask)
            vals = x[mask]
            max_val = vals.max()
            return x.shape[0]-pos[np.where(vals==max_val)[0][-1]]
        return target.rolling(window=n,min_periods=min_periods).apply(_argmax,raw=True)
    values = target.to_numpy(dtype=float)
    out = ts_argmax_2d(values,n,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ts_median(target:pd.DataFrame|pd.Series,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    if isinstance(target,pd.Series):
        return target.rolling(window=n,min_periods=min_periods).median()
    values = target.to_numpy(dtype=float)
    out = ts_median_2d(values,n,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ts_zscore(target:pd.DataFrame|pd.Series,n,min_periods=None):
    if min_periods is None:
        min_periods=int(n/2)
    if isinstance(target,pd.Series):
        rolling = target.rolling(window=n,min_periods=min_periods)
        ts_std = rolling.std(ddof=1).where(lambda x: x>1e-10)
        return ((target-rolling.mean())/ts_std).replace([np.inf,-np.inf],np.nan)
    values = target.to_numpy(dtype=float)
    out = ts_zscore_2d(values,n,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ema1(target:pd.DataFrame|pd.Series,n,min_periods=None):   # alpha=2/(1+n)
    if min_periods is None:
        min_periods=int(n/2)
    alpha = 2.0/(1+n)
    if isinstance(target,pd.Series):
        return target.ewm(alpha=alpha,adjust=True,ignore_na=False,min_periods=min_periods).mean()
    values = target.to_numpy(dtype=float)
    out = ema_2d(values,alpha,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ema2(target:pd.DataFrame|pd.Series,n,min_periods=None):   # alpha=1/(1+n)
    if min_periods is None:
        min_periods=int(n/2)
    alpha = 1.0/(1+n)
    if isinstance(target,pd.Series):
        return target.ewm(alpha=alpha,adjust=True,ignore_na=False,min_periods=min_periods).mean()
    values = target.to_numpy(dtype=float)
    out = ema_2d(values,alpha,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ema3(target:pd.DataFrame|pd.Series,alpha,min_periods=None):   # alpha is given by a (0<a<1)
    if min_periods is None:
        min_periods=2
    if isinstance(target,pd.Series):
        return target.ewm(alpha=alpha,adjust=True,ignore_na=False,min_periods=min_periods).mean()
    values = target.to_numpy(dtype=float)
    out = ema_2d(values,alpha,min_periods)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def wma(target:pd.DataFrame|pd.Series,w):
    if isinstance(target,pd.Series):
        weights = np.asarray(w,dtype=float)
        def _wma(x):
            if x.shape[0]<weights.shape[0]:
                return np.nan
            if not np.isfinite(x[-1]):
                return np.nan
            mask = np.isfinite(x)
            weight_sum = weights[mask].sum()
            if weight_sum==0.0:
                return np.nan
            return np.average(x[mask],weights=weights[mask])
        return target.rolling(window=weights.shape[0],min_periods=1).apply(_wma,raw=True)
    values = target.to_numpy(dtype=float)
    out = wma_2d(values,w)
    return pd.DataFrame(out,index=target.index,columns=target.columns)

def ts_reg(X,y,n,min_periods=None,rettype=0):
    return ts_multireg([X],y,n,min_periods,rettype)

def ts_multireg(X_list,y_df,n,min_periods=None,rettype=0):
    """
    rettype
    0: 残差
    3: 决定系数
    """
    if min_periods is None:
        min_periods = int(n/2)
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

def cs_multireg(X_list,y_df,min_cs=30):
    y = y_df.to_numpy(dtype=np.float64)
    nd,ns = y.shape
    K = len(X_list)
    X = np.empty((nd,ns,K),dtype=np.float64)
    for k in range(K):
        X[:,:,k] = X_list[k].to_numpy(dtype=np.float64)
    out = cs_multireg_2d(X=X,y=y,min_cs=min_cs)
    return pd.DataFrame(out,index=y_df.index,columns=y_df.columns)

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


# ===== combo ===== 

# 2fac
def add(df1,df2):
    return df1+df2

def sub(df1,df2):
    return df1-df2

def mul(df1,df2):
    return df1*df2

def div(df1,df2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return (df1/df2).replace([np.inf,-np.inf],np.nan)
    
def safe_div(df1,df2,eps=5e-2):
    v1,v2 = df1.to_numpy(dtype=float),df2.to_numpy(dtype=float)
    v2 = np.where(np.abs(v2)<eps,np.where(v2>=0,eps,-eps),v2)
    out = v1/v2
    out = np.where(np.isinf(out),np.nan,out)
    return pd.DataFrame(out,index=df1.index,columns=df1.columns)

def mix(df1,df2):
    return mul(cs_zscore(df1),cs_zscore(df2))

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

def ts_corr(df1,df2,n,min_periods=None,hold_nan=True):
    """
    hold_nan=True: 当日df1或df2为nan值时,返回nan值
    hold_nan=False: 仍利用窗口内的有效值计算相关系数
    """
    if min_periods is None:
        min_periods=int(n/2)
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
    if min_periods is None:
        min_periods=int(n/2)
    xv=x.to_numpy(dtype=float)
    yv=y.to_numpy(dtype=float)
    out=ts_reg_2d(xv,yv,n,min_periods,int(rettype))
    return pd.DataFrame(out,index=y.index,columns=y.columns)

# 1fac+1con
def adjust_by(df,con,d=0.5):
    mask = con.notna()
    con = con.fillna(0).astype(bool)
    df = df.where(mask)
    r = cs_rank(df)
    w = pd.DataFrame(
        np.where(con,1+d,1-d),
        index=df.index,
        columns=df.columns
    )
    return r*w

def reverse_by(df,con):
    mask = con.notna()
    con = con.fillna(0).astype(bool)
    return df.where(~con,-df).where(mask)

def reverse_rank_by(df,con):
    mask = con.notna()
    con = con.fillna(0).astype(bool)
    pos = df.where(con)
    neg = df.where((~con)&mask)
    out = cs_rank(pos,pct=True,ascending=True).fillna(0.0)+cs_rank(neg,pct=True,ascending=False).fillna(0.0)
    return out.where(mask)

# 1fac+2con
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


# 2fac+1con
def if_else(con,a,b):
    return cs_rank(a).where(con,cs_rank(b))

# supplement
def step(df):
    r = cs_rank(df)
    return (r>0.5).astype(float)

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




