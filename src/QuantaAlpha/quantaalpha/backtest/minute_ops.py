import os
os.environ["NUMBA_NUM_THREADS"] = "12"
import numpy as np
from numba import njit,prange
from .minute_tools import MinuteFactorEngine

@MinuteFactorEngine.register("mean")
@njit(cache=True,parallel=True)
def ops_mean(arr):
    nd,ns = arr.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    for c in prange(ns):
        s = 0.0
        cnt = 0
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                s += v
                cnt += 1
        if cnt>0:
            out[c] = s/cnt
    return out

@MinuteFactorEngine.register("std")
@njit(cache=True,parallel=True)
def ops_std(arr):
    nd,ns = arr.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    for c in prange(ns):
        s = 0.0
        sq = 0.0
        cnt = 0
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                s += v
                sq += v*v
                cnt += 1
        if cnt>1:
            mean = s/cnt
            var = (sq/cnt)-(mean*mean)
            if var>0:
                out[c] = np.sqrt(var)
            else:
                out[c] = 0.0
    return out


@MinuteFactorEngine.register("kurt")
@njit(cache=True, parallel=True)
def ops_kurt(arr):
    nd, ns = arr.shape
    out = np.full(ns, np.nan, dtype=np.float32)
    for c in prange(ns):
        s = 0.0
        cnt = 0
        for r in range(nd):
            v = arr[r, c]
            if np.isfinite(v):
                s += v
                cnt += 1
        if cnt < 4:
            continue
        mean = s / cnt
        m2 = 0.0
        m4 = 0.0
        for r in range(nd):
            v = arr[r, c]
            if np.isfinite(v):
                centered = v - mean
                centered_sq = centered * centered
                m2 += centered_sq
                m4 += centered_sq * centered_sq
        if m2 > 1e-12:
            second_moment = m2 / cnt
            fourth_moment = m4 / cnt
            out[c] = fourth_moment / (second_moment * second_moment) - 3.0
        else:
            out[c] = 0.0
    return out

@MinuteFactorEngine.register("sum")
@njit(cache=True,parallel=True)
def ops_sum(arr):
    nd,ns = arr.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    for c in prange(ns):
        s = 0.0
        all_nan = True
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                s += v
                all_nan = False
        if not all_nan:
            out[c] = s
    return out

@MinuteFactorEngine.register("min")
@njit(cache=True,parallel=True)
def ops_min(arr):
    nd,ns = arr.shape
    out=np.full(ns,np.nan,dtype=np.float32)
    for c in prange(ns):
        min_v = np.inf
        found = False
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                if v<min_v:
                    min_v = v
                    found = True
        if found:
            out[c] = min_v
    return out

@MinuteFactorEngine.register("max")
@njit(cache=True,parallel=True)
def ops_max(arr):
    nd,ns = arr.shape
    out=np.full(ns,np.nan,dtype=np.float32)
    for c in prange(ns):
        max_v = -np.inf
        found = False
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                if v>max_v:
                    max_v = v
                    found = True
        if found:
            out[c] = max_v
    return out

@MinuteFactorEngine.register("add")
@njit(cache=True,parallel=True)
def ops_add(arr1,arr2):
    nd,ns = arr1.shape
    out = np.full((nd,ns),np.nan,dtype=np.float32)
    for c in prange(ns):
        for r in range(nd):
            v1 = arr1[r,c]
            v2 = arr2[r,c]
            if np.isfinite(v1) and np.isfinite(v2):
                out[r,c] = v1+v2
    return out

@MinuteFactorEngine.register("sub")
@njit(cache=True,parallel=True)
def ops_sub(arr1,arr2):
    nd,ns = arr1.shape
    out = np.full((nd,ns),np.nan,dtype=np.float32)
    for c in prange(ns):
        for r in range(nd):
            v1 = arr1[r,c]
            v2 = arr2[r,c]
            if np.isfinite(v1) and np.isfinite(v2):
                out[r,c] = v1-v2
    return out

@MinuteFactorEngine.register("mul")
@njit(cache=True,parallel=True)
def ops_mul(arr1,arr2):
    nd,ns = arr1.shape
    out = np.full((nd,ns),np.nan,dtype=np.float32)
    for c in prange(ns):
        for r in range(nd):
            v1 = arr1[r,c]
            v2 = arr2[r,c]
            if np.isfinite(v1) and np.isfinite(v2):
                out[r,c] = v1*v2
    return out

@MinuteFactorEngine.register("div")
@njit(cache=True,parallel=True)
def ops_div(arr1,arr2):
    nd,ns = arr1.shape
    out = np.full((nd,ns),np.nan,dtype=np.float32)
    for c in prange(ns):
        for r in range(nd):
            v1 = arr1[r,c]
            v2 = arr2[r,c]
            if np.isfinite(v1) and np.isfinite(v2) and np.abs(v2)>1e-12:
                out[r,c] = v1/v2
    return out

@MinuteFactorEngine.register("safe_div", param_order=("eps",))
@njit(cache=True,parallel=True)
def ops_safe_div(arr1,arr2,eps=5e-2):
    nd,ns = arr1.shape
    out = np.full((nd,ns),np.nan,dtype=np.float32)
    feps = np.float32(eps)
    for c in prange(ns):
        for r in range(nd):
            v1 = arr1[r,c]
            v2 = arr2[r,c]
            if np.isfinite(v1) and np.isfinite(v2):
                denom = v2
                if np.abs(denom) < feps:
                    denom = feps if denom >= 0 else -feps
                out[r,c] = v1/denom
    return out

@MinuteFactorEngine.register("pct")
@njit(cache=True,parallel=True)
def ops_pct(arr1,arr2):
    nd,ns = arr1.shape
    out = np.full((nd,ns),np.nan,dtype=np.float32)
    for c in prange(ns):
        for r in range(nd):
            v1 = arr1[r,c]
            v2 = arr2[r,c]
            if np.isfinite(v1) and np.isfinite(v2) and np.abs(v2)>1e-12:
                out[r,c] = v1/v2-1
    return out

@MinuteFactorEngine.register("log")
@njit(cache=True,parallel=True)
def ops_log(arr):
    nd,ns = arr.shape
    out = np.full((nd,ns),np.nan,dtype=np.float32)
    for c in prange(ns):
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v) and v > 0:
                out[r,c] = np.log(v)
    return out

@MinuteFactorEngine.register("abs")
@njit(cache=True,parallel=True)
def ops_abs(arr):
    nd,ns = arr.shape
    out = np.full((nd,ns),np.nan,dtype=np.float32)
    for c in prange(ns):
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                out[r,c] = np.abs(v)
    return out

@MinuteFactorEngine.register("pos")
@njit(cache=True,parallel=True)
def ops_pos(arr):
    nd,ns = arr.shape
    out = np.full((nd,ns),np.nan,dtype=np.float32)
    for c in prange(ns):
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                if v>0:
                    out[r,c] = v
                else:
                    out[r,c] = 0
    return out

@MinuteFactorEngine.register("neg")
@njit(cache=True,parallel=True)
def ops_neg(arr):
    nd,ns = arr.shape
    out = np.full((nd,ns),np.nan,dtype=np.float32)
    for c in prange(ns):
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                out[r,c] = -v
    return out

@MinuteFactorEngine.register("neg_part")
@njit(cache=True,parallel=True)
def ops_neg_part(arr):
    nd,ns = arr.shape
    out = np.full((nd,ns),np.nan,dtype=np.float32)
    for c in prange(ns):
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                if v>0:
                    out[r,c] = 0
                else:
                    out[r,c] = v
    return out


@MinuteFactorEngine.register("identity")
def ops_identity(arr):
    values = np.asarray(arr, dtype=np.float32)
    if values.ndim == 1:
        return values.copy()
    return values.copy()


@MinuteFactorEngine.register("last")
def ops_last(arr):
    values = np.asarray(arr, dtype=np.float32)
    if values.ndim == 1:
        return values.copy()
    if values.shape[0] == 0:
        return np.full(values.shape[1], np.nan, dtype=np.float32)
    return values[-1].copy()

@njit(cache=True)
def _swap2(a,idx,i,j):
    tmp=a[i];a[i]=a[j];a[j]=tmp
    tmpi=idx[i];idx[i]=idx[j];idx[j]=tmpi

@njit(cache=True)
def _insertion_sort_pair(a,idx,lo,hi):
    for i in range(lo+1,hi+1):
        key=a[i]
        keyi=idx[i]
        j=i-1
        while j>=lo and a[j]>key:
            a[j+1]=a[j]
            idx[j+1]=idx[j]
            j-=1
        a[j+1]=key
        idx[j+1]=keyi

@njit(cache=True)
def _partition(a,idx,lo,hi):
    pivot=a[(lo+hi)//2]
    i=lo
    j=hi
    while True:
        while a[i]<pivot:
            i+=1
        while a[j]>pivot:
            j-=1
        if i>=j:
            return j
        _swap2(a,idx,i,j)
        i+=1
        j-=1

@njit(cache=True)
def _quicksort_pair_ws(a,idx,lo,hi,stack_lo,stack_hi):
    if hi-lo<=32:
        _insertion_sort_pair(a,idx,lo,hi)
        return
    top=0
    stack_lo[top]=lo
    stack_hi[top]=hi
    top+=1
    while top>0:
        top-=1
        l=stack_lo[top]
        h=stack_hi[top]
        while h-l>32:
            p=_partition(a,idx,l,h)
            if p-l<h-(p+1):
                stack_lo[top]=p+1
                stack_hi[top]=h
                top+=1
                h=p
            else:
                stack_lo[top]=l
                stack_hi[top]=p
                top+=1
                l=p+1
        _insertion_sort_pair(a,idx,l,h)

@MinuteFactorEngine.register("rank")
@njit(cache=True,parallel=True)
def ops_rank(values):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan,np.float64)
    for i in prange(nd):
        row=values[i]
        cnt=0
        for j in range(ns):
            if np.isfinite(row[j]):
                cnt+=1
        if cnt==0:
            continue
        vals=np.empty(cnt,np.float64)
        idxs=np.empty(cnt,np.int32)
        k=0
        for j in range(ns):
            v=row[j]
            if np.isfinite(v):
                vals[k]=v
                idxs[k]=j
                k+=1
        stack_lo=np.empty(64,np.int32)
        stack_hi=np.empty(64,np.int32)
        _quicksort_pair_ws(vals,idxs,0,cnt-1,stack_lo,stack_hi)
        if cnt==1:
            out[i,idxs[0]]=0.5 
            continue
        # dense rank
        uniq=1
        prev=vals[0]
        for k in range(1,cnt):
            cur=vals[k]
            if cur!=prev:
                uniq+=1
                prev=cur
        if uniq==1:
            for k in range(cnt):
                out[i,idxs[k]]=0.5
        else:
            denom=uniq-1
            r=0
            prev=vals[0]
            out[i,idxs[0]] = 0.0 
            for k in range(1,cnt):
                cur=vals[k]
                if cur!=prev:
                    r+=1
                    prev=cur
                x = r/denom
                out[i,idxs[k]] = x 
    return out

@MinuteFactorEngine.register("neutralize")
@njit(cache=True,parallel=True)
def ops_neutralize(values):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for i in prange(nd):
        s=0.0
        c=0
        for j in range(ns):
            v=values[i,j]
            if np.isfinite(v):
                s+=v
                c+=1
        if c>0:
            mean=s/c
            for j in range(ns):
                v=values[i,j]
                if np.isfinite(v):
                    out[i,j] = v-mean
    return out

@MinuteFactorEngine.register("zscore")
@njit(cache=True,parallel=True)
def ops_zscore(values):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for i in prange(nd):
        s=0.0
        ss=0.0
        c=0
        for j in range(ns):
            v=values[i,j]
            if np.isfinite(v):
                s+=v
                ss+=v*v
                c+=1
        if c==0:
            continue
        elif c==1:
            for j in range(ns):
                if np.isfinite(values[i,j]):
                    out[i,j] = 0.0
        else:
            mean=s/c
            var=(ss-s*s/c)/(c-1)
            if var<=0.0:
                for j in range(ns):
                    if np.isfinite(values[i,j]):
                        out[i,j] = 0.0
            else:
                std=np.sqrt(var)
                for j in range(ns):
                    v=values[i,j]
                    if np.isfinite(v):
                        out[i,j]=(v-mean)/std
    return out

@MinuteFactorEngine.register("scale")
@njit(cache=True,parallel=True)
def ops_scale(values):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for i in prange(nd):
        cs_min = np.inf
        cs_max = -np.inf
        c = 0
        for j in range(ns):
            v=values[i,j]
            if np.isfinite(v):
                cs_min = np.minimum(cs_min,v)
                cs_max = np.maximum(cs_max,v)
                c+=1
        if c>0:
            if cs_min==cs_max:
                for j in range(ns):
                    v=values[i,j]
                    if np.isfinite(v):
                        out[i,j] = 0.5
            else:
                for j in range(ns):
                    v=values[i,j]
                    if np.isfinite(v):
                        out[i,j] = (v-cs_min)/(cs_max-cs_min)
    return out

@MinuteFactorEngine.register("corr",param_order=("shift",))
@njit(cache=True,parallel=True)
def ops_corr(arr1,arr2,shift=0):
    nd,ns = arr1.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    sh = int(shift)
    t0 = 0
    t1 = nd
    if sh>0:
        t0 = sh
    elif sh<0:
        t1 = nd+sh
    if t1-t0<2:
        return out
    for c in prange(ns):
        s1 = 0.0
        s2 = 0.0
        cnt = 0
        for r in range(t0,t1):
            v1 = arr1[r-sh,c]
            v2 = arr2[r,c]
            if np.isfinite(v1) and np.isfinite(v2):
                s1 += v1
                s2 += v2
                cnt += 1
        if cnt < 2:
            continue
        mean1 = s1/ cnt
        mean2 = s2/cnt
        cov = 0.0
        sq1 = 0.0
        sq2 = 0.0
        for r in range(t0,t1):
            v1 = arr1[r-sh,c]
            v2 = arr2[r,c]
            if np.isfinite(v1) and np.isfinite(v2):
                d1 = v1-mean1
                d2 = v2-mean2
                cov += d1*d2
                sq1 += d1*d1
                sq2 += d2*d2
        if sq1>0 and sq2>0:
            out[c] = cov/np.sqrt(sq1*sq2)
    return out


@MinuteFactorEngine.register("ts_rank", param_order=("n",))
@njit(cache=True,parallel=True)
def ops_ts_rank(values, n=5):
    nd,ns = values.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    window = min(max(int(n), 1), nd)
    start = nd - window
    for c in prange(ns):
        latest = values[nd - 1, c]
        if not np.isfinite(latest):
            continue
        less = 0
        equal = 0
        count = 0
        for r in range(start, nd):
            v = values[r, c]
            if not np.isfinite(v):
                continue
            count += 1
            if v < latest:
                less += 1
            elif v == latest:
                equal += 1
        if count > 0:
            out[c] = (less + 0.5 * equal) / count
    return out


@MinuteFactorEngine.register("ts_zscore", param_order=("n",))
@njit(cache=True,parallel=True)
def ops_ts_zscore(values, n=5):
    nd,ns = values.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    window = min(max(int(n), 1), nd)
    start = nd - window
    for c in prange(ns):
        latest = values[nd - 1, c]
        if not np.isfinite(latest):
            continue
        s = 0.0
        sq = 0.0
        count = 0
        for r in range(start, nd):
            v = values[r, c]
            if np.isfinite(v):
                s += v
                sq += v * v
                count += 1
        if count <= 1:
            out[c] = 0.0 if count == 1 else np.nan
            continue
        mean = s / count
        var = (sq / count) - (mean * mean)
        if var <= 0.0:
            out[c] = 0.0
            continue
        out[c] = (latest - mean) / np.sqrt(var)
    return out


@MinuteFactorEngine.register("ts_corr", param_order=("n",))
@njit(cache=True,parallel=True)
def ops_ts_corr(arr1,arr2,n=5):
    nd,ns = arr1.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    window = min(max(int(n), 1), nd)
    start = nd - window
    for c in prange(ns):
        s1 = 0.0
        s2 = 0.0
        count = 0
        for r in range(start, nd):
            v1 = arr1[r, c]
            v2 = arr2[r, c]
            if np.isfinite(v1) and np.isfinite(v2):
                s1 += v1
                s2 += v2
                count += 1
        if count < 2:
            continue
        mean1 = s1 / count
        mean2 = s2 / count
        cov = 0.0
        sq1 = 0.0
        sq2 = 0.0
        for r in range(start, nd):
            v1 = arr1[r, c]
            v2 = arr2[r, c]
            if np.isfinite(v1) and np.isfinite(v2):
                d1 = v1 - mean1
                d2 = v2 - mean2
                cov += d1 * d2
                sq1 += d1 * d1
                sq2 += d2 * d2
        if sq1 > 0 and sq2 > 0:
            out[c] = cov / np.sqrt(sq1 * sq2)
    return out


@MinuteFactorEngine.register("ts_beta", param_order=("n",))
@njit(cache=True,parallel=True)
def ops_ts_beta(arr1,arr2,n=5):
    nd,ns = arr1.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    window = min(max(int(n), 1), nd)
    start = nd - window
    for c in prange(ns):
        sx = 0.0
        sy = 0.0
        count = 0
        for r in range(start, nd):
            x = arr1[r, c]
            y = arr2[r, c]
            if np.isfinite(x) and np.isfinite(y):
                sx += x
                sy += y
                count += 1
        if count < 2:
            continue
        mx = sx / count
        my = sy / count
        cov = 0.0
        varx = 0.0
        for r in range(start, nd):
            x = arr1[r, c]
            y = arr2[r, c]
            if np.isfinite(x) and np.isfinite(y):
                dx = x - mx
                dy = y - my
                cov += dx * dy
                varx += dx * dx
        if varx > 0:
            out[c] = cov / varx
    return out


@MinuteFactorEngine.register("ts_regression", param_order=("n",))
@njit(cache=True,parallel=True)
def ops_ts_regression(arr1,arr2,n=5):
    return ops_ts_beta(arr1,arr2,n)

# ===== Extentions Below ======

# ----- User: zhuzehua Begin -----

# ----- User: zhuzehua End -----

# ----- User: linwensheng Begin -----

# ----- User: linwensheng End -----

# ----- User: zhangchengyuan Begin -----
@MinuteFactorEngine.register("calc_guiji_impl")
@njit(cache=True, parallel=True)
def ops_calc_guiji_impl(closes, opens, amounts):
    nd, ns = closes.shape
    out = np.full(ns, np.nan, dtype=np.float32)
    for j in prange(ns):
        sum_log_ret = 0.0
        sum_amt = 0.0
        valid_count = 0
        for i in range(nd):
            c = closes[i, j]
            o = opens[i, j]
            amt = amounts[i, j]
            if np.isfinite(c) and np.isfinite(o) and o != 0 and np.isfinite(amt) and amt > 0:
                ret = np.abs(c / o - 1)
                sum_log_ret += np.log1p(ret)
                sum_amt += amt
                valid_count += 1
        if sum_amt > 0 and valid_count > 0:
            out[j] = sum_log_ret / sum_amt
            
    return out

@MinuteFactorEngine.register("calc_wcr_impl")
@njit(cache=True, parallel=True)
def ops_calc_wcr_impl(closes, volumes):
    T, N = closes.shape
    out = np.full(N, np.nan, dtype=np.float32)
    for j in prange(N):
        sum_vol_close = 0.0
        total_vol = 0.0
        sum_close = 0.0
        valid_t = 0
        for i in range(T):
            c = closes[i, j]
            v = volumes[i, j]
            if np.isfinite(c) and np.isfinite(v) and v >= 0:
                sum_vol_close += v * c
                total_vol += v
                sum_close += c
                valid_t += 1
        if valid_t > 0 and total_vol > 0 and sum_close > 0:
            numerator = sum_vol_close / total_vol
            denominator = sum_close / valid_t
            out[j] = numerator / denominator
    return out

@MinuteFactorEngine.register("calc_wskew_impl")
@njit(cache=True, parallel=True)
def ops_calc_wskew_impl(closes, volumes):
    T, N = closes.shape
    out = np.full(N, np.nan, dtype=np.float32)
    for j in prange(N):
        sum_close = 0.0
        total_vol = 0.0
        count = 0
        for i in range(T):
            c = closes[i, j]
            v = volumes[i, j]
            if np.isfinite(c) and np.isfinite(v):
                sum_close += c
                total_vol += v
                count += 1
        if count < 2 or total_vol == 0:
            continue
        mean_close = sum_close / count
        sum_vol_cub_diff = 0.0 
        sum_sq_diff = 0.0
        valid_pass2 = False
        for i in range(T):
            c = closes[i, j]
            v = volumes[i, j]
            if np.isfinite(c) and np.isfinite(v):
                diff = c - mean_close
                sum_sq_diff += diff * diff
                sum_vol_cub_diff += v * (diff ** 3)
                valid_pass2 = True
        if valid_pass2:
            var_close = sum_sq_diff / count
            if var_close > 1e-8: # Avoid division by zero if price is flat
                std_close = np.sqrt(var_close)
                numerator = sum_vol_cub_diff / total_vol
                denominator = std_close ** 3
                out[j] = numerator / denominator
            else:
                out[j] = 0.0 # No volatility, skewness is 0
    return out

@MinuteFactorEngine.register("calc_entropy_impl")
@njit(cache=True, parallel=True)
def ops_calc_entropy_impl(closes, volumes):
    T, N = closes.shape
    out = np.full(N, np.nan, dtype=np.float32)
    for j in prange(N):
        total_amt = 0.0
        valid_cnt = 0
        for i in range(T):
            c = closes[i, j]
            v = volumes[i, j]
            if np.isfinite(c) and np.isfinite(v) and v > 0:
                total_amt += c * v
                valid_cnt += 1
        if total_amt <= 1e-4 or valid_cnt < 5:
            out[j] = np.nan 
            continue
        entropy_sum = 0.0
        for i in range(T):
            c = closes[i, j]
            v = volumes[i, j]
            if np.isfinite(c) and np.isfinite(v) and v > 0:
                amt = c * v
                p = amt / total_amt
                if p > 1e-12:
                    entropy_sum += p * np.log(p)
        out[j] = -entropy_sum
    return out

@MinuteFactorEngine.register("calc_corr_impl")
@njit(cache=True, parallel=True)
def ops_calc_corr_impl(closes):
    T, N = closes.shape
    out = np.full(N, np.nan, dtype=np.float32)
    returns = np.empty((T, N), dtype=np.float32)
    market_ret = np.zeros(T, dtype=np.float32)
    for j in prange(N):
        for i in range(T):
            if i == 0:
                returns[i, j] = 0.0 # 第一分钟设为0
            else:
                c_curr = closes[i, j]
                c_prev = closes[i-1, j]
                if np.isfinite(c_curr) and np.isfinite(c_prev) and c_prev > 0:
                    returns[i, j] = c_curr / c_prev - 1
                else:
                    returns[i, j] = np.nan
    for i in range(T):
        sum_r = 0.0
        cnt = 0
        for j in range(N):
            r = returns[i, j]
            if np.isfinite(r):
                sum_r += r
                cnt += 1
        if cnt > 0:
            market_ret[i] = sum_r / cnt
        else:
            market_ret[i] = 0.0
    for j in prange(N):
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_xx = 0.0
        sum_yy = 0.0
        count = 0
        for i in range(1, T): # 跳过第0分钟
            x = returns[i, j]     # 个股收益
            y = market_ret[i]     # 市场收益
            if np.isfinite(x) and np.isfinite(y):
                sum_x += x
                sum_y += y
                sum_xy += x * y
                sum_xx += x * x
                sum_yy += y * y
                count += 1
        if count > 10:
            cov = count * sum_xy - sum_x * sum_y
            var_x = count * sum_xx - sum_x * sum_x
            var_y = count * sum_yy - sum_y * sum_y
            if var_x > 0 and var_y > 0:
                out[j] = cov / np.sqrt(var_x * var_y)
    return out

@MinuteFactorEngine.register("calc_extreme_res_RM_MAX_5")
@njit(cache=True, parallel=True)
def ops_calc_extreme_res_RM_MAX_5(closes):
    T, N = closes.shape
    out = np.full(N, np.nan, dtype=np.float32)
    returns = np.empty((T, N), dtype=np.float32)
    market_metric = np.zeros(T, dtype=np.float32) # 存储 RM
    for j in prange(N):
        for i in range(T):
            if i == 0:
                returns[i, j] = 0.0
            else:
                c = closes[i, j]
                prev = closes[i-1, j]
                if np.isfinite(c) and np.isfinite(prev) and prev > 0:
                    returns[i, j] = c / prev - 1
                else:
                    returns[i, j] = np.nan
    for i in range(1, T):
        sum_r = 0.0
        cnt = 0
        for j in range(N):
            r = returns[i, j]
            if np.isfinite(r):
                sum_r += r
                cnt += 1
        if cnt > 1:
            mean_val = sum_r / cnt
            market_metric[i] = mean_val
        else:
            market_metric[i] = 0.0
    k = 5
    for j in prange(N):
        col_rets = returns[:, j].copy()
        
        for i in range(T):
            if np.isnan(col_rets[i]):
                col_rets[i] = 0.0 
        sorted_indices = np.argsort(col_rets)
        target_indices = sorted_indices[-k:]
        metric_sum = 0.0
        metric_cnt = 0
        for idx in target_indices:
            if idx > 0:
                metric_sum += market_metric[idx]
                metric_cnt += 1
        if metric_cnt > 0:
            out[j] = metric_sum / metric_cnt
    return out

@MinuteFactorEngine.register("calc_extreme_res_STD_MAX_5")
@njit(cache=True, parallel=True)
def ops_calc_extreme_res_STD_MAX_5(closes):
    T, N = closes.shape
    out = np.full(N, np.nan, dtype=np.float32)
    returns = np.empty((T, N), dtype=np.float32)
    market_metric = np.zeros(T, dtype=np.float32)
    for j in prange(N):
        for i in range(T):
            if i == 0:
                returns[i, j] = 0.0
            else:
                c = closes[i, j]
                prev = closes[i-1, j]
                if np.isfinite(c) and np.isfinite(prev) and prev > 0:
                    returns[i, j] = c / prev - 1
                else:
                    returns[i, j] = np.nan
    for i in range(1, T):
        sum_r = 0.0
        sum_sq_r = 0.0
        cnt = 0
        for j in range(N):
            r = returns[i, j]
            if np.isfinite(r):
                sum_r += r
                sum_sq_r += r * r
                cnt += 1
        if cnt > 1:
            mean_val = sum_r / cnt
            var_val = (sum_sq_r / cnt) - (mean_val * mean_val)
            market_metric[i] = np.sqrt(max(0.0, var_val))
        else:
            market_metric[i] = 0.0
    k = 5
    for j in prange(N):
        col_rets = returns[:, j].copy()
        for i in range(T):
            if np.isnan(col_rets[i]):
                col_rets[i] = 0.0
        sorted_indices = np.argsort(col_rets)
        target_indices = sorted_indices[-k:]
        metric_sum = 0.0
        metric_cnt = 0
        for idx in target_indices:
            if idx > 0:
                metric_sum += market_metric[idx]
                metric_cnt += 1
        if metric_cnt > 0:
            out[j] = metric_sum / metric_cnt    
    return out

@MinuteFactorEngine.register("calc_amt_res_MAX_5")
@njit(cache=True, parallel=True)
def ops_calc_amt_res_MAX_5(closes, volumes):
    T, N = closes.shape
    out = np.full(N, np.nan, dtype=np.float32)
    returns = np.empty((T, N), dtype=np.float32)
    amounts = np.empty((T, N), dtype=np.float32)
    mkt_total_amt = np.zeros(T, dtype=np.float32)
    for j in prange(N):
        for i in range(T):
            c = closes[i, j]
            v = volumes[i, j]        
            if i == 0:
                returns[i, j] = 0.0
            else:
                prev = closes[i-1, j]
                if np.isfinite(c) and np.isfinite(prev) and prev > 0:
                    returns[i, j] = c / prev - 1
                else:
                    returns[i, j] = 0.0            
            if np.isfinite(c) and np.isfinite(v):
                amounts[i, j] = c * v
            else:
                amounts[i, j] = 0.0                
    for i in range(T):
        sum_amt = 0.0
        for j in range(N):
            sum_amt += amounts[i, j]
        mkt_total_amt[i] = sum_amt
    k = 5
    for j in prange(N):
        sum_pct_all = 0.0
        cnt_all = 0
        pct_series = np.zeros(T, dtype=np.float32)    
        for i in range(T):
            if mkt_total_amt[i] > 0:
                pct = amounts[i, j] / mkt_total_amt[i]
                pct_series[i] = pct
                sum_pct_all += pct
                cnt_all += 1
        if cnt_all == 0 or sum_pct_all == 0:
            continue
        avg_pct_all = sum_pct_all / cnt_all
        col_rets = returns[:, j]
        sorted_indices = np.argsort(col_rets)
        target_indices = sorted_indices[-k:]
        sum_pct_extr = 0.0
        cnt_extr = 0
        for idx in target_indices:
            sum_pct_extr += pct_series[idx]
            cnt_extr += 1    
        if cnt_extr > 0:
            avg_pct_extr = sum_pct_extr / cnt_extr
            out[j] = avg_pct_extr / avg_pct_all
    return out

@MinuteFactorEngine.register("calc_amt_res_MIN_5")
@njit(cache=True, parallel=True)
def ops_calc_amt_res_MIN_5(closes, volumes):
    T, N = closes.shape
    out = np.full(N, np.nan, dtype=np.float32)
    returns = np.empty((T, N), dtype=np.float32)
    amounts = np.empty((T, N), dtype=np.float32)
    mkt_total_amt = np.zeros(T, dtype=np.float32)
    for j in prange(N):
        for i in range(T):
            c = closes[i, j]
            v = volumes[i, j]    
            if i == 0:
                returns[i, j] = 0.0
            else:
                prev = closes[i-1, j]
                if np.isfinite(c) and np.isfinite(prev) and prev > 0:
                    returns[i, j] = c / prev - 1
                else:
                    returns[i, j] = 0.0    
            if np.isfinite(c) and np.isfinite(v):
                amounts[i, j] = c * v
            else:
                amounts[i, j] = 0.0            
    for i in range(T):
        sum_amt = 0.0
        for j in range(N):
            sum_amt += amounts[i, j]
        mkt_total_amt[i] = sum_amt
    k = 5
    for j in prange(N):
        sum_pct_all = 0.0
        cnt_all = 0
        pct_series = np.zeros(T, dtype=np.float32)
        for i in range(T):
            if mkt_total_amt[i] > 0:
                pct = amounts[i, j] / mkt_total_amt[i]
                pct_series[i] = pct
                sum_pct_all += pct
                cnt_all += 1
        if cnt_all == 0 or sum_pct_all == 0:
            continue
        avg_pct_all = sum_pct_all / cnt_all
        col_rets = returns[:, j]
        sorted_indices = np.argsort(col_rets)
        target_indices = sorted_indices[:k]
        sum_pct_extr = 0.0
        cnt_extr = 0
        for idx in target_indices:
            sum_pct_extr += pct_series[idx]
            cnt_extr += 1
        if cnt_extr > 0:
            avg_pct_extr = sum_pct_extr / cnt_extr
            out[j] = avg_pct_extr / avg_pct_all

    return out

@MinuteFactorEngine.register("calc_pre_amt_MAX_30")
@njit(cache=True, parallel=True)
def ops_calc_pre_amt_MAX_30(closes, volumes):
    T, N = closes.shape
    out = np.full(N, np.nan, dtype=np.float32)
    returns = np.empty((T, N), dtype=np.float32)
    amounts = np.empty((T, N), dtype=np.float32)
    mkt_total_amt = np.zeros(T, dtype=np.float32)
    for j in prange(N):
        for i in range(T):
            c = closes[i, j]
            v = volumes[i, j]    
            if i == 0:
                returns[i, j] = 0.0
            else:
                prev = closes[i-1, j]
                if np.isfinite(c) and np.isfinite(prev) and prev > 0:
                    returns[i, j] = c / prev - 1
                else:
                    returns[i, j] = 0.0
            if np.isfinite(c) and np.isfinite(v):
                amounts[i, j] = c * v
            else:
                amounts[i, j] = 0.0
    for i in range(T):
        sum_amt = 0.0
        for j in range(N):
            sum_amt += amounts[i, j]
        mkt_total_amt[i] = sum_amt
    k = 30
    for j in prange(N):
        pct_series = np.zeros(T, dtype=np.float32)
        sum_pct_all = 0.0
        cnt_all = 0
        for i in range(T):
            if mkt_total_amt[i] > 0:
                pct = amounts[i, j] / mkt_total_amt[i]
                pct_series[i] = pct
                sum_pct_all += pct
                cnt_all += 1
        if cnt_all == 0 or sum_pct_all == 0:
            continue
        avg_pct_daily = sum_pct_all / cnt_all
        col_rets = returns[:, j]
        sorted_indices = np.argsort(col_rets)
        target_indices = sorted_indices[-k:]
        sum_pre_pct = 0.0
        cnt_valid = 0
        for idx in target_indices:
            pre_idx = idx - 1
            if pre_idx >= 0 and pre_idx < T:
                sum_pre_pct += pct_series[pre_idx]
                cnt_valid += 1
        if cnt_valid > 0:
            avg_pre_pct = sum_pre_pct / cnt_valid
            out[j] = avg_pre_pct / avg_pct_daily
    return out

@MinuteFactorEngine.register("calc_pre_amt_MIN_30")
@njit(cache=True, parallel=True)
def ops_calc_pre_amt_MIN_30(closes, volumes):
    T, N = closes.shape
    out = np.full(N, np.nan, dtype=np.float32)
    returns = np.empty((T, N), dtype=np.float32)
    amounts = np.empty((T, N), dtype=np.float32)
    mkt_total_amt = np.zeros(T, dtype=np.float32)
    for j in prange(N):
        for i in range(T):
            c = closes[i, j]
            v = volumes[i, j]    
            if i == 0:
                returns[i, j] = 0.0
            else:
                prev = closes[i-1, j]
                if np.isfinite(c) and np.isfinite(prev) and prev > 0:
                    returns[i, j] = c / prev - 1
                else:
                    returns[i, j] = 0.0
            if np.isfinite(c) and np.isfinite(v):
                amounts[i, j] = c * v
            else:
                amounts[i, j] = 0.0
    for i in range(T):
        sum_amt = 0.0
        for j in range(N):
            sum_amt += amounts[i, j]
        mkt_total_amt[i] = sum_amt
    k = 30
    for j in prange(N):
        pct_series = np.zeros(T, dtype=np.float32)
        sum_pct_all = 0.0
        cnt_all = 0        
        for i in range(T):
            if mkt_total_amt[i] > 0:
                pct = amounts[i, j] / mkt_total_amt[i]
                pct_series[i] = pct
                sum_pct_all += pct
                cnt_all += 1        
        if cnt_all == 0 or sum_pct_all == 0:
            continue        
        avg_pct_daily = sum_pct_all / cnt_all
        col_rets = returns[:, j]
        sorted_indices = np.argsort(col_rets)
        target_indices = sorted_indices[:k]        
        sum_pre_pct = 0.0
        cnt_valid = 0        
        for idx in target_indices:
            pre_idx = idx - 1
            if pre_idx >= 0 and pre_idx < T:
                sum_pre_pct += pct_series[pre_idx]
                cnt_valid += 1        
        if cnt_valid > 0:
            avg_pre_pct = sum_pre_pct / cnt_valid
            out[j] = avg_pre_pct / avg_pct_daily    
    return out
# ----- User: zhangchengyuan End -----

# ----- User: liangxiwen Begin -----
@MinuteFactorEngine.register("dazzling_vol")
@njit(cache=True,parallel=True)
def ops_dazzling_vol(closes,volumes):
    nd,ns = closes.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    win = 5
    for c in prange(ns):
        s = 0.0
        sq = 0.0
        cnt = 0
        for t in range(1,nd):
            v1 = volumes[t,c]
            v0 = volumes[t-1,c]
            if np.isfinite(v1) and np.isfinite(v0):
                dv = v1-v0
                if np.isfinite(dv):
                    s += dv
                    sq += dv*dv
                    cnt += 1
        if cnt==0:
            continue
        mean_dv = s/cnt
        std_dv = 0.0
        if cnt>1:
            var = (sq/cnt)-mean_dv*mean_dv
            if var>0.0:
                std_dv = np.sqrt(var)
        thr = mean_dv+std_dv
        sum_evt = 0.0
        cnt_evt = 0
        for m in range(1,nd):
            v1 = volumes[m,c]
            v0 = volumes[m-1,c]
            if not(np.isfinite(v1) and np.isfinite(v0)):
                continue
            dv = v1-v0
            if not np.isfinite(dv):
                continue
            if dv<=thr:
                continue
            endm = m+win-1
            if endm>=nd:
                endm = nd-1
            rs = 0.0
            rsq = 0.0
            rcnt = 0
            for t in range(m,endm+1):
                p1 = closes[t,c]
                p0 = closes[t-1,c]
                if np.isfinite(p1) and np.isfinite(p0) and np.abs(p0)>1e-12:
                    r = p1/p0-1.0
                    if np.isfinite(r):
                        rs += r
                        rsq += r*r
                        rcnt += 1
            if rcnt<2:
                continue
            rmean = rs/rcnt
            rvar = (rsq/rcnt)-rmean*rmean
            if rvar<0.0:
                rvar = 0.0
            rstd = np.sqrt(rvar)
            sum_evt += rstd
            cnt_evt += 1
        if cnt_evt>0:
            out[c] = sum_evt/cnt_evt
    return out

@MinuteFactorEngine.register("dazzling_ret")
@njit(cache=True,parallel=True)
def ops_dazzling_ret(closes,volumes):
    nm,ns = closes.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    win = 5
    for c in prange(ns):
        s = 0.0
        sq = 0.0
        cnt = 0
        for t in range(1,nm):
            v1 = volumes[t,c]
            v0 = volumes[t-1,c]
            if np.isfinite(v1) and np.isfinite(v0):
                dv = v1-v0
                if np.isfinite(dv):
                    s += dv
                    sq += dv*dv
                    cnt += 1
        if cnt==0:
            continue
        mean_dv = s/cnt
        std_dv = 0.0
        if cnt>1:
            var = (sq/cnt)-mean_dv*mean_dv
            if var>0.0:
                std_dv = np.sqrt(var)
        thr = mean_dv+std_dv
        sum_evt = 0.0
        cnt_evt = 0
        for t in range(1,nm):
            v1 = volumes[t,c]
            v0 = volumes[t-1,c]
            if not(np.isfinite(v1) and np.isfinite(v0)):
                continue
            dv = v1-v0
            if not np.isfinite(dv):
                continue
            if dv<=thr:
                continue
            endt = t+win-1
            if endt>=nm:
                endt = nm-1
            p0 = closes[t-1,c]
            p1 = closes[endt,c]
            if np.isfinite(p0) and np.isfinite(p1) and np.abs(p0)>1e-12:
                r = p1/p0-1.0
                if np.isfinite(r):
                    sum_evt += r
                    cnt_evt += 1
        if cnt_evt>0:
            out[c] = sum_evt/cnt_evt
    return out

@MinuteFactorEngine.register("nozero_abs")
@njit(cache=True,parallel=True)
def ops_nozero_abs(arr):
    nd,ns = arr.shape
    out = np.full((nd,ns),np.nan,dtype=np.float32)
    for j in prange(ns):
        for i in range(nd):
            v = arr[i,j]
            if np.isfinite(v) and v!=0:
                out[i,j] = np.abs(v)
    return out
# ----- User: liangxiwen End -----
