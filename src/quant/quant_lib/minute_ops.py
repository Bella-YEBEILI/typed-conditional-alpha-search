import os
os.environ["NUMBA_NUM_THREADS"] = "12"
import numpy as np
from numba import njit,prange
from quant.quant_lib.minute_tools import MinuteFactorEngine

# ===== abstract =====

# 2D->1D:
# mean,std,sum,min,max,maxdd,skew,kurt,var,cvar
# 2D->2D:
# log,abs,pos,neg,neg_part,rank,neutralize,zscore,scale
# [2D,2D]->1D:
# corr
# [2D,2D]->2D:
# add,sub,mul,div,pct,weight_by

# ===== Basic Below ======

# 2D -> 1D

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

@MinuteFactorEngine.register("wmean")
@njit(cache=True,parallel=True)
def ops_wmean(arr_values,arr_weights):
    nd,ns = arr_values.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    for c in prange(ns):
        s = 0.0
        w = 0.0
        for r in range(nd):
            v = arr_values[r,c]
            wt = arr_weights[r,c]
            if np.isfinite(v) and np.isfinite(wt):
                s += v*wt
                w += wt
        if w > 1e-12:
            out[c] = s/w
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

@MinuteFactorEngine.register("wsum")
@njit(cache=True,parallel=True)
def ops_wsum(arr_values,arr_weights):
    nd,ns = arr_values.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    for c in prange(ns):
        s = 0.0
        any_valid = False
        for r in range(nd):
            v = arr_values[r,c]
            wt = arr_weights[r,c]
            if np.isfinite(v) and np.isfinite(wt):
                s += v*wt
                any_valid = True
        if any_valid:
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

@MinuteFactorEngine.register("maxdd")
@njit(cache=True,parallel=True)
def ops_maxdd(arr):
    nd,ns = arr.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    for j in prange(ns):
        nav = 1.0
        peak = 1.0
        mdd = 0.0
        cnt = 0
        for i in range(nd):
            r = arr[i,j]
            if np.isfinite(r):
                nav *= 1.0+r
                if nav>peak:
                    peak = nav
                else:
                    dd = 1.0-nav/peak
                    if dd>mdd:
                        mdd = dd
                cnt += 1
        if cnt>0:
            out[j] = mdd
    return out

@MinuteFactorEngine.register("skew")
@njit(cache=True,parallel=True)
def ops_skew(arr):
    nd,ns = arr.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    for c in prange(ns):
        s1 = 0.0
        cnt = 0
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                s1 += v
                cnt += 1
        if cnt<3:
            continue
        mu = s1/cnt
        m2 = 0.0
        m3 = 0.0
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                d = v-mu
                d2 = d*d
                m2 += d2
                m3 += d2*d
        if m2<=0:
            continue
        m2 /= cnt
        m3 /= cnt
        g1 = m3/(m2*np.sqrt(m2))
        out[c] = np.sqrt(cnt*(cnt-1.0))/(cnt-2.0)*g1
    return out

@MinuteFactorEngine.register("kurt")
@njit(cache=True,parallel=True)
def ops_kurt(arr):
    nd,ns = arr.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    for c in prange(ns):
        s1 = 0.0
        cnt = 0
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                s1 += v
                cnt += 1
        if cnt<4:
            continue

        mu = s1/cnt
        m2 = 0.0
        m4 = 0.0
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                d = v-mu
                d2 = d*d
                m2 += d2
                m4 += d2*d2

        if m2<=0:
            continue

        m2 /= cnt
        m4 /= cnt
        g2 = m4/(m2*m2)-3.0
        out[c] = ((cnt-1.0)/((cnt-2.0)*(cnt-3.0)))*((cnt+1.0)*g2+6.0)
    return out

# var or cvar: 刻画尾部风险
@MinuteFactorEngine.register("var",param_order=("alpha",))
@njit(cache=True,parallel=True)
def ops_var(arr,alpha):
    nd,ns = arr.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    if not (0.0<alpha<=1.0):
        return out
    for c in prange(ns):
        cnt = 0
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                cnt += 1
        if cnt==0:
            continue

        vals = np.empty(cnt,dtype=np.float32)
        k0 = 0
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                vals[k0] = v
                k0 += 1

        vals.sort()
        k = int(np.ceil(alpha*cnt))
        if k<1:
            k = 1
        elif k>cnt:
            k = cnt
        out[c] = vals[k-1]
    return out

@MinuteFactorEngine.register("cvar",param_order=("alpha",))
@njit(cache=True,parallel=True)
def ops_cvar(arr,alpha):
    nd,ns = arr.shape
    out = np.full(ns,np.nan,dtype=np.float32)
    if not (0.0<alpha<=1.0):
        return out
    for c in prange(ns):
        cnt = 0
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                cnt += 1
        if cnt==0:
            continue

        vals = np.empty(cnt,dtype=np.float32)
        k0 = 0
        for r in range(nd):
            v = arr[r,c]
            if np.isfinite(v):
                vals[k0] = v
                k0 += 1

        vals.sort()
        k = int(np.ceil(alpha*cnt))
        if k<1:
            k = 1
        elif k>cnt:
            k = cnt

        s = 0.0
        for i in range(k):
            s += vals[i]
        out[c] = s/k
    return out

# 2D -> 2D
@MinuteFactorEngine.register("identity")
def ops_identity(arr):
    values = np.asarray(arr, dtype=np.float32)
    if values.ndim == 1:
        return values.copy()
    return values.copy()

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


# [2D,2D] -> 1D

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

# [2D,2D] -> 2D

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

@MinuteFactorEngine.register("weight_by")
@njit(cache=True,parallel=True)
def ops_weight_by(arr1,arr2):
    nd,ns = arr1.shape
    out = np.full((nd,ns),np.nan,dtype=np.float32)
    for c in prange(ns):
        den = 0.0
        for r in range(nd):
            w = arr2[r,c]
            if np.isfinite(w) and w>0:
                den += w
        if den<=0:
            continue
        for r in range(nd):
            x = arr1[r,c]
            w = arr2[r,c]
            if np.isfinite(x) and np.isfinite(w) and w>0:
                out[r,c] = x*w/den
    return out


