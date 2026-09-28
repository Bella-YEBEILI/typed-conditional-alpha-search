import numpy as np
from numba import njit,prange

"""
rank base
"""
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

@njit(cache=True)
def _sort_pair(vals,idxs,cnt):
    if cnt>1:
        stack_lo = np.empty(cnt,np.int32)
        stack_hi = np.empty(cnt,np.int32)
        _quicksort_pair_ws(vals,idxs,0,cnt-1,stack_lo,stack_hi)

@njit(cache=True)
def _rank_write(vals,idxs,cnt,out,row,ascending):
    if cnt==0:
        return
    if cnt==1:
        out[row,idxs[0]] = 0.5
        return
    _sort_pair(vals,idxs,cnt)
    uniq = 1
    prev = vals[0]
    for k in range(1,cnt):
        cur = vals[k]
        if cur!=prev:
            uniq += 1
            prev = cur
    if uniq==1:
        for k in range(cnt):
            out[row,idxs[k]] = 0.5
        return
    denom = uniq-1
    r = 0
    prev = vals[0]
    for k in range(cnt):
        cur = vals[k]
        if k>0 and cur!=prev:
            r += 1
            prev = cur
        z = r/denom
        out[row,idxs[k]] = z if ascending else 1.0-z

@njit(cache=True)
def _rank_to_1d(vals,idxs,cnt,out,ascending):
    if cnt==0:
        return
    if cnt==1:
        out[idxs[0]] = 0.5
        return

    _sort_pair(vals,idxs,cnt)

    uniq = 1
    prev = vals[0]
    for k in range(1,cnt):
        cur = vals[k]
        if cur!=prev:
            uniq += 1
            prev = cur

    if uniq==1:
        for k in range(cnt):
            out[idxs[k]] = 0.5
        return

    denom = uniq-1
    r = 0
    prev = vals[0]
    for k in range(cnt):
        cur = vals[k]
        if k>0 and cur!=prev:
            r += 1
            prev = cur
        z = r/denom
        out[idxs[k]] = z if ascending else 1.0-z

"""
construct condition 
"""
@njit(cache=True,parallel=True)
def eq_scalar_kernel(x,y):
    nd,ns = x.shape
    out = np.empty((nd,ns),dtype=np.int8)
    if not np.isfinite(y):
        out[:,:] = -1
        return out
    for j in prange(ns):
        for i in range(nd):
            xv = x[i,j]
            if np.isfinite(xv):
                out[i,j] = 1 if xv==y else 0
            else:
                out[i,j] = -1
    return out

@njit(cache=True,parallel=True)
def eq_array_kernel(x,y):
    nd,ns = x.shape
    out = np.empty((nd,ns),dtype=np.int8)
    for j in prange(ns):
        for i in range(nd):
            xv = x[i,j]
            yv = y[i,j]
            if np.isfinite(xv) and np.isfinite(yv):
                out[i,j] = 1 if xv==yv else 0
            else:
                out[i,j] = -1
    return out

@njit(cache=True,parallel=True)
def le_scalar_kernel(x,y):
    nd,ns = x.shape
    out = np.empty((nd,ns),dtype=np.int8)
    if not np.isfinite(y):
        out[:,:] = -1
        return out
    for j in prange(ns):
        for i in range(nd):
            xv = x[i,j]
            if np.isfinite(xv):
                out[i,j] = 1 if xv<=y else 0
            else:
                out[i,j] = -1
    return out

@njit(cache=True,parallel=True)
def le_array_kernel(x,y):
    nd,ns = x.shape
    out = np.empty((nd,ns),dtype=np.int8)
    for j in prange(ns):
        for i in range(nd):
            xv = x[i,j]
            yv = y[i,j]
            if np.isfinite(xv) and np.isfinite(yv):
                out[i,j] = 1 if xv<=yv else 0
            else:
                out[i,j] = -1
    return out

@njit(cache=True,parallel=True)
def ge_scalar_kernel(x,y):
    nd,ns = x.shape
    out = np.empty((nd,ns),dtype=np.int8)
    if not np.isfinite(y):
        out[:,:] = -1
        return out
    for j in prange(ns):
        for i in range(nd):
            xv = x[i,j]
            if np.isfinite(xv):
                out[i,j] = 1 if xv>=y else 0
            else:
                out[i,j] = -1
    return out

@njit(cache=True,parallel=True)
def ge_array_kernel(x,y):
    nd,ns = x.shape
    out = np.empty((nd,ns),dtype=np.int8)
    for j in prange(ns):
        for i in range(nd):
            xv = x[i,j]
            yv = y[i,j]
            if np.isfinite(xv) and np.isfinite(yv):
                out[i,j] = 1 if xv>=yv else 0
            else:
                out[i,j] = -1
    return out

@njit(cache=True,parallel=True)
def con_and_kernel(x,y):
    nd,ns = x.shape
    out = np.empty((nd,ns),dtype=np.int8)
    for i in prange(nd):
        for j in range(ns):
            a = x[i,j]
            b = y[i,j]
            if a==0 or b==0:
                out[i,j] = 0
            elif a==1 and b==1:
                out[i,j] = 1
            else:
                out[i,j] = -1
    return out

@njit(cache=True,parallel=True)
def con_or_kernel(x,y):
    nd,ns = x.shape
    out = np.empty((nd,ns),dtype=np.int8)
    for i in prange(nd):
        for j in range(ns):
            a = x[i,j]
            b = y[i,j]
            if a==1 or b==1:
                out[i,j] = 1
            elif a==0 and b==0:
                out[i,j] = 0
            else:
                out[i,j] = -1
    return out

@njit(cache=True,parallel=True)
def con_not_kernel(x):
    nd,ns = x.shape
    out = np.empty((nd,ns),dtype=np.int8)
    for i in prange(nd):
        for j in range(ns):
            a = x[i,j]
            if a==1:
                out[i,j] = 0
            elif a==0:
                out[i,j] = 1
            else:
                out[i,j] = -1
    return out

"""
construct group
"""
@njit(cache=True,parallel=True)
def bucket_kernel(x,n):
    nd,ns = x.shape
    out = np.full((nd,ns),-1,dtype=np.int32)
    for i in prange(nd):
        m = 0
        idxs = np.empty(ns,dtype=np.int32)
        vals = np.empty(ns,dtype=np.float64)
        stack_lo = np.empty(ns,dtype=np.int32)
        stack_hi = np.empty(ns,dtype=np.int32)
        for j in range(ns):
            val = x[i,j]
            if np.isfinite(val):
                idxs[m] = j
                vals[m] = val
                m += 1
        if m==0:
            continue
        if m==1:
            out[i,idxs[0]] = 0
            continue
        _quicksort_pair_ws(vals,idxs,0,m-1,stack_lo,stack_hi)
        rank = 0
        ranks = np.empty(m,dtype=np.int32)
        ranks[0] = 0
        for j in range(1,m):
            if vals[j]!=vals[j-1]:
                rank += 1
            ranks[j] = rank
        if rank==0:
            for j in range(m):
                out[i,idxs[j]] = 0
        else:
            for j in range(m):
                g = int(np.floor(ranks[j]/rank*(n-1e-9)))
                if g<0:
                    g = 0
                if g>=n:
                    g = n-1
                out[i,idxs[j]] = g
    return out

@njit(cache=True,parallel=True)
def group_merge_kernel(x,y):
    nd,ns = x.shape
    max_y = -1
    for i in range(nd):
        for j in range(ns):
            v = y[i,j]
            if v>max_y:
                max_y = v
    if max_y<0:
        return np.full((nd,ns),-1,dtype=np.int32)
    y_count = max_y+1
    out = np.empty((nd,ns),dtype=np.int32)
    for i in prange(nd):
        for j in range(ns):
            a = x[i,j]
            b = y[i,j]
            if a<0 or b<0:
                out[i,j] = -1
            else:
                out[i,j] = a*y_count+b
    return out

"""
cross section
"""
@njit(cache=True,parallel=True)
def cs_rank_kernel(x,pct,ascending):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan,np.float64)
    for i in prange(nd):
        row = x[i]
        cnt = 0
        for j in range(ns):
            if np.isfinite(row[j]):
                cnt += 1
        if cnt==0:
            continue
        vals = np.empty(cnt,np.float64)
        idxs = np.empty(cnt,np.int32)
        k = 0
        for j in range(ns):
            v = row[j]
            if np.isfinite(v):
                vals[k] = v
                idxs[k] = j
                k += 1
        stack_lo = np.empty(ns,np.int32)
        stack_hi = np.empty(ns,np.int32)
        _quicksort_pair_ws(vals,idxs,0,cnt-1,stack_lo,stack_hi)
        if cnt==1:
            out[i,idxs[0]] = 0.5 if pct else 1.0
            continue
        uniq = 1
        prev = vals[0]
        for k in range(1,cnt):
            cur = vals[k]
            if cur!=prev:
                uniq += 1
                prev = cur
        if pct:
            if uniq==1:
                for k in range(cnt):
                    out[i,idxs[k]] = 0.5
            else:
                denom = uniq-1
                r = 0
                prev = vals[0]
                out[i,idxs[0]] = 0.0 if ascending else 1.0
                for k in range(1,cnt):
                    cur = vals[k]
                    if cur!=prev:
                        r += 1
                        prev = cur
                    z = r/denom
                    out[i,idxs[k]] = z if ascending else 1-z
        else:
            r = 1
            prev = vals[0]
            out[i,idxs[0]] = 1.0 if ascending else float(uniq)
            for k in range(1,cnt):
                cur = vals[k]
                if cur!=prev:
                    r += 1
                    prev = cur
                out[i,idxs[k]] = float(r) if ascending else float(uniq-r+1)
    return out

@njit(cache=True,parallel=True)
def cs_neutralize_kernel(x):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for i in prange(nd):
        s = 0.0
        c = 0
        for j in range(ns):
            v = x[i,j]
            if np.isfinite(v):
                s += v
                c += 1
        if c>0:
            mean = s/c
            for j in range(ns):
                v = x[i,j]
                if np.isfinite(v):
                    out[i,j] = v-mean
    return out

@njit(cache=True,parallel=True)
def cs_zscore_kernel(x):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for i in prange(nd):
        s = 0.0
        ss = 0.0
        c = 0
        for j in range(ns):
            v = x[i,j]
            if np.isfinite(v):
                s += v
                ss += v*v
                c += 1
        if c==0:
            continue
        elif c==1:
            for j in range(ns):
                if np.isfinite(x[i,j]):
                    out[i,j] = 0.0
        else:
            mean = s/c
            var = (ss-s*s/c)/(c-1)
            if var<=0.0:
                for j in range(ns):
                    if np.isfinite(x[i,j]):
                        out[i,j] = 0.0
            else:
                std = np.sqrt(var)
                for j in range(ns):
                    v = x[i,j]
                    if np.isfinite(v):
                        out[i,j] = (v-mean)/std
    return out

@njit(cache=True,parallel=True)
def cs_weighted_zscore_kernel(x,w):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for i in prange(nd):
        sw = 0.0
        swv = 0.0
        c = 0
        for j in range(ns):
            v = x[i,j]
            z = w[i,j]
            if np.isfinite(v) and np.isfinite(z):
                sw += z
                swv += z*v
                c += 1
        if c==0:
            continue
        elif c==1:
            for j in range(ns):
                if np.isfinite(x[i,j]) and np.isfinite(w[i,j]):
                    out[i,j] = 0.0
        else:
            if sw==0.0:
                continue
            mean = swv/sw
            ss = 0.0
            for j in range(ns):
                v = x[i,j]
                z = w[i,j]
                if np.isfinite(v) and np.isfinite(z):
                    ss += z*(v-mean)*(v-mean)
            var = ss/(c-1)
            if var<=0.0:
                for j in range(ns):
                    if np.isfinite(x[i,j]) and np.isfinite(w[i,j]):
                        out[i,j] = 0.0
            else:
                std = np.sqrt(var)
                for j in range(ns):
                    v = x[i,j]
                    z = w[i,j]
                    if np.isfinite(v) and np.isfinite(z):
                        out[i,j] = (v-mean)/std
    return out

@njit(cache=True,parallel=True)
def cs_scale_kernel(x):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for i in prange(nd):
        mn = np.inf
        mx = -np.inf
        c = 0
        for j in range(ns):
            v = x[i,j]
            if np.isfinite(v):
                mn = np.minimum(mn,v)
                mx = np.maximum(mx,v)
                c += 1
        if c>0:
            if mn==mx:
                for j in range(ns):
                    v = x[i,j]
                    if np.isfinite(v):
                        out[i,j] = 0.5
            else:
                for j in range(ns):
                    v = x[i,j]
                    if np.isfinite(v):
                        out[i,j] = (v-mn)/(mx-mn)
    return out

@njit(cache=True,parallel=True)
def group_rank_kernel(x,g):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    n = int(g.max()+1)
    if n<=0:
        return out
    for i in prange(nd):
        gcnt = np.zeros(n,dtype=np.int32)
        for j in range(ns):
            z = g[i,j]
            v = x[i,j]
            if z>=0 and np.isfinite(v):
                gcnt[z] += 1
        gs = np.empty(n+1,dtype=np.int32)
        gs[0] = 0
        for z in range(n):
            gs[z+1] = gs[z]+gcnt[z]
        tot = gs[n]
        if tot==0:
            continue
        wp = np.empty(n,dtype=np.int32)
        for z in range(n):
            wp[z] = gs[z]
        vals = np.empty(tot,dtype=np.float64)
        idxs = np.empty(tot,dtype=np.int32)
        for j in range(ns):
            z = g[i,j]
            v = x[i,j]
            if z>=0 and np.isfinite(v):
                p = wp[z]
                vals[p] = v
                idxs[p] = j
                wp[z] = p+1
        sl = np.empty(tot,dtype=np.int32)
        sh = np.empty(tot,dtype=np.int32)
        rb = np.empty(tot,dtype=np.int32)
        for z in range(n):
            c = gcnt[z]
            if c<=0:
                continue
            a = gs[z]
            b = a+c
            if c==1:
                out[i,idxs[a]] = 0.5
                continue
            _quicksort_pair_ws(vals,idxs,a,b-1,sl,sh)
            r = 0
            prev = vals[a]
            rb[a] = 0
            for k in range(a+1,b):
                cur = vals[k]
                if cur!=prev:
                    r += 1
                    prev = cur
                rb[k] = r
            if r==0:
                for k in range(a,b):
                    out[i,idxs[k]] = 0.5
            else:
                for k in range(a,b):
                    out[i,idxs[k]] = rb[k]/r
    return out

@njit(cache=True,parallel=True)
def group_neutralize_kernel(x,g):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    n = int(g.max()+1)
    if n<=0:
        return out
    for i in prange(nd):
        s = np.zeros(n,dtype=np.float64)
        c = np.zeros(n,dtype=np.int32)
        for j in range(ns):
            z = g[i,j]
            v = x[i,j]
            if z>=0 and np.isfinite(v):
                s[z] += v
                c[z] += 1
        for j in range(ns):
            z = g[i,j]
            v = x[i,j]
            if z>=0 and np.isfinite(v):
                out[i,j] = v-s[z]/c[z]
    return out

@njit(cache=True,parallel=True)
def group_zscore_kernel(x,g):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    n = int(g.max()+1)
    if n<=0:
        return out
    for i in prange(nd):
        s = np.zeros(n,dtype=np.float64)
        ss = np.zeros(n,dtype=np.float64)
        c = np.zeros(n,dtype=np.int32)
        for j in range(ns):
            z = g[i,j]
            v = x[i,j]
            if z>=0 and np.isfinite(v):
                s[z] += v
                ss[z] += v*v
                c[z] += 1
        for j in range(ns):
            z = g[i,j]
            v = x[i,j]
            if z>=0 and np.isfinite(v):
                cnt = c[z]
                if cnt==1:
                    out[i,j] = 0.0
                else:
                    mean = s[z]/cnt
                    var = (ss[z]-s[z]*s[z]/cnt)/(cnt-1)
                    if var<=0.0:
                        out[i,j] = 0.0
                    else:
                        out[i,j] = (v-mean)/np.sqrt(var)
    return out

@njit(cache=True,parallel=True)
def group_scale_kernel(x,g):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    n = int(g.max()+1)
    if n<=0:
        return out
    for i in prange(nd):
        mx = np.full(n,-np.inf)
        mn = np.full(n,np.inf)
        for j in range(ns):
            z = g[i,j]
            v = x[i,j]
            if z>=0 and np.isfinite(v):
                if v>mx[z]:
                    mx[z] = v
                if v<mn[z]:
                    mn[z] = v
        for j in range(ns):
            z = g[i,j]
            v = x[i,j]
            if z>=0 and np.isfinite(v):
                if mx[z]-mn[z]<1e-8:
                    out[i,j] = 0.5
                else:
                    out[i,j] = (v-mn[z])/(mx[z]-mn[z])
    return out

"""
ts
"""
@njit(cache=True,parallel=True)
def ts_rank_kernel(x,n,min_periods,pct):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            v = x[i,j]
            if np.isfinite(v):
                start = max(0,i+1-n)
                c = 0
                s = 0
                b = 0
                for t in range(start,i):
                    z = x[t,j]
                    if np.isfinite(z):
                        if z<v:
                            s += 1
                        elif z>v:
                            b += 1
                        c += 1
                c += 1
                if c<min_periods:
                    continue
                r = (c+s-b+1)/2
                if pct:
                    r = r/c
                out[i,j] = r
    return out

@njit(cache=True,parallel=True)
def ts_decay_linear_kernel(x,n):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(nd):
            v = x[i,j]
            if np.isfinite(v):
                num = 0.0
                den = 0.0
                for t in range(n):
                    k = i-t
                    w = n-t
                    if k<0:
                        break
                    z = x[k,j]
                    if np.isfinite(z):
                        num += z*w
                        den += w
                out[i,j] = num/den
    return out

@njit(cache=True,parallel=True)
def ts_sum_kernel(x,n,min_periods):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            s = 0.0
            c = 0
            for t in range(start,i+1):
                v = x[t,j]
                if np.isfinite(v):
                    s += v
                    c += 1
            if c>=min_periods:
                out[i,j] = s
    return out

@njit(cache=True,parallel=True)
def ts_prod_kernel(x,n,min_periods):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            p = 1.0
            c = 0
            for t in range(start,i+1):
                v = x[t,j]
                if np.isfinite(v):
                    p *= v
                    c += 1
            if c>=min_periods:
                out[i,j] = p
    return out

@njit(cache=True,parallel=True)
def ts_mean_kernel(x,n,min_periods):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            s = 0.0
            c = 0
            for t in range(start,i+1):
                v = x[t,j]
                if np.isfinite(v):
                    s += v
                    c += 1
            if c>=min_periods:
                out[i,j] = s/c
    return out

@njit(cache=True,parallel=True)
def ts_var_kernel(x,n,min_periods):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            s = 0.0
            ss = 0.0
            c = 0
            for t in range(start,i+1):
                v = x[t,j]
                if np.isfinite(v):
                    s += v
                    ss += v*v
                    c += 1
            if c>=min_periods and c>1:
                out[i,j] = (ss-s*s/c)/(c-1)
    return out

@njit(cache=True,parallel=True)
def ts_ir_kernel(x,n,min_periods):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            s = 0.0
            ss = 0.0
            c = 0
            for t in range(start,i+1):
                v = x[t,j]
                if np.isfinite(v):
                    s += v
                    ss += v*v
                    c += 1
            if c>=min_periods and c>1:
                mean = s/c
                var = (ss-s*s/c)/(c-1)
                if var<=0.0:
                    continue
                std = np.sqrt(var)
                if std<1e-10:
                    continue
                out[i,j] = mean/std
    return out

@njit(cache=True,parallel=True)
def ts_skew_kernel(x,n,min_periods):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            mean = 0.0
            m2 = 0.0
            m3 = 0.0
            c = 0
            for t in range(start,i+1):
                v = x[t,j]
                if np.isfinite(v):
                    c += 1
                    d = v-mean
                    dn = d/c
                    term = d*dn*(c-1)
                    m3 += term*dn*(c-2)-3*dn*m2
                    m2 += term
                    mean += dn
            if c<min_periods or c<=2 or m2==0.0:
                continue
            z2 = m2/c
            z3 = m3/c
            g = z3/(z2**1.5)
            out[i,j] = np.sqrt(c*(c-1))/(c-2)*g
    return out

@njit(cache=True,parallel=True)
def ts_kur_kernel(x,n,min_periods):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            mean = 0.0
            m2 = 0.0
            m3 = 0.0
            m4 = 0.0
            c = 0
            for t in range(start,i+1):
                v = x[t,j]
                if np.isfinite(v):
                    c += 1
                    d = v-mean
                    dn = d/c
                    dn2 = dn*dn
                    term = d*dn*(c-1)
                    m4 += term*dn2*(c*c-3*c+3)+6*dn2*m2-4*dn*m3
                    m3 += term*dn*(c-2)-3*dn*m2
                    m2 += term
                    mean += dn
            if c>=min_periods and c>3 and m2!=0.0:
                z2 = m2/c
                z4 = m4/c
                g = z4/(z2*z2)-3.0
                out[i,j] = ((c-1)/((c-2)*(c-3)))*((c+1)*g+6.0)
    return out

@njit(cache=True,parallel=True)
def ts_min_kernel(x,n,min_periods):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            z = np.inf
            c = 0
            for t in range(start,i+1):
                v = x[t,j]
                if np.isfinite(v):
                    if v<z:
                        z = v
                    c += 1
            if c>=min_periods:
                out[i,j] = z
    return out

@njit(cache=True,parallel=True)
def ts_max_kernel(x,n,min_periods):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            z = -np.inf
            c = 0
            for t in range(start,i+1):
                v = x[t,j]
                if np.isfinite(v):
                    if v>z:
                        z = v
                    c += 1
            if c>=min_periods:
                out[i,j] = z
    return out

@njit(cache=True,parallel=True)
def ts_argmin_kernel(x,n,min_periods):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            c = 0
            z = np.inf
            d = 0
            for t in range(i,start-1,-1):
                v = x[t,j]
                if np.isfinite(v):
                    if v<z:
                        z = v
                        d = i-t
                    c += 1
            if c>=min_periods:
                out[i,j] = d+1
    return out

@njit(cache=True,parallel=True)
def ts_argmax_kernel(x,n,min_periods):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            c = 0
            z = -np.inf
            d = 0
            for t in range(i,start-1,-1):
                v = x[t,j]
                if np.isfinite(v):
                    if v>z:
                        z = v
                        d = i-t
                    c += 1
            if c>=min_periods:
                out[i,j] = d+1
    return out

@njit(cache=True,parallel=True)
def ts_median_kernel(x,n,min_periods):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        tmp = np.empty(n,dtype=np.float64)
        idx = np.empty(n,dtype=np.int32)
        sl = np.empty(n,dtype=np.int32)
        sh = np.empty(n,dtype=np.int32)
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            c = 0
            for t in range(start,i+1):
                v = x[t,j]
                if np.isfinite(v):
                    tmp[c] = v
                    idx[c] = c
                    c += 1
            if c>=min_periods:
                _quicksort_pair_ws(tmp,idx,0,c-1,sl,sh)
                if c%2==1:
                    out[i,j] = tmp[c//2]
                else:
                    out[i,j] = 0.5*(tmp[c//2-1]+tmp[c//2])
    return out

@njit(cache=True,parallel=True)
def ts_zscore_kernel(x,n,min_periods):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        s = 0.0
        ss = 0.0
        c = 0
        for i in range(nd):
            v = x[i,j]
            if np.isfinite(v):
                s += v
                ss += v*v
                c += 1
            if i>=n:
                z = x[i-n,j]
                if np.isfinite(z):
                    s -= z
                    ss -= z*z
                    c -= 1
            if c>=min_periods and c>=2 and np.isfinite(v):
                mean = s/c
                var = (ss-s*s/c)/(c-1)
                if var>0.0:
                    std = np.sqrt(var)
                    if std>1e-10:
                        out[i,j] = (v-mean)/std
    return out

@njit(cache=True,parallel=True)
def ema_kernel(x,alpha,min_periods):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    beta = 1-alpha
    for j in prange(ns):
        s = 0.0
        w = 0.0
        c = 0
        for i in range(nd):
            v = x[i,j]
            s *= beta
            w *= beta
            if np.isfinite(v):
                s += v
                w += 1.0
                c += 1
            if c>=min_periods and w>=0.0:
                out[i,j] = s/w
    return out

@njit(cache=True,parallel=True)
def wma_kernel(x,w):
    nd,ns = x.shape
    n = w.shape[0]
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(n-1,nd):
            if np.isfinite(x[i,j]):
                s = 0.0
                sw = 0.0
                for t in range(i+1-n,i+1):
                    v = x[t,j]
                    if np.isfinite(v):
                        z = w[t+n-i-1]
                        s += v*z
                        sw += z
                if sw!=0.0:
                    out[i,j] = s/sw
    return out

"""
condition
"""
@njit(cache=True,parallel=True)
def ts_cut_kernel(x,y,n,hq,min_periods):
    nd,ns = x.shape
    high = np.full((nd,ns),np.nan)
    low = np.full((nd,ns),np.nan)
    for j in prange(ns):
        vals = np.empty(n,np.float64)
        keys = np.empty(n,np.float64)
        idx = np.empty(n,np.int64)
        stack_lo = np.empty(n,np.int64)
        stack_hi = np.empty(n,np.int64)
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            count = 0
            for t in range(start,i+1):
                v = x[t,j]
                z = y[t,j]
                if np.isfinite(v) and np.isfinite(z):
                    vals[count] = v
                    keys[count] = z
                    idx[count] = count
                    count += 1
            if count<min_periods:
                continue
            high_count = int(np.floor(count*hq))
            low_count = count-high_count
            if high_count<1 or low_count<1:
                continue
            _quicksort_pair_ws(keys,idx,0,count-1,stack_lo,stack_hi)
            high_sum = 0.0
            low_sum = 0.0
            for k in range(count-high_count,count):
                high_sum += vals[idx[k]]
            for k in range(low_count):
                low_sum += vals[idx[k]]
            high[i,j] = high_sum/high_count
            low[i,j] = low_sum/low_count
    return high,low

@njit(cache=True)
def _adjust_rank_write(vals,idxs,cnt,con,out,row,d):
    if cnt==0:
        return
    if cnt==1:
        j = idxs[0]
        c = con[row,j]
        if c==1:
            out[row,j] = 0.5*(1.0+d)
        elif c==0:
            out[row,j] = 0.5*(1.0-d)
        return
    _sort_pair(vals,idxs,cnt)
    uniq = 1
    prev = vals[0]
    for k in range(1,cnt):
        cur = vals[k]
        if cur!=prev:
            uniq += 1
            prev = cur
    if uniq==1:
        for k in range(cnt):
            j = idxs[k]
            c = con[row,j]
            if c==1:
                out[row,j] = 0.5*(1.0+d)
            elif c==0:
                out[row,j] = 0.5*(1.0-d)
        return
    denom = uniq-1
    r = 0
    prev = vals[0]
    for k in range(cnt):
        cur = vals[k]
        if k>0 and cur!=prev:
            r += 1
            prev = cur
        j = idxs[k]
        c = con[row,j]
        z = r/denom
        if c==1:
            out[row,j] = z*(1.0+d)
        elif c==0:
            out[row,j] = z*(1.0-d)

@njit(cache=True,parallel=True)
def adjust_by_kernel(x,con,d):
    nd,ns = x.shape
    out = np.empty((nd,ns),np.float64)
    for i in prange(nd):
        vals = np.empty(ns,np.float64)
        idxs = np.empty(ns,np.int32)
        cnt = 0
        for j in range(ns):
            out[i,j] = np.nan
            v = x[i,j]
            if np.isfinite(v):
                vals[cnt] = v
                idxs[cnt] = j
                cnt += 1
        _adjust_rank_write(vals,idxs,cnt,con,out,i,d)
    return out

@njit(cache=True,parallel=True)
def reverse_by_kernel(x,con):
    nd,ns = x.shape
    out = np.empty((nd,ns),np.float64)
    for i in prange(nd):
        for j in range(ns):
            c = con[i,j]
            v = x[i,j]
            if c==1:
                out[i,j] = -v
            elif c==0:
                out[i,j] = v
            else:
                out[i,j] = np.nan
    return out

@njit(cache=True,parallel=True)
def reverse_rank_by_kernel(x,con):
    nd,ns = x.shape
    out = np.empty((nd,ns),np.float64)
    for i in prange(nd):
        pos_vals = np.empty(ns,np.float64)
        pos_idxs = np.empty(ns,np.int32)
        neg_vals = np.empty(ns,np.float64)
        neg_idxs = np.empty(ns,np.int32)
        pcnt = 0
        ncnt = 0
        for j in range(ns):
            c = con[i,j]
            v = x[i,j]
            if c==-1:
                out[i,j] = np.nan
            else:
                out[i,j] = 0.0
            if c==1 and np.isfinite(v):
                pos_vals[pcnt] = v
                pos_idxs[pcnt] = j
                pcnt += 1
            elif c==0 and np.isfinite(v):
                neg_vals[ncnt] = v
                neg_idxs[ncnt] = j
                ncnt += 1
        _rank_write(pos_vals,pos_idxs,pcnt,out,i,True)
        _rank_write(neg_vals,neg_idxs,ncnt,out,i,False)
    return out

@njit(cache=True,parallel=True)
def trade_when_kernel(x,con,q):
    nd,ns = x.shape
    rank = np.empty((nd,ns),np.float64)
    for i in prange(nd):
        vals = np.empty(ns,np.float64)
        idxs = np.empty(ns,np.int32)
        cnt = 0
        for j in range(ns):
            rank[i,j] = np.nan
            v = x[i,j]
            if np.isfinite(v):
                vals[cnt] = v
                idxs[cnt] = j
                cnt += 1
        _rank_write(vals,idxs,cnt,rank,i,True)
    out = np.empty((nd,ns),np.float64)
    for j in prange(ns):
        last = np.nan
        for i in range(nd):
            c = con[i,j]
            v = rank[i,j]
            if c==-1 or not np.isfinite(v):
                out[i,j] = np.nan
                continue
            if c==1 or not np.isfinite(last):
                last = v
            else:
                last = (1.0-q)*last+q*v
            out[i,j] = last
    return out

"""
cross
"""
@njit(cache=True,parallel=True)
def addr_kernel(x,y):
    nd,ns = x.shape
    out = np.empty((nd,ns),np.float64)
    for i in prange(nd):
        rx = np.empty(ns,np.float64)
        xvals = np.empty(ns,np.float64)
        xidxs = np.empty(ns,np.int32)
        yvals = np.empty(ns,np.float64)
        yidxs = np.empty(ns,np.int32)
        xcnt = 0
        ycnt = 0
        for j in range(ns):
            rx[j] = np.nan
            out[i,j] = np.nan
            vx = x[i,j]
            if np.isfinite(vx):
                xvals[xcnt] = vx
                xidxs[xcnt] = j
                xcnt += 1
            vy = y[i,j]
            if np.isfinite(vy):
                yvals[ycnt] = vy
                yidxs[ycnt] = j
                ycnt += 1
        _rank_to_1d(xvals,xidxs,xcnt,rx,True)
        _rank_write(yvals,yidxs,ycnt,out,i,True)
        for j in range(ns):
            a = rx[j]
            b = out[i,j]
            if np.isfinite(a) and np.isfinite(b):
                out[i,j] = a+b
            else:
                out[i,j] = np.nan
    return out


@njit(cache=True,parallel=True)
def subr_kernel(x,y):
    nd,ns = x.shape
    out = np.empty((nd,ns),np.float64)
    for i in prange(nd):
        rx = np.empty(ns,np.float64)
        xvals = np.empty(ns,np.float64)
        xidxs = np.empty(ns,np.int32)
        yvals = np.empty(ns,np.float64)
        yidxs = np.empty(ns,np.int32)
        xcnt = 0
        ycnt = 0
        for j in range(ns):
            rx[j] = np.nan
            out[i,j] = np.nan
            vx = x[i,j]
            if np.isfinite(vx):
                xvals[xcnt] = vx
                xidxs[xcnt] = j
                xcnt += 1
            vy = y[i,j]
            if np.isfinite(vy):
                yvals[ycnt] = vy
                yidxs[ycnt] = j
                ycnt += 1
        _rank_to_1d(xvals,xidxs,xcnt,rx,True)
        _rank_write(yvals,yidxs,ycnt,out,i,True)
        for j in range(ns):
            a = rx[j]
            b = out[i,j]
            if np.isfinite(a) and np.isfinite(b):
                out[i,j] = a-b
            else:
                out[i,j] = np.nan
    return out

@njit(cache=True,parallel=True)
def mulr_kernel(x,y):
    nd,ns = x.shape
    out = np.empty((nd,ns),np.float64)
    for i in prange(nd):
        rx = np.empty(ns,np.float64)
        xvals = np.empty(ns,np.float64)
        xidxs = np.empty(ns,np.int32)
        yvals = np.empty(ns,np.float64)
        yidxs = np.empty(ns,np.int32)
        xcnt = 0
        ycnt = 0
        for j in range(ns):
            rx[j] = np.nan
            out[i,j] = np.nan
            vx = x[i,j]
            if np.isfinite(vx):
                xvals[xcnt] = vx
                xidxs[xcnt] = j
                xcnt += 1
            vy = y[i,j]
            if np.isfinite(vy):
                yvals[ycnt] = vy
                yidxs[ycnt] = j
                ycnt += 1
        _rank_to_1d(xvals,xidxs,xcnt,rx,True)
        _rank_write(yvals,yidxs,ycnt,out,i,True)
        for j in range(ns):
            a = rx[j]
            b = out[i,j]
            if np.isfinite(a) and np.isfinite(b):
                out[i,j] = a*b
            else:
                out[i,j] = np.nan
    return out


@njit(cache=True,parallel=True)
def mulz_kernel(x,y):
    nd,ns = x.shape
    out = np.empty((nd,ns),np.float64)
    for i in prange(nd):
        sx = 0.0
        ssx = 0.0
        cx = 0
        sy = 0.0
        ssy = 0.0
        cy = 0
        for j in range(ns):
            vx = x[i,j]
            if np.isfinite(vx):
                sx += vx
                ssx += vx*vx
                cx += 1
            vy = y[i,j]
            if np.isfinite(vy):
                sy += vy
                ssy += vy*vy
                cy += 1
        if cx<=1 or cy<=1:
            for j in range(ns):
                out[i,j] = np.nan
            continue
        mx = sx/cx
        my = sy/cy
        varx = (ssx-sx*sx/cx)/(cx-1)
        vary = (ssy-sy*sy/cy)/(cy-1)
        if varx<=0.0 or vary<=0.0:
            for j in range(ns):
                out[i,j] = np.nan
            continue
        stdx = np.sqrt(varx)
        stdy = np.sqrt(vary)
        for j in range(ns):
            vx = x[i,j]
            vy = y[i,j]
            if np.isfinite(vx) and np.isfinite(vy):
                out[i,j] = ((vx-mx)/stdx)*((vy-my)/stdy)
            else:
                out[i,j] = np.nan
    return out

@njit(cache=True,parallel=True)
def div_kernel(x,y):
    nd,ns = x.shape
    out = np.empty((nd,ns),np.float64)
    for i in prange(nd):
        for j in range(ns):
            a = x[i,j]
            b = y[i,j]
            if np.isfinite(a) and np.isfinite(b) and b!=0.0:
                z = a/b
                if np.isfinite(z):
                    out[i,j] = z
                else:
                    out[i,j] = np.nan
            else:
                out[i,j] = np.nan
    return out

@njit(cache=True,parallel=True)
def maximum_kernel(x:np.ndarray,y:np.ndarray)->np.ndarray:
    out = np.empty_like(x,dtype=np.float64)
    for i in prange(x.shape[0]):
        for j in range(x.shape[1]):
            xv = x[i,j]
            yv = y[i,j]
            if np.isnan(xv) and np.isnan(yv):
                out[i,j] = np.nan
            elif np.isnan(xv):
                out[i,j] = yv
            elif np.isnan(yv):
                out[i,j] = xv
            elif xv>=yv:
                out[i,j] = xv
            else:
                out[i,j] = yv
    return out

@njit(cache=True,parallel=True)
def maximumr_kernel(x:np.ndarray,y:np.ndarray)->np.ndarray:
    nd,ns = x.shape
    out = np.empty((nd,ns),np.float64)
    for i in prange(nd):
        rx = np.empty(ns,np.float64)
        xvals = np.empty(ns,np.float64)
        xidxs = np.empty(ns,np.int32)
        yvals = np.empty(ns,np.float64)
        yidxs = np.empty(ns,np.int32)
        xcnt = 0
        ycnt = 0
        for j in range(ns):
            rx[j] = np.nan
            out[i,j] = np.nan
            vx = x[i,j]
            if np.isfinite(vx):
                xvals[xcnt] = vx
                xidxs[xcnt] = j
                xcnt += 1
            vy = y[i,j]
            if np.isfinite(vy):
                yvals[ycnt] = vy
                yidxs[ycnt] = j
                ycnt += 1
        _rank_to_1d(xvals,xidxs,xcnt,rx,True)
        _rank_write(yvals,yidxs,ycnt,out,i,True)
        for j in range(ns):
            a = rx[j]
            b = out[i,j]
            if np.isnan(a) and np.isnan(b):
                out[i,j] = np.nan
            elif np.isnan(a):
                out[i,j] = b
            elif np.isnan(b):
                out[i,j] = a
            elif a>=b:
                out[i,j] = a
            else:
                out[i,j] = b
    return out

@njit(cache=True,parallel=True)
def minimum_kernel(x:np.ndarray,y:np.ndarray)->np.ndarray:
    out = np.empty_like(x,dtype=np.float64)
    for i in prange(x.shape[0]):
        for j in range(x.shape[1]):
            xv = x[i,j]
            yv = y[i,j]
            if np.isnan(xv) and np.isnan(yv):
                out[i,j] = np.nan
            elif np.isnan(xv):
                out[i,j] = yv
            elif np.isnan(yv):
                out[i,j] = xv
            elif xv<=yv:
                out[i,j] = xv
            else:
                out[i,j] = yv
    return out

@njit(cache=True,parallel=True)
def minimumr_kernel(x:np.ndarray,y:np.ndarray)->np.ndarray:
    nd,ns = x.shape
    out = np.empty((nd,ns),np.float64)
    for i in prange(nd):
        rx = np.empty(ns,np.float64)
        xvals = np.empty(ns,np.float64)
        xidxs = np.empty(ns,np.int32)
        yvals = np.empty(ns,np.float64)
        yidxs = np.empty(ns,np.int32)
        xcnt = 0
        ycnt = 0
        for j in range(ns):
            rx[j] = np.nan
            out[i,j] = np.nan
            vx = x[i,j]
            if np.isfinite(vx):
                xvals[xcnt] = vx
                xidxs[xcnt] = j
                xcnt += 1
            vy = y[i,j]
            if np.isfinite(vy):
                yvals[ycnt] = vy
                yidxs[ycnt] = j
                ycnt += 1
        _rank_to_1d(xvals,xidxs,xcnt,rx,True)
        _rank_write(yvals,yidxs,ycnt,out,i,True)
        for j in range(ns):
            a = rx[j]
            b = out[i,j]
            if np.isnan(a) and np.isnan(b):
                out[i,j] = np.nan
            elif np.isnan(a):
                out[i,j] = b
            elif np.isnan(b):
                out[i,j] = a
            elif a<=b:
                out[i,j] = a
            else:
                out[i,j] = b
    return out

@njit(cache=True,parallel=True)
def fd_sum_kernel(x,y,w1,w2,w3,w4):
    nd,ns = x.shape
    out = np.empty((nd,ns),dtype=np.float64)
    for i in prange(nd):
        for j in range(ns):
            a = x[i,j]
            b = y[i,j]
            if not (np.isfinite(a) and np.isfinite(b)):
                out[i,j] = 0.0
                continue
            if a>0.0 and b>0.0:
                w = w1
            elif a<0.0 and b<0.0:
                w = w2
            elif a>0.0 and b<0.0:
                w = w3
            elif a<0.0 and b>0.0:
                w = w4
            else:
                w = 0.0
            out[i,j] = np.abs(a)*np.abs(b)*w
    return out

@njit(cache=True,parallel=True)
def ts_corr_kernel(x,y,n,min_periods,hold_nan):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        c = 0
        for i in range(nd):
            a = x[i,j]
            b = y[i,j]
            if np.isfinite(a) and np.isfinite(b):
                sx += a
                sy += b
                sxx += a*a
                syy += b*b
                sxy += a*b
                c += 1
            if i>=n:
                a0 = x[i-n,j]
                b0 = y[i-n,j]
                if np.isfinite(a0) and np.isfinite(b0):
                    sx -= a0
                    sy -= b0
                    sxx -= a0*a0
                    syy -= b0*b0
                    sxy -= a0*b0
                    c -= 1
            if c>=min_periods:
                if (not hold_nan) or (np.isfinite(a) and np.isfinite(b)):
                    cov = (sxy-sx*sy/c)/(c-1)
                    vx = (sxx-sx*sx/c)/(c-1)
                    vy = (syy-sy*sy/c)/(c-1)
                    if vx>0.0 and vy>0.0:
                        den = np.sqrt(vx*vy)
                        if den>1e-10:
                            out[i,j] = cov/den
    return out

@njit(cache=True,parallel=True)
def ts_reg_kernel(x,y,n,rettype,min_periods):
    nd,ns = y.shape
    out = np.full((nd,ns),np.nan)
    for j in prange(ns):
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        sxy = 0.0
        syy = 0.0
        c = 0
        for i in range(nd):
            a = x[i,j]
            b = y[i,j]
            if np.isfinite(a) and np.isfinite(b):
                sx += a
                sy += b
                sxx += a*a
                sxy += a*b
                syy += b*b
                c += 1
            if i>=n:
                a0 = x[i-n,j]
                b0 = y[i-n,j]
                if np.isfinite(a0) and np.isfinite(b0):
                    sx -= a0
                    sy -= b0
                    sxx -= a0*a0
                    sxy -= a0*b0
                    syy -= b0*b0
                    c -= 1
            if c>=min_periods:
                mx = sx/c
                my = sy/c
                den = sxx-sx*sx/c
                if den!=0.0:
                    beta = (sxy-sx*sy/c)/den
                    alpha = my-beta*mx
                    if rettype==1:
                        out[i,j] = beta
                    elif rettype==2:
                        out[i,j] = alpha
                    elif rettype==3:
                        vy = syy-sy*sy/c
                        if vy>0.0:
                            cov = sxy-sx*sy/c
                            out[i,j] = cov*cov/(den*vy)
                    else:
                        if np.isfinite(a) and np.isfinite(b):
                            out[i,j] = b-(alpha+beta*a)
    return out

@njit(cache=True,parallel=True)
def ts_multireg_kernel(x,y,n,rettype,min_periods):
    nd,ns = y.shape
    k = x.shape[2]
    p = k+1
    out = np.full((nd,ns),np.nan,dtype=np.float64)
    for j in prange(ns):
        xtx = np.zeros((p,p),dtype=np.float64)
        xty = np.zeros(p,dtype=np.float64)
        c = 0
        for i in range(nd):
            ok = np.isfinite(y[i,j])
            if ok:
                for a in range(k):
                    if not np.isfinite(x[i,j,a]):
                        ok = False
                        break
            if ok:
                yy = y[i,j]
                xtx[0,0] += 1.0
                xty[0] += yy
                for a in range(k):
                    xa = x[i,j,a]
                    xtx[0,a+1] += xa
                    xtx[a+1,0] += xa
                    xty[a+1] += xa*yy
                    for b in range(k):
                        xtx[a+1,b+1] += xa*x[i,j,b]
                c += 1
            if i>=n:
                ok0 = np.isfinite(y[i-n,j])
                if ok0:
                    for a in range(k):
                        if not np.isfinite(x[i-n,j,a]):
                            ok0 = False
                            break
                if ok0:
                    yy0 = y[i-n,j]
                    xtx[0,0] -= 1.0
                    xty[0] -= yy0
                    for a in range(k):
                        xa0 = x[i-n,j,a]
                        xtx[0,a+1] -= xa0
                        xtx[a+1,0] -= xa0
                        xty[a+1] -= xa0*yy0
                        for b in range(k):
                            xtx[a+1,b+1] -= xa0*x[i-n,j,b]
                    c -= 1
            if c>=min_periods and c>=p:
                for d in range(p):
                    xtx[d,d] += 1e-10
                beta = np.linalg.solve(xtx,xty)
                for d in range(p):
                    xtx[d,d] -= 1e-10
                if rettype==3:
                    sse = 0.0
                    sy = 0.0
                    syy = 0.0
                    start = max(0,i-n+1)
                    for t in range(start,i+1):
                        ok2 = np.isfinite(y[t,j])
                        if ok2:
                            for a in range(k):
                                if not np.isfinite(x[t,j,a]):
                                    ok2 = False
                                    break
                        if ok2:
                            yy = y[t,j]
                            sy += yy
                            syy += yy*yy
                            yh = beta[0]
                            for a in range(k):
                                yh += x[t,j,a]*beta[a+1]
                            e = yy-yh
                            sse += e*e
                    vy = syy-sy*sy/c
                    if vy>0.0:
                        out[i,j] = 1.0-sse/vy
                else:
                    ok3 = np.isfinite(y[i,j])
                    if ok3:
                        for a in range(k):
                            if not np.isfinite(x[i,j,a]):
                                ok3 = False
                                break
                    if ok3:
                        yh = beta[0]
                        for a in range(k):
                            yh += x[i,j,a]*beta[a+1]
                        out[i,j] = y[i,j]-yh
    return out

@njit(cache=True,parallel=True)
def cs_corr_kernel(x,y):
    nd,ns = x.shape
    out = np.full(nd,np.nan,np.float64)
    for i in prange(nd):
        c = 0
        sx = 0.0
        sy = 0.0
        for j in range(ns):
            a = x[i,j]
            b = y[i,j]
            if np.isfinite(a) and np.isfinite(b):
                c += 1
                sx += a
                sy += b
        if c<2:
            continue
        mx = sx/c
        my = sy/c
        sxx = 0.0
        syy = 0.0
        sxy = 0.0
        for j in range(ns):
            a = x[i,j]
            b = y[i,j]
            if np.isfinite(a) and np.isfinite(b):
                dx = a-mx
                dy = b-my
                sxx += dx*dx
                syy += dy*dy
                sxy += dx*dy
        if sxx>0.0 and syy>0.0:
            out[i] = sxy/np.sqrt(sxx*syy)
    return out

@njit(parallel=True,cache=True)
def cs_reg_kernel(x,y,rettype,min_cs):
    nd,ns = y.shape
    out = np.full((nd,ns),np.nan,dtype=np.float64)
    for i in prange(nd):
        sx = 0.0
        sy = 0.0
        cnt = 0
        for j in range(ns):
            vx = x[i,j]
            vy = y[i,j]
            if not np.isnan(vx) and not np.isnan(vy):
                sx += vx
                sy += vy
                cnt += 1
        if cnt<min_cs:
            continue
        mx = sx/cnt
        my = sy/cnt
        sxx = 0.0
        sxy = 0.0
        for j in range(ns):
            vx = x[i,j]
            vy = y[i,j]
            if not np.isnan(vx) and not np.isnan(vy):
                dx = vx-mx
                dy = vy-my
                sxx += dx*dx
                sxy += dx*dy
        if sxx==0.0:
            continue
        beta = sxy/sxx
        alpha = my-beta*mx
        for j in range(ns):
            vx = x[i,j]
            vy = y[i,j]
            if not np.isnan(vx) and not np.isnan(vy):
                if rettype==0:
                    out[i,j] = vy-alpha-beta*vx
                elif rettype==1:
                    out[i,j] = alpha+beta*vx
                elif rettype==2:
                    out[i,j] = beta
                elif rettype==3:
                    out[i,j] = alpha
    return out

@njit(cache=True,parallel=True)
def cs_multireg_kernel(x,y,min_cs):
    nd,ns = y.shape
    k = x.shape[2]
    p = k+1
    out = np.full((nd,ns),np.nan)
    for i in prange(nd):
        c = 0
        idx = np.empty(ns,dtype=np.int32)
        for j in range(ns):
            ok = np.isfinite(y[i,j])
            if ok:
                for a in range(k):
                    if not np.isfinite(x[i,j,a]):
                        ok = False
                        break
            if ok:
                idx[c] = j
                c += 1
        if c>=min_cs and c>=p:
            xtx = np.zeros((p,p),dtype=np.float64)
            xty = np.zeros(p,dtype=np.float64)
            for t in range(c):
                j = idx[t]
                yy = y[i,j]
                xtx[0,0] += 1.0
                xty[0] += yy
                for a in range(k):
                    xa = x[i,j,a]
                    xtx[0,a+1] += xa
                    xtx[a+1,0] += xa
                    xty[a+1] += xa*yy
                    for b in range(k):
                        xtx[a+1,b+1] += xa*x[i,j,b]
            for d in range(p):
                xtx[d,d] += 1e-10
            beta = np.linalg.solve(xtx,xty)
            for t in range(c):
                j = idx[t]
                yh = beta[0]
                for a in range(k):
                    yh += x[i,j,a]*beta[a+1]
                out[i,j] = y[i,j]-yh
    return out

@njit(cache=True,parallel=True)
def if_else_kernel(a,b,con):
    nd,ns = a.shape
    out = np.empty((nd,ns),np.float64)
    for i in prange(nd):
        ra = np.empty(ns,np.float64)
        avals = np.empty(ns,np.float64)
        aidxs = np.empty(ns,np.int32)
        bvals = np.empty(ns,np.float64)
        bidxs = np.empty(ns,np.int32)
        acnt = 0
        bcnt = 0
        for j in range(ns):
            ra[j] = np.nan
            out[i,j] = np.nan
            va = a[i,j]
            if np.isfinite(va):
                avals[acnt] = va
                aidxs[acnt] = j
                acnt += 1
            vb = b[i,j]
            if np.isfinite(vb):
                bvals[bcnt] = vb
                bidxs[bcnt] = j
                bcnt += 1
        _rank_to_1d(avals,aidxs,acnt,ra,True)
        _rank_write(bvals,bidxs,bcnt,out,i,True)
        for j in range(ns):
            c = con[i,j]
            if c==1:
                out[i,j] = ra[j]
            elif c==0:
                pass
            else:
                out[i,j] = np.nan
    return out

"""
other funcs
"""
@njit(cache=True)
def mean_kernel(x):
    s = 0.0
    c = 0
    n = x.shape[0]
    for i in range(n):
        v = x[i]
        if np.isfinite(v):
            s += v
            c += 1
    if c==0:
        return np.nan
    return s/c

@njit(cache=True)
def std_kernel(x):
    s = 0.0
    ss = 0.0
    c = 0
    n = x.shape[0]
    for i in range(n):
        v = x[i]
        if np.isfinite(v):
            s += v
            ss += v*v
            c += 1
    if c<=1:
        return np.nan
    var = (ss-s*s/c)/(c-1)
    if var<=0.0:
        return 0.0
    return np.sqrt(var)

@njit(cache=True)
def ir_kernel(x):
    s = 0.0
    ss = 0.0
    c = 0
    n = x.shape[0]
    for i in range(n):
        v = x[i]
        if np.isfinite(v):
            s += v
            ss += v*v
            c += 1
    if c<=1:
        return np.nan
    mean = s/c
    var = (ss-s*s/c)/(c-1)
    if var<=0.0:
        return np.nan
    std = np.sqrt(var)
    if std<1e-10:
        return np.nan
    return mean/std

@njit(cache=True,parallel=True)
def mask_kernel(x,y):
    nd,ns = x.shape
    out = np.full((nd,ns),np.nan)
    for i in prange(nd):
        for j in range(ns):
            if y[i,j]==1:
                out[i,j] = x[i,j]
    return out

@njit(cache=True)
def p2p_corr_kernel(x,y):
    sx = 0.0
    sy = 0.0
    cnt = 0
    for i in range(x.shape[0]):
        vx = x[i]
        vy = y[i]
        if not np.isnan(vx) and not np.isnan(vy):
            sx += vx
            sy += vy
            cnt += 1
    if cnt<2:
        return np.nan
    mx = sx/cnt
    my = sy/cnt
    cov = 0.0
    xvar = 0.0
    yvar = 0.0
    for i in range(x.shape[0]):
        vx = x[i]
        vy = y[i]
        if not np.isnan(vx) and not np.isnan(vy):
            dx = vx-mx
            dy = vy-my
            cov += dx*dy
            xvar += dx*dx
            yvar += dy*dy
    if xvar==0.0 or yvar==0.0:
        return np.nan
    return cov/np.sqrt(xvar*yvar)
