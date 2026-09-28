import os
os.environ["NUMBA_NUM_THREADS"] = "12"
import numpy as np
from numba import njit,prange
from .minute_tools import MinuteFactorEngine

@njit(cache=True,parallel=True)
def ts_rank_2d(values,n,min_periods,pct):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            val = values[i,j]
            if np.isfinite(val):
                start = max(0,i+1-n)
                sum_count = 0
                smaller_count = 0
                bigger_count = 0
                for t in range(start,i):
                    val_t = values[t,j]
                    if np.isfinite(val_t):
                        if val_t<val:
                            smaller_count+=1
                        elif val_t>val:
                            bigger_count+=1
                        sum_count+=1
                sum_count+=1
                if sum_count<min_periods:
                    continue
                rank = (sum_count+smaller_count-bigger_count+1)/2
                if pct:
                    rank = rank/sum_count
                out[i,j]=rank
    return out

@njit(cache=True,parallel=True)
def ts_decay_linear_2d(values,n):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(nd):
            val = values[i,j]
            if np.isfinite(val):
                num=0.0
                den=0.0
                for t in range(n):
                    idx = i-t
                    w = n-t
                    if idx<0:
                        break
                    val_idx = values[idx,j]
                    if np.isfinite(val_idx):
                        num += val_idx*w
                        den += w
                out[i,j] = num/den
    return out

@njit(cache=True,parallel=True)
def ts_sum_2d(values,n,min_periods):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            ts_sum = 0
            count = 0
            for t in range(start,i+1):
                val = values[t,j]
                if np.isfinite(val):
                    ts_sum += val
                    count += 1
            if count<min_periods:
                continue
            out[i,j] = ts_sum
    return out

@njit(cache=True,parallel=True)
def ts_prod_2d(values,n,min_periods):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            ts_prod = 1.0
            count = 0
            for t in range(start,i+1):
                val = values[t,j]
                if np.isfinite(val):
                    ts_prod *= val
                    count += 1
            if count<min_periods:
                continue
            out[i,j] = ts_prod
    return out


@njit(cache=True,parallel=True)
def ts_mean_2d(values,n,min_periods):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            ts_sum = 0
            den = 0
            for t in range(start,i+1):
                val = values[t,j]
                if np.isfinite(val):
                    ts_sum += val
                    den += 1
            if den<min_periods:
                continue
            out[i,j] = ts_sum/den
    return out

@njit(cache=True,parallel=True)
def ts_var_2d(values,n,min_periods):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            ts_sum = 0
            square_sum = 0
            den = 0
            for t in range(start,i+1):
                val = values[t,j]
                if np.isfinite(val):
                    ts_sum += val
                    square_sum += val**2
                    den += 1
            if (den<min_periods) or (den<=1):
                continue
            out[i,j] = (square_sum-(ts_sum**2)/den)/(den-1)
    return out

@njit(cache=True,parallel=True)
def ts_ir_2d(values,n,min_periods):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            ts_sum = 0
            square_sum = 0
            den = 0
            for t in range(start,i+1):
                val = values[t,j]
                if np.isfinite(val):
                    ts_sum += val
                    square_sum += val**2
                    den += 1
            if (den<min_periods) or (den<=1):
                continue
            ts_mean = ts_sum/den
            ts_var = (square_sum-(ts_sum**2)/den)/(den-1)
            if ts_var<=0.0:
                continue
            ts_std = np.sqrt(ts_var)
            if ts_std<1e-10:
                continue
            out[i,j] = ts_mean/ts_std
    return out

@njit(cache=True,parallel=True)
def ts_skew_2d(values,n,min_periods):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start=max(0,i+1-n)
            mean=0.0
            M2=0.0
            M3=0.0
            count=0
            for t in range(start,i+1):
                v=values[t,j]
                if np.isfinite(v):
                    count+=1
                    delta=v-mean
                    delta_n=delta/count
                    term1=delta*delta_n*(count-1)
                    M3+=term1*delta_n*(count-2)-3*delta_n*M2
                    M2+=term1
                    mean+=delta_n
            if count<min_periods or count<=2 or M2==0.0:
                continue
            m2=M2/count
            m3=M3/count
            g1=m3/(m2**1.5)  
            skew=np.sqrt(count*(count-1))/(count-2)*g1  
            out[i,j]=skew
    return out

@njit(cache=True,parallel=True)
def ts_kur_2d(values,n,min_periods):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start=max(0,i+1-n)
            mean=0.0
            M2=0.0
            M3=0.0
            M4=0.0
            count=0
            for t in range(start,i+1):
                v=values[t,j]
                if np.isfinite(v):
                    count+=1
                    delta=v-mean
                    delta_n=delta/count
                    delta_n2=delta_n*delta_n
                    term1=delta*delta_n*(count-1)
                    M4+=term1*delta_n2*(count*count-3*count+3)+6*delta_n2*M2-4*delta_n*M3
                    M3+=term1*delta_n*(count-2)-3*delta_n*M2
                    M2+=term1
                    mean+=delta_n
            if (count>=min_periods) and (count>3) and (M2!=0.0):
                m2=M2/count
                m4=M4/count
                g2=m4/(m2*m2)-3.0
                kur=((count-1)/((count-2)*(count-3)))*((count+1)*g2+6.0)
                out[i,j]=kur
    return out

@njit(cache=True,parallel=True)
def ts_min_2d(values,n,min_periods):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            ts_min = np.inf
            count = 0
            for t in range(start,i+1):
                val = values[t,j]
                if np.isfinite(val):
                    ts_min = min(val,ts_min)
                    count+=1
            if count>=min_periods:
                out[i,j] = ts_min
    return out

@njit(cache=True,parallel=True)
def ts_max_2d(values,n,min_periods):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            ts_max = -np.inf
            count = 0
            for t in range(start,i+1):
                val = values[t,j]
                if np.isfinite(val):
                    ts_max = max(val,ts_max)
                    count+=1
            if count>=min_periods:
                out[i,j] = ts_max
    return out

@njit(cache=True,parallel=True)
def ts_argmin_2d(values,n,min_periods):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            count = 0          # 记录窗口内有效值个数
            min_val = np.inf   # 记录当前最小值
            distance = 0       # 记录最小值出现日期与当前处理日期的差值
            for t in range(i,start-1,-1):
                val = values[t,j]
                if np.isfinite(val):
                    if val<min_val:    # 取最晚出现的最小值计算 
                        min_val = val
                        distance = i-t
                    count+=1
            if count>=min_periods:
                out[i,j] = distance+1  # 将结果范围调整到1~n
    return out

@njit(cache=True,parallel=True)
def ts_argmax_2d(values,n,min_periods):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            count = 0          
            max_val = -np.inf   
            distance = 0       
            for t in range(i,start-1,-1):
                val = values[t,j]
                if np.isfinite(val):
                    if val>max_val:     
                        max_val = val
                        distance = i-t
                    count+=1
            if count>=min_periods:
                out[i,j] = distance+1  
    return out

@njit(cache=True,parallel=True)
def ts_median_2d(values,n,min_periods): 
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        tmp = np.empty(n,dtype=np.float64)
        for i in range(min_periods-1,nd):
            start = max(0,i+1-n)
            count = 0
            for t in range(start,i+1):
                val = values[t,j]
                if np.isfinite(val):
                    tmp[count] = val
                    count += 1
            if count>=min_periods:
                vals = tmp[:count]
                vals.sort()
                if count%2==1:
                    med=vals[count//2]
                else:
                    med=0.5*(vals[count//2-1]+vals[count//2])
                out[i,j]=med
    return out

@njit(cache=True,parallel=True)
def ts_zscore_2d(values,n,min_periods):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        ts_sum = 0.0
        square_sum = 0.0
        count = 0 
        for i in range(nd):         # 滑动窗口优化
            val = values[i,j]
            if np.isfinite(val):
                ts_sum += val
                square_sum += val**2
                count += 1
            if i >= n:
                val_old = values[i-n,j]
                if np.isfinite(val_old):
                    ts_sum -= val_old
                    square_sum -= val_old**2
                    count -= 1
            if (count>=min_periods) and (count>=2) and np.isfinite(val):
                ts_mean = ts_sum/count
                ts_var = (square_sum-(ts_sum**2)/count)/(count-1)
                if ts_var>0.0:
                    ts_std = np.sqrt(ts_var)
                    if ts_std>1e-10:
                        out[i,j] = (val-ts_mean)/ts_std
    return out

@njit(cache=True,parallel=True)
def ema_2d(values,alpha,min_periods):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    beta=1-alpha
    for j in prange(ns):
        weighted_sum = 0.0
        weight_sum = 0.0
        count = 0
        for i in range(nd):
            val = values[i,j]
            weighted_sum *= beta
            weight_sum *= beta
            if np.isfinite(val):
                weighted_sum += val
                weight_sum += 1.0
                count += 1
            if count>=min_periods and weight_sum>=0.0:
                out[i,j] = weighted_sum/weight_sum
    return out

@njit(cache=True,parallel=True)
def wma_2d(values,w):
    nd,ns=values.shape
    n=w.shape[0]
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        for i in range(n-1,nd):
            if np.isfinite(values[i,j]):
                weighted_sum = 0.0
                weight_sum = 0.0
                for t in range(i+1-n,i+1):
                    val = values[t,j]
                    if np.isfinite(val):
                        weight = w[t+n-i-1]
                        weighted_sum += val*weight
                        weight_sum += weight
                if weight_sum!=0.0:
                    out[i,j] = weighted_sum/weight_sum
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
def cs_rank_2d(values,pct,ascending):
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
            out[i,idxs[0]]=0.5 if pct else 1.0
            continue
        # dense rank
        uniq=1
        prev=vals[0]
        for k in range(1,cnt):
            cur=vals[k]
            if cur!=prev:
                uniq+=1
                prev=cur
        if pct:
            if uniq==1:
                for k in range(cnt):
                    out[i,idxs[k]]=0.5
            else:
                denom=uniq-1
                r=0
                prev=vals[0]
                out[i,idxs[0]] = 0.0 if ascending else 1.0
                for k in range(1,cnt):
                    cur=vals[k]
                    if cur!=prev:
                        r+=1
                        prev=cur
                    x = r/denom
                    out[i,idxs[k]] = x if ascending else 1-x
        else:
            r=1
            prev=vals[0]
            out[i,idxs[0]] = 1.0 if ascending else float(uniq)
            for k in range(1,cnt):
                cur=vals[k]
                if cur!=prev:
                    r+=1
                    prev=cur
                out[i,idxs[k]] = float(r) if ascending else float(uniq-r+1)
    return out

@MinuteFactorEngine.register("neutralize")
@njit(cache=True,parallel=True)
def cs_neutralize_2d(values):
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
def cs_zscore_2d(values):
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

@njit(cache=True,parallel=True)
def cs_weighted_zscore_2d(values,weights):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for i in prange(nd):
        sw=0.0
        swv=0.0
        c=0
        for j in range(ns):
            v=values[i,j]
            w=weights[i,j]
            if np.isfinite(v) and np.isfinite(w):
                sw+=w
                swv+=w*v
                c+=1
        if c==0:
            continue
        elif c==1:
            for j in range(ns):
                if np.isfinite(values[i,j]) and np.isfinite(weights[i,j]):
                    out[i,j]=0.0
        else:
            if sw==0.0:
                continue
            wmean=swv/sw
            ss=0.0
            for j in range(ns):
                v=values[i,j]
                w=weights[i,j]
                if np.isfinite(v) and np.isfinite(w):
                    ss+=(v-wmean)*(v-wmean)
            var=ss/(c-1)
            if var<=0.0:
                for j in range(ns):
                    if np.isfinite(values[i,j]) and np.isfinite(weights[i,j]):
                        out[i,j]=0.0
            else:
                std=np.sqrt(var)
                for j in range(ns):
                    v=values[i,j]
                    w=weights[i,j]
                    if np.isfinite(v) and np.isfinite(w):
                        out[i,j]=(v-wmean)/std
    return out


@njit(cache=True,parallel=True)
def cs_scale_2d(values):
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

@njit(cache=True,parallel=True)
def cs_multireg_2d(X,y,min_cs):
    nd,ns = y.shape
    K = X.shape[2]
    p = K+1
    out = np.full((nd,ns),np.nan)
    for i in prange(nd):
        cnt = 0
        valid_idxs = np.empty(ns,dtype=np.int32)
        for j in range(ns):
            ok = False
            if np.isfinite(y[i,j]):
                ok = True
                for k in range(K):
                    if not np.isfinite(X[i,j,k]):
                        ok = False
                        break
            if ok:
                valid_idxs[cnt] = j
                cnt+=1
        if cnt>=min_cs and cnt>=p:
            XtX = np.zeros((p,p),dtype=np.float64)
            Xty = np.zeros(p,dtype=np.float64)
            for t in range(cnt):
                j = valid_idxs[t]
                yj = y[i,j]
                XtX[0,0] += 1.0
                Xty[0] += yj
                for k in range(K):
                    vk = X[i,j,k]
                    XtX[0,k+1] += vk
                    XtX[k+1,0] += vk
                    Xty[k+1] += vk*yj
                    for m in range(K):
                        XtX[k+1,m+1] += vk*X[i,j,m]
            for d in range(p):
                XtX[d,d] += 1e-10
            beta = np.linalg.solve(XtX,Xty)
            for t in range(cnt):
                j = valid_idxs[t]
                yhat = beta[0]
                for k in range(K):
                    yhat += X[i,j,k]*beta[k+1]
                out[i,j] = y[i,j]-yhat
    return out

@njit(cache=True,parallel=True)
def cs_corr_2d(x,y):
    nd,ns = x.shape
    out = np.full(nd,np.nan,np.float64)
    for i in prange(nd):
        cnt = 0
        sx = 0.0
        sy = 0.0
        for j in range(ns):
            a = x[i,j]
            b = y[i,j]
            if np.isfinite(a) and np.isfinite(b):
                cnt += 1
                sx += a
                sy += b
        if cnt<2:
            continue

        mx = sx/cnt
        my = sy/cnt

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

        if sxx<=0.0 or syy<=0.0:
            continue
        out[i] = sxy/np.sqrt(sxx*syy)
    return out

@njit(cache=True,parallel=True)
def cs_mean_2d(values):
    nd,ns = values.shape
    out = np.full(nd,np.nan,np.float64)
    for i in prange(nd):
        s = 0.0
        cnt = 0
        for j in range(ns):
            v = values[i,j]
            if np.isfinite(v):
                s += v
                cnt += 1
        if cnt>0:
            out[i] = s/cnt
    return out

@njit(cache=True)
def cs_mean_1d(values):
    s = 0.0
    cnt = 0
    n = values.shape[0]
    for i in range(n):
        v = values[i]
        if np.isfinite(v):
            s += v
            cnt += 1
    if cnt==0:
        return np.nan
    return s/cnt

@njit(cache=True,parallel=True)
def bucket_2d(values,n):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for i in prange(nd):
        m = 0
        idxs = np.empty(ns,dtype=np.int32)
        vals = np.empty(ns,dtype=np.float64)
        # 第一次扫描: 处理NaN值
        for j in range(ns):   
            val = values[i,j]
            if np.isfinite(val):
                idxs[m] = j
                vals[m] = val
                m += 1
        if m==0:
            continue
        if m==1:
            out[i,idxs[0]] = 0.0
            continue
        orders = np.argsort(vals[:m])
        ranks = np.empty(m,dtype=np.int32)
        rank = 0
        # 第二次扫描: 给出排序值
        for j in range(m-1):  
            pos_current = orders[j]
            pos_next = orders[j+1]
            val_current = vals[pos_current]
            val_next = vals[pos_next]
            ranks[pos_current] = rank
            if val_next != val_current:
                rank += 1
        ranks[pos_next] = rank
        # 第三次扫描: 填充组号
        if rank==0: # rank=0代表只有1个唯一值,给出0.5的排序值           
            for j in range(m):
                out[i,idxs[j]] = 0.5
        else:
            for j in range(m):
                out[i,idxs[j]] = np.floor(ranks[j]/rank*(n-1e-9))
    return out

@njit(cache=True,parallel=True)
def group_rank_2d(values,gcodes,gnum):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    if gnum<=0:
        return out
    for i in prange(nd):
        # 构造组哈希结构
        gcounts=np.zeros(gnum,dtype=np.int32)
        for j in range(ns):
            g=gcodes[i,j]
            v=values[i,j]
            if g>=0 and g<gnum and np.isfinite(v):
                gcounts[g]+=1
        gstarts=np.empty(gnum+1,dtype=np.int32)
        gstarts[0]=0
        for g in range(gnum):
            gstarts[g+1]=gstarts[g]+gcounts[g]
        tot=gstarts[gnum]
        if tot==0:
            continue
        writepos=np.empty(gnum,dtype=np.int32)
        for g in range(gnum):
            writepos[g]=gstarts[g]
        vals=np.empty(tot,dtype=np.float64)
        idxs=np.empty(tot,dtype=np.int32)
        for j in range(ns):
            g=gcodes[i,j]
            v=values[i,j]
            if g>=0 and g<gnum and np.isfinite(v):
                p=writepos[g]
                vals[p]=v
                idxs[p]=j
                writepos[g]=p+1
        stack_lo=np.empty(64,dtype=np.int32)
        stack_hi=np.empty(64,dtype=np.int32)
        rbuf=np.empty(tot,dtype=np.int32)
        # 每组内排序
        for g in range(gnum):
            cnt=gcounts[g]
            if cnt<=0:
                continue
            start=gstarts[g]
            end=start+cnt
            if cnt==1:
                out[i,idxs[start]]=0.5
                continue
            _quicksort_pair_ws(vals,idxs,start,end-1,stack_lo,stack_hi)
            r=0
            prev=vals[start]
            rbuf[start]=0
            for k in range(start+1,end):
                cur=vals[k]
                if cur!=prev:
                    r+=1
                    prev=cur
                rbuf[k]=r
            if r==0:
                for k in range(start,end):
                    out[i,idxs[k]]=0.5
            else:
                denom=r
                for k in range(start,end):
                    out[i,idxs[k]]=rbuf[k]/denom
    return out

@njit(cache=True,parallel=True)
def group_neutralize_2d(values,gcodes,gnum):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for i in prange(nd):
        gsum=np.zeros(gnum,dtype=np.float64)
        gcnt=np.zeros(gnum,dtype=np.int32)
        for j in range(ns):
            g=gcodes[i,j]
            v=values[i,j]
            if g!=-1 and np.isfinite(v):
                gsum[g]+=v
                gcnt[g]+=1
        for j in range(ns):
            g=gcodes[i,j]
            v=values[i,j]
            if g!=-1 and np.isfinite(v):
                c=gcnt[g]
                out[i,j]=v-gsum[g]/c
    return out

@njit(cache=True,parallel=True)
def group_zscore_2d(values,gcodes,gnum):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for i in prange(nd):
        gsum=np.zeros(gnum,dtype=np.float64)
        gsqsum=np.zeros(gnum,dtype=np.float64)
        gcnt=np.zeros(gnum,dtype=np.int32)
        for j in range(ns):
            g=gcodes[i,j]
            v=values[i,j]
            if g!=-1 and np.isfinite(v):
                gsum[g]+=v
                gsqsum[g]+=v**2
                gcnt[g]+=1
        for j in range(ns):
            g=gcodes[i,j]
            v=values[i,j]
            if g!=-1 and np.isfinite(v):
                c=gcnt[g]
                if c==1:
                    out[i,j]=0.0
                else:
                    gmean=gsum[g]/c
                    gvar=(gsqsum[g]-(gsum[g]*gsum[g])/c)/(c-1)
                    if gvar<=0.0:
                        out[i,j]=0.0
                    else:
                        out[i,j]=(v-gmean)/np.sqrt(gvar)
    return out

@njit(cache=True,parallel=True)
def group_scale_2d(values,gcodes,gnum):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for i in prange(nd):
        gmaxs=np.full(gnum,-np.inf)
        gmins=np.full(gnum,np.inf)
        for j in range(ns):
            g=gcodes[i,j]
            v=values[i,j]
            if g!=-1 and np.isfinite(v):
                gmaxs[g] = np.maximum(v,gmaxs[g])
                gmins[g] = np.minimum(v,gmins[g])
        for j in range(ns):
            g=gcodes[i,j]
            v=values[i,j]
            if g!=-1 and np.isfinite(v):
                gmax = gmaxs[g]
                gmin = gmins[g]
                if gmax-gmin<1e-8:
                    out[i,j] = 0.5
                else:
                    out[i,j] = (v-gmin)/(gmax-gmin)
    return out

@njit(cache=True,parallel=True)
def group_mean_2d(values,gcodes,gnum):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for i in prange(nd):
        gsum=np.zeros(gnum,dtype=np.float64)
        gcnt=np.zeros(gnum,dtype=np.int32)
        for j in range(ns):
            g=gcodes[i,j]
            v=values[i,j]
            if g!=-1 and np.isfinite(v):
                gsum[g]+=v
                gcnt[g]+=1
        for j in range(ns):
            g=gcodes[i,j]
            v=values[i,j]
            if g!=-1 and np.isfinite(v):
                c=gcnt[g]
                gmean=gsum[g]/c
                out[i,j]=gmean
    return out
 
@njit(cache=True,parallel=True)
def trade_when_2d(values,con1,con2):
    nd,ns=values.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        last = np.nan
        for i in range(nd):
            c1 = con1[i,j]
            c2 = con2[i,j]
            if c1 or c2:
                last = np.nan
            if c1 and (not c2):
                v = values[i,j]
                if np.isfinite(v):
                    last = v
            out[i,j] = last
    return out

@njit(cache=True,parallel=True)
def ts_corr_2d(values1,values2,n,min_periods,hold_nan):
    nd,ns=values1.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        sumx = 0.0
        sumy = 0.0
        sumxx = 0.0
        sumyy = 0.0
        sumxy = 0.0
        count = 0
        for i in range(nd):
            x = values1[i,j]
            y = values2[i,j]
            if np.isfinite(x) and np.isfinite(y):
                sumx+=x
                sumy+=y
                sumxx+=x**2
                sumyy+=y**2
                sumxy+=x*y
                count+=1
            if i>=n:
                x0 = values1[i-n,j]
                y0 = values2[i-n,j]
                if np.isfinite(x0) and np.isfinite(y0):
                    sumx-=x0
                    sumy-=y0
                    sumxx-=x0**2
                    sumyy-=y0**2
                    sumxy-=x0*y0
                    count-=1
            if (count>=min_periods):
                if (~hold_nan) or (np.isfinite(y) and np.isfinite(x)):
                    cov = (sumxy-sumx*sumy/count)/(count-1)
                    varx = (sumxx-sumx**2/count)/(count-1)
                    vary = (sumyy-sumy**2/count)/(count-1)
                    if varx>0.0 and vary>0.0:
                        denom = np.sqrt(varx*vary)
                        if denom>1e-10:
                            out[i,j] = cov/denom
    return out

@njit(cache=True,parallel=True)
def fd_sum_2d(z1,z2,w1,w2,w3,w4):
    nd,ns=z1.shape
    out=np.empty((nd,ns),dtype=np.float64)
    for i in prange(nd):
        for j in range(ns):
            a=z1[i,j]
            b=z2[i,j]
            if not (np.isfinite(a) and np.isfinite(b)):
                out[i,j]=0.0
                continue
            if a>0.0 and b>0.0:
                w=w1
            elif a<0.0 and b<0.0:
                w=w2
            elif a>0.0 and b<0.0:
                w=w3
            elif a<0.0 and b>0.0:
                w=w4
            else:
                w=0.0
            out[i,j]=np.abs(a)*np.abs(b)*w
    return out
    
@njit(cache=True,parallel=True)
def ts_reg_2d(xv,yv,n,min_periods,rettype):
    nd,ns=yv.shape
    out=np.full((nd,ns),np.nan)
    for j in prange(ns):
        sumx=0.0
        sumy=0.0
        sumxx=0.0
        sumxy=0.0
        sumyy=0.0
        cnt=0
        for i in range(nd):
            x=xv[i,j]
            y=yv[i,j]
            if np.isfinite(x) and np.isfinite(y):
                sumx+=x
                sumy+=y
                sumxx+=x*x
                sumxy+=x*y
                sumyy+=y*y
                cnt+=1
            if i>=n:
                x0=xv[i-n,j]
                y0=yv[i-n,j]
                if not np.isnan(x0) and not np.isnan(y0):
                    sumx-=x0
                    sumy-=y0
                    sumxx-=x0*x0
                    sumxy-=x0*y0
                    sumyy-=y0*y0
                    cnt-=1
            if cnt>=min_periods:
                mx=sumx/cnt
                my=sumy/cnt
                den=sumxx-sumx*sumx/cnt
                if den!=0.0:
                    b=(sumxy-sumx*sumy/cnt)/den
                    a=my-b*mx
                    if rettype==1:
                        out[i,j]=b
                    elif rettype==2:
                        out[i,j]=a
                    elif rettype==3:
                        var_y = sumyy-(sumy**2)/cnt
                        if var_y>0.0:
                            covxy = sumxy-sumx*sumy/cnt
                            out[i,j] = (covxy**2)/(den*var_y)
                    else:
                        if not np.isnan(x) and not np.isnan(y):
                            out[i,j]=y-(a+b*x)
    return out

@njit(cache=True,parallel=True)
def ts_multireg_2d(Xv,yv,is_valid,n,min_periods,rettype):
    nd,ns = yv.shape
    K = Xv.shape[2]
    p = K+1
    out = np.full((nd,ns),np.nan,dtype=np.float64)
    for j in prange(ns):
        XtX = np.zeros((p,p),dtype=np.float64)
        Xty = np.zeros(p,dtype=np.float64)
        cnt = 0
        for i in range(nd):
            if is_valid[i,j]:
                y = yv[i,j]
                XtX[0,0] += 1.0
                Xty[0] += y
                for k in range(K):
                    xk = Xv[i,j,k]
                    XtX[0,k+1] += xk
                    XtX[k+1,0] += xk
                    Xty[k+1] += xk*y
                    for m in range(K):
                        XtX[k+1,m+1] += xk*Xv[i,j,m]
                cnt += 1
            if i>=n:
                i0 = i-n
                if is_valid[i0,j]:
                    y0 = yv[i0,j]
                    XtX[0,0] -= 1.0
                    Xty[0] -= y0
                    for k in range(K):
                        xk0 = Xv[i0,j,k]
                        XtX[0,k+1] -= xk0
                        XtX[k+1,0] -= xk0
                        Xty[k+1] -= xk0*y0
                        for m in range(K):
                            XtX[k+1,m+1] -= xk0*Xv[i0,j,m]
                    cnt -= 1
            if cnt>=min_periods and cnt>=p:
                for d in range(p):
                    XtX[d,d] += 1e-10
                beta = np.linalg.solve(XtX,Xty)
                for d in range(p):
                    XtX[d,d] -= 1e-10
                if rettype==3:
                    sse = 0.0
                    sumy = 0.0
                    sumyy = 0.0
                    for t in range(max(0,i-n+1),i+1):
                        if is_valid[t,j]:
                            yt = yv[t,j]
                            sumy += yt
                            sumyy += yt*yt
                            yhat = beta[0]
                            for k in range(K):
                                yhat += Xv[t,j,k]*beta[k+1]
                            e = yt-yhat
                            sse += e**2
                    var_y = sumyy-(sumy**2)/cnt
                    if var_y>0.0:
                        out[i,j] = 1.0-sse/var_y
                else: 
                    if is_valid[i,j]:
                        yhat = beta[0]
                        for k in range(K):
                            yhat += Xv[i,j,k]*beta[k+1]
                        out[i,j] = yv[i,j]-yhat
    return out

@njit(cache=True,parallel=True)
def group_topbot_2d(returns,ranking,n_bins,qt):
    nd,ns=returns.shape
    grprets=np.full((nd,n_bins),0.0,dtype=np.float64)
    toprets=np.full(nd,np.nan,dtype=np.float64)
    botrets=np.full(nd,np.nan,dtype=np.float64)
    long_effs=np.zeros(nd,dtype=np.int64)
    scale=n_bins-1e-9
    thr=1.0-qt
    for i in prange(nd):
        gs=np.zeros(n_bins,dtype=np.float64)
        gc=np.zeros(n_bins,dtype=np.int64)
        top_sum=0.0
        top_cnt=0
        bot_sum=0.0
        bot_cnt=0
        le=0
        for j in range(ns):
            rk=ranking[i,j]
            if np.isfinite(rk):
                if rk>thr:
                    le+=1
                r=returns[i,j]
                if np.isfinite(r):
                    g=int(rk*scale)
                    if g<0:
                        g=0
                    elif g>=n_bins:
                        g=n_bins-1
                    gs[g]+=r
                    gc[g]+=1
                    if rk<qt:
                        top_sum+=r
                        top_cnt+=1
                    if rk>thr:
                        bot_sum+=r
                        bot_cnt+=1
        long_effs[i]=le
        for k in range(n_bins):
            if gc[k]>0:
                grprets[i,k]=gs[k]/gc[k]
            else:
                grprets[i,k]=0.0
        if top_cnt>0:
            toprets[i]=top_sum/top_cnt
        if bot_cnt>0:
            botrets[i]=bot_sum/bot_cnt
    return grprets,toprets,botrets,long_effs

@njit(cache=True)
def calc_turnover_2d(ranking,qt):
    nd,ns=ranking.shape
    top_trade=np.zeros(nd,dtype=np.float64)
    bot_trade=np.zeros(nd,dtype=np.float64)
    thr=1.0-qt
    for i in range(1,nd):
        m_top=0
        m_top_prev=0
        inter_top=0
        m_bot=0
        m_bot_prev=0
        inter_bot=0
        for j in range(ns):
            r=ranking[i,j]
            rp=ranking[i-1,j]
            cur_top=np.isfinite(r) and (r<qt)
            pre_top=np.isfinite(rp) and (rp<qt)
            if cur_top:
                m_top+=1
            if pre_top:
                m_top_prev+=1
            if cur_top and pre_top:
                inter_top+=1
            cur_bot=np.isfinite(r) and (r>thr)
            pre_bot=np.isfinite(rp) and (rp>thr)
            if cur_bot:
                m_bot+=1
            if pre_bot:
                m_bot_prev+=1
            if cur_bot and pre_bot:
                inter_bot+=1
        w_cur=0.0
        w_pre=0.0
        if m_top>0:
            w_cur=1.0/m_top
        if m_top_prev>0:
            w_pre=1.0/m_top_prev
        top_trade[i]=(m_top-inter_top)*w_cur+(m_top_prev-inter_top)*w_pre+inter_top*abs(w_cur-w_pre)
        w_cur=0.0
        w_pre=0.0
        if m_bot>0:
            w_cur=1.0/m_bot
        if m_bot_prev>0:
            w_pre=1.0/m_bot_prev
        bot_trade[i]=(m_bot-inter_bot)*w_cur+(m_bot_prev-inter_bot)*w_pre+inter_bot*abs(w_cur-w_pre)
    return top_trade,bot_trade

@njit(cache=True,parallel=True)
def fast_corr_2d(X):
    nd,ns = X.shape
    Z = np.zeros((nd,ns),dtype=np.float64)
    mask = np.zeros((nd,ns),dtype=np.float64)
    for j in prange(ns): 
        s=0.0
        ss=0.0
        c=0
        for i in range(nd):
            v=X[i,j]
            if np.isfinite(v):
                s+=v
                ss+=v*v
                c+=1
        if c<=1:
            continue
        mean=s/c
        var=(ss-s*s/c)/(c-1)
        if var>1e-12:
            std = np.sqrt(var)
            for i in range(nd):
                v = X[i,j] 
                if np.isfinite(v):
                    Z[i,j] = (v-mean)/std
                    mask[i,j] = 1.0
    num = Z.T@Z
    den = mask.T@mask
    for i in range(ns):
        for j in range(ns):
            d = den[i,j]-1.0
            if d>1e-9:
                num[i,j] /= d
            else:
                num[i,j] = np.nan
    for i in range(ns):
        num[i,i] = 1.0
    return num

@njit(cache=True,parallel=True)
def precise_corr_2d(X):
    nd,ns = X.shape
    out = np.full((ns,ns),np.nan)
    for i in prange(ns):
        cnt = 0
        for k in range(nd):
            if np.isfinite(X[k,i]):
                cnt+=1
        if cnt>1:
            out[i,i] = 1.0 
        for j in range(i+1,ns):
            sx = 0.0
            sy = 0.0
            sxx = 0.0
            syy = 0.0
            sxy = 0.0
            cnt = 0
            for k in range(nd):
                vx = X[k,i]
                vy = X[k,j]
                if np.isfinite(vx) and np.isfinite(vy):
                    sx += vx
                    sy += vy
                    sxx += vx*vx
                    syy += vy*vy
                    sxy += vx*vy
                    cnt += 1    
            if cnt>1:
                term1 = cnt*sxx-sx*sx
                term2 = cnt*syy-sy*sy
                if term1>0 and term2>0:
                    val = (cnt*sxy-sx*sy)/np.sqrt(term1*term2)
                    out[i,j] = val
                    out[j,i] = val
    return out

@njit(cache=True,parallel=True)
def cs_pc1_2d(X,min_cs):
    nd,ns,K = X.shape
    out = np.full((nd,ns),np.nan)
    for i in prange(nd):
        valid_idxs = np.empty(ns,dtype=np.int32)
        cnt = 0
        for j in range(ns):
            ok = True
            for k in range(K):
                if not np.isfinite(X[i,j,k]):
                    ok = False
                    break
            if ok:
                valid_idxs[cnt] = j
                cnt += 1
        if cnt>=min_cs:
            M = np.empty((cnt,K),dtype=np.float64)
            for t in range(cnt):
                j = valid_idxs[t]
                for k in range(K):
                    M[t,k] = X[i,j,k]
            means = np.zeros(K,dtype=np.float64)
            for k in range(K):
                s = 0.0
                for t in range(cnt):
                    s += M[t,k]
                m = s/cnt
                means[k] = m
                for t in range(cnt):
                    M[t,k] -= m
            S = np.zeros((K,K),dtype=np.float64)
            for a in range(K):
                for b in range(a,K):
                    s = 0.0
                    for t in range(cnt):
                        s += M[t,a]*M[t,b]
                    s /= cnt
                    S[a,b] = s
                    if a!= b:
                        S[b,a] = s
            for d in range(K):
                S[d,d] += 1e-12
            w,v = np.linalg.eigh(S)
            idx_max = 0
            maxw = w[0]
            for k in range(1,K):
                if w[k]>maxw:
                    maxw = w[k]
                    idx_max = k
            pc1 = v[:,idx_max]
            if pc1[0]<0.0:
                for k in range(K):
                    pc1[k] = -pc1[k]
            for t in range(cnt):
                j = valid_idxs[t]
                s = 0.0
                for k in range(K):
                    s += M[t,k]*pc1[k]
                out[i,j] = s
    return out




            





        

            


            
        




