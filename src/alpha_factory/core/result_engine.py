import numpy as np
from numba import njit,prange
from ..factory_ops.numpy_funcs import con_and,con_not,ts_delay,cs_rank,mask
from ..factory_ops.numba_kernels import _quicksort_pair_ws
from ..factory_data.data_manager import DataManager
from .factor import FactorResult


class ResultEngine:
    def __init__(self,dm:DataManager,period:str="is"):
        self.dm = dm
        self.period = period
        self.top_pct = 0.2
        self.ann_days = 243

        self.is_start = 730
        self.is_end = 2674
        self.os_start = 2675

        self._check_period()

        limit_up_cto = self._load_data("limit_up_cto")
        tradables = self._load_data("tradables")
        self.buyables = ts_delay(con_and(tradables,con_not(limit_up_cto)),-1)[:-2]

        self.universe = self._load_data("standards")[:-2]
        self.ret = mask(ts_delay(self._load_data("trade_returns"),-2)[:-2],con_and(self.buyables,self.universe))
        self.ret_rank = cs_rank(self.ret,True,True)

    def calc(self,value):
        value = self._slice(value)[:-2]

        ic,rankic,rankicir,longret,turnover,coverage,rankics = result_calc_numba_kernel(
            value,
            self.universe,
            self.buyables,
            self.ret,
            self.ret_rank,
            self.top_pct,
            self.ann_days,
        )

        return FactorResult(
            ic=ic,
            rankic=rankic,
            rankicir=rankicir,
            longret=longret,
            turnover=turnover,
            coverage=coverage,
            rankics=rankics,
        )

    def _check_period(self):
        if self.period not in ("is","os","full"):
            raise ValueError(f"invalid period: {self.period}")

    def _load_data(self,name):
        arr = self.dm.get_data(name)
        return self._slice(arr)

    def _slice(self,arr):
        if self.period=="is":
            return arr[self.is_start:self.is_end+1]
        if self.period=="os":
            return arr[self.os_start:]
        if self.period=="full":
            return arr[self.is_start:]
        raise ValueError(f"invalid period: {self.period}")


@njit(cache=True)
def _mean1d(x):
    s = 0.0
    c = 0
    for i in range(x.shape[0]):
        v = x[i]
        if np.isfinite(v):
            s += v
            c += 1
    if c==0:
        return np.nan
    return s/c


@njit(cache=True)
def _ir1d(x):
    s = 0.0
    ss = 0.0
    c = 0
    for i in range(x.shape[0]):
        v = x[i]
        if np.isfinite(v):
            s += v
            ss += v*v
            c += 1
    if c<=1:
        return np.nan
    m = s/c
    var = (ss-s*s/c)/(c-1)
    if var<=0.0:
        return np.nan
    sd = np.sqrt(var)
    if sd<1e-10:
        return np.nan
    return m/sd


@njit(cache=True,parallel=True)
def result_calc_numba_kernel(value,universe,buyables,ret,ret_rank,top_pct,ann_days):
    nd,ns = value.shape
    rankics = np.full(nd,np.nan,np.float64)
    ics = np.full(nd,np.nan,np.float64)
    longrets = np.full(nd,np.nan,np.float64)
    coverages = np.full(nd,np.nan,np.float64)
    top_mask = np.zeros((nd,ns),np.uint8)
    top_counts = np.zeros(nd,np.int32)
    threshold = 1.0-top_pct

    for i in prange(nd):
        denom = 0
        numer = 0
        cnt = 0

        vals = np.empty(ns,np.float64)
        idxs = np.empty(ns,np.int32)

        for j in range(ns):
            if universe[i,j]==1:
                denom += 1
                if np.isfinite(value[i,j]):
                    numer += 1

            v = value[i,j]
            if buyables[i,j]==1 and np.isfinite(v):
                vals[cnt] = v
                idxs[cnt] = j
                cnt += 1

        if denom>0:
            coverages[i] = numer/denom

        if cnt==0:
            continue

        if cnt>1:
            stack_lo = np.empty(cnt,np.int32)
            stack_hi = np.empty(cnt,np.int32)
            _quicksort_pair_ws(vals,idxs,0,cnt-1,stack_lo,stack_hi)

        uniq = 1
        if cnt>1:
            prev = vals[0]
            for k in range(1,cnt):
                cur = vals[k]
                if cur!=prev:
                    uniq += 1
                    prev = cur

        sxr = 0.0
        syr = 0.0
        sxxr = 0.0
        syyr = 0.0
        sxyr = 0.0
        cr = 0

        sxi = 0.0
        syi = 0.0
        sxxi = 0.0
        syyi = 0.0
        sxyi = 0.0
        ci = 0

        top_ret_sum = 0.0
        tc = 0

        r = 0
        prev = vals[0]
        denom_rank = uniq-1

        for k in range(cnt):
            if cnt==1 or uniq==1:
                z = 0.5
            else:
                cur = vals[k]
                if k>0 and cur!=prev:
                    r += 1
                    prev = cur
                z = r/denom_rank

            j = idxs[k]

            rr = ret_rank[i,j]
            if np.isfinite(rr):
                sxr += z
                syr += rr
                sxxr += z*z
                syyr += rr*rr
                sxyr += z*rr
                cr += 1

            rv = ret[i,j]
            if np.isfinite(rv):
                vx = vals[k]
                sxi += vx
                syi += rv
                sxxi += vx*vx
                syyi += rv*rv
                sxyi += vx*rv
                ci += 1

            if z>=threshold:
                top_mask[i,j] = 1
                tc += 1
                if np.isfinite(rv):
                    top_ret_sum += rv

        top_counts[i] = tc
        if tc>0:
            longrets[i] = top_ret_sum/tc*ann_days
        else:
            longrets[i] = 0.0

        if cr>=2:
            vx = sxxr-sxr*sxr/cr
            vy = syyr-syr*syr/cr
            cv = sxyr-sxr*syr/cr
            if vx>0.0 and vy>0.0:
                rankics[i] = cv/np.sqrt(vx*vy)

        if ci>=2:
            vx = sxxi-sxi*sxi/ci
            vy = syyi-syi*syi/ci
            cv = sxyi-sxi*syi/ci
            if vx>0.0 and vy>0.0:
                ics[i] = cv/np.sqrt(vx*vy)

    turnovers = np.full(nd-1,np.nan,np.float64)
    for i in prange(1,nd):
        c0 = top_counts[i-1]
        c1 = top_counts[i]
        s = 0.0
        for j in range(ns):
            w0 = 0.0
            w1 = 0.0
            if c0>0 and top_mask[i-1,j]==1:
                w0 = 1.0/c0
            if c1>0 and top_mask[i,j]==1:
                w1 = 1.0/c1
            s += abs(w1-w0)
        turnovers[i-1] = s

    return (
        _mean1d(ics),
        _mean1d(rankics),
        _ir1d(rankics),
        _mean1d(longrets),
        _mean1d(turnovers),
        _mean1d(coverages),
        rankics,
    )
