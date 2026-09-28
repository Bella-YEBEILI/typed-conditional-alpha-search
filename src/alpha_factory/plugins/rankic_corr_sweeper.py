import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit,prange


class RankicCorrSweeper:
    def __init__(self,rankics_dir=None):
        self.root = Path(__file__).resolve().parents[1]
        self.rankics_dir = Path(rankics_dir) if rankics_dir is not None else self.root/"excavate_data"/"rankics"

    def sweep(self,
              rankic_thresholds,
              corr_thresholds,
              db_paths=None,
              only_seed=True,
              rankicir_thresholds=(None,),
              max_factors=None):
        spec_ids = self._load_spec_ids(db_paths,only_seed)
        ids,rankics = self._load_rankics(spec_ids)
        rankic,rankicir = _rankic_stats(rankics)

        base = np.isfinite(rankic)
        min_rankic = min(rankic_thresholds)
        base &= rankic>=min_rankic
        finite_ir_thresholds = [x for x in rankicir_thresholds if x is not None]
        if len(finite_ir_thresholds)>0:
            base &= rankicir>=min(finite_ir_thresholds)

        keep_idx = np.where(base)[0]
        if max_factors is not None and keep_idx.shape[0]>max_factors:
            order = keep_idx[np.argsort(rankic[keep_idx])[::-1]]
            keep_idx = order[:max_factors]

        ids = [ids[i] for i in keep_idx]
        rankics = rankics[keep_idx]
        rankic = rankic[keep_idx]
        rankicir = rankicir[keep_idx]
        corr = _pair_corr_matrix(rankics)

        rows = []
        selected = {}
        for rankic_threshold in rankic_thresholds:
            for rankicir_threshold in rankicir_thresholds:
                for corr_threshold in corr_thresholds:
                    picked = self._greedy_select(
                        rankic,
                        rankicir,
                        corr,
                        rankic_threshold,
                        rankicir_threshold,
                        corr_threshold,
                    )
                    picked_ids = [ids[i] for i in picked]
                    key = (rankic_threshold,rankicir_threshold,corr_threshold)
                    selected[key] = picked_ids
                    rows.append(self._summary_row(
                        rankic_threshold,
                        rankicir_threshold,
                        corr_threshold,
                        picked,
                        rankic,
                        rankicir,
                        corr,
                    ))

        return pd.DataFrame(rows),selected

    def load_rankic_table(self,db_paths=None,only_seed=True):
        spec_ids = self._load_spec_ids(db_paths,only_seed)
        ids,rankics = self._load_rankics(spec_ids)
        rankic,rankicir = _rankic_stats(rankics)
        return pd.DataFrame({
            "spec_id":ids,
            "rankic":rankic,
            "rankicir":rankicir,
        }).sort_values("rankic",ascending=False)

    def _load_spec_ids(self,db_paths,only_seed):
        if db_paths is None:
            return [p.stem for p in self.rankics_dir.glob("*.npy")]

        ids = []
        for db_path in db_paths:
            path = Path(db_path)
            if not path.is_absolute():
                path = self.root/path
            where = "WHERE is_seed=1" if only_seed else ""
            with sqlite3.connect(str(path)) as conn:
                rows = conn.execute(f"SELECT DISTINCT spec_id FROM eval {where}").fetchall()
            ids.extend([row[0] for row in rows])
        return sorted(set(ids))

    def _load_rankics(self,spec_ids):
        ids = []
        arrs = []
        for spec_id in spec_ids:
            path = self.rankics_dir/f"{spec_id}.npy"
            if path.exists():
                ids.append(spec_id)
                arrs.append(np.load(path).astype(np.float64))
        return ids,np.vstack(arrs)

    @staticmethod
    def _greedy_select(rankic,rankicir,corr,rankic_threshold,rankicir_threshold,corr_threshold):
        order = np.argsort(rankic)[::-1]
        picked = []
        for i in order:
            if not np.isfinite(rankic[i]) or rankic[i]<rankic_threshold:
                continue
            if rankicir_threshold is not None and (not np.isfinite(rankicir[i]) or rankicir[i]<rankicir_threshold):
                continue
            ok = True
            for j in picked:
                c = corr[i,j]
                if not np.isfinite(c):
                    continue
                if c>corr_threshold:
                    ok = False
                    break
            if ok:
                picked.append(i)
        return picked

    @staticmethod
    def _summary_row(rankic_threshold,rankicir_threshold,corr_threshold,picked,rankic,rankicir,corr):
        if len(picked)==0:
            return {
                "rankic_threshold":rankic_threshold,
                "rankicir_threshold":rankicir_threshold,
                "corr_threshold":corr_threshold,
                "count":0,
                "avg_rankic":np.nan,
                "min_rankic":np.nan,
                "avg_rankicir":np.nan,
                "max_pair_corr":np.nan,
                "avg_pair_corr":np.nan,
            }

        pair_corrs = []
        for a in range(len(picked)):
            for b in range(a+1,len(picked)):
                c = corr[picked[a],picked[b]]
                if np.isfinite(c):
                    pair_corrs.append(c)

        return {
            "rankic_threshold":rankic_threshold,
            "rankicir_threshold":rankicir_threshold,
            "corr_threshold":corr_threshold,
            "count":len(picked),
            "avg_rankic":float(np.nanmean(rankic[picked])),
            "min_rankic":float(np.nanmin(rankic[picked])),
            "avg_rankicir":float(np.nanmean(rankicir[picked])),
            "max_pair_corr":max(pair_corrs) if pair_corrs else np.nan,
            "avg_pair_corr":sum(pair_corrs)/len(pair_corrs) if pair_corrs else np.nan,
        }


@njit(cache=True,parallel=True)
def _rankic_stats(rankics):
    n,t = rankics.shape
    rankic = np.empty(n,np.float64)
    rankicir = np.empty(n,np.float64)
    for i in prange(n):
        s = 0.0
        ss = 0.0
        c = 0
        for j in range(t):
            v = rankics[i,j]
            if np.isfinite(v):
                s += v
                ss += v*v
                c += 1
        if c==0:
            rankic[i] = np.nan
            rankicir[i] = np.nan
            continue
        m = s/c
        rankic[i] = m
        if c<=1:
            rankicir[i] = np.nan
            continue
        var = (ss-s*s/c)/(c-1)
        if var<=0.0:
            rankicir[i] = np.nan
        else:
            sd = np.sqrt(var)
            rankicir[i] = m/sd if sd>=1e-10 else np.nan
    return rankic,rankicir


@njit(cache=True,parallel=True)
def _pair_corr_matrix(x):
    n,t = x.shape
    out = np.empty((n,n),np.float32)
    for i in prange(n):
        out[i,i] = 1.0
        for j in range(i+1,n):
            c = _corr_1d(x[i],x[j],t)
            out[i,j] = c
            out[j,i] = c
    return out


@njit(cache=True)
def _corr_1d(x,y,t):
    sx = 0.0
    sy = 0.0
    cnt = 0
    for i in range(t):
        vx = x[i]
        vy = y[i]
        if np.isfinite(vx) and np.isfinite(vy):
            sx += vx
            sy += vy
            cnt += 1
    if cnt<2:
        return np.nan

    mx = sx/cnt
    my = sy/cnt
    cov = 0.0
    vxsum = 0.0
    vysum = 0.0
    for i in range(t):
        vx = x[i]
        vy = y[i]
        if np.isfinite(vx) and np.isfinite(vy):
            dx = vx-mx
            dy = vy-my
            cov += dx*dy
            vxsum += dx*dx
            vysum += dy*dy
    if vxsum==0.0 or vysum==0.0:
        return np.nan
    return cov/np.sqrt(vxsum*vysum)
