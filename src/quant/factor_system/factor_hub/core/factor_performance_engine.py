import numpy as np
import pandas as pd
from typing import Any
from quant.quant_lib.analysis import cs_rank

DEFAULT_PARAMS = {
    "start":None,
    "end":None,
    "cost":0.0012,
    "trading_days":243
}


class FactorPerformanceEngine:

    @staticmethod
    def agg_rets(rets,trading_days,annualize=True):
        if rets.empty:
            return np.nan,np.nan,np.nan
        nav = (1+rets).cumprod()
        final_nav = nav.iloc[-1]
        n = len(rets)
        n_years = n/trading_days

        if final_nav<=0 or not np.isfinite(final_nav):
            ann_ret = np.nan
        else:
            ann_ret = final_nav**(1/n_years)-1

        ret = ann_ret if annualize else final_nav-1
        vol = np.sqrt(trading_days)*rets.std()
        ir = ann_ret/vol if (np.isfinite(ann_ret) and np.isfinite(vol) and vol>0) else np.nan

        dd = nav/nav.cummax()-1
        max_dd = -dd.min()
        return ret,ir,max_dd

    def calc_basic_performance(self,
                               result:dict[str,Any],
                               params:dict[str,Any]=DEFAULT_PARAMS,
                               annualize:bool=True)->dict[str,float]:
        if not isinstance(result,dict) or len(result)==0:
            raise ValueError("result must be non-empty dict[str,Any]")
        
        # params
        start = params["start"]
        end = params["end"]
        cost = params["cost"]
        trading_days = params["trading_days"]

        # series
        long_rets = result.get("long_rets",pd.Series(dtype=float)).loc[start:end]
        long_nums = result.get("long_nums",pd.Series(dtype=float)).loc[start:end]
        long_turnovers = result.get("long_turnovers",pd.Series(dtype=float)).loc[start:end]

        long_netrets = long_rets-(long_turnovers/2*cost)

        short_rets = result.get("short_rets",pd.Series(dtype=float)).loc[start:end]
        short_turnovers = result.get("short_turnovers",pd.Series(dtype=float)).loc[start:end]

        short_netrets = short_rets-(short_turnovers/2*cost)

        ls_rets = (long_rets+short_rets)/2
        ls_netrets = (long_netrets+short_netrets)/2

        ics = result.get("ics",pd.Series(dtype=float)).loc[start:end]
        rankics = result.get("rankics",pd.Series(dtype=float)).loc[start:end]
        precisions = result.get("precisions",pd.Series(dtype=float)).loc[start:end]
        coverages = result.get("coverages",pd.Series(dtype=float)).loc[start:end]
       
        # performance
        long_ret,long_ir,long_maxdd = FactorPerformanceEngine.agg_rets(long_rets,trading_days,annualize)
        long_netret,long_netir,long_netmaxdd = FactorPerformanceEngine.agg_rets(long_netrets,trading_days,annualize)
        
        ls_ret,ls_ir,ls_maxdd = FactorPerformanceEngine.agg_rets(ls_rets,trading_days,annualize)
        ls_netret,ls_netir,ls_netmaxdd = FactorPerformanceEngine.agg_rets(ls_netrets,trading_days,annualize)

        ic = ics.mean()

        rankic = rankics.mean()
        rankicir = (rankic/rankics.std()*np.sqrt(trading_days) if rankics.std()>0 else np.nan)

        long_turnover = long_turnovers.mean()
        ls_turnover = (long_turnover+short_turnovers.mean())/2

        group_rets_df = result.get("group_rets",pd.DataFrame()).loc[start:end]
        if group_rets_df.empty:
            mono = np.nan
            monoir = np.nan
        else:
            ranks = cs_rank(group_rets_df,pct=False).values
            group_ids = np.arange(ranks.shape[1],dtype=float)
            ranks_dm = ranks-ranks.mean(axis=1,keepdims=True)
            gids_dm = group_ids-group_ids.mean()
            num = ranks_dm@gids_dm
            denom = np.sqrt((ranks_dm**2).sum(axis=1)*(gids_dm**2).sum())
            monos = num/np.where(denom>0,denom,np.nan)
            if np.all(np.isnan(monos)):
                mono = np.nan
                monoir = np.nan
            else:
                mono = np.nanmean(monos)
                std = np.nanstd(monos,ddof=0)
                monoir = mono/std if std>0 else np.nan
        
        precision = precisions.mean()

        long_num = long_nums.mean()
        coverage = coverages.mean()

        return {
            "long_ret":long_ret,
            "long_ir":long_ir,
            "long_maxdd":long_maxdd,
            "long_netret":long_netret,
            "long_netir":long_netir,
            "long_netmaxdd":long_netmaxdd,
            "ls_ret":ls_ret,
            "ls_ir":ls_ir,
            "ls_maxdd":ls_maxdd,
            "ls_netret":ls_netret,
            "ls_netir":ls_netir,
            "ls_netmaxdd":ls_netmaxdd,
            "ic":ic,
            "rankic":rankic,
            "rankicir":rankicir,
            "mono":mono,
            "monoir":monoir,
            "precision":precision,
            "long_turnover":long_turnover,
            "ls_turnover":ls_turnover,
            "long_num":long_num,
            "coverage":coverage,
        }
