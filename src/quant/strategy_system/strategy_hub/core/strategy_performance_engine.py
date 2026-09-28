import pandas as pd
import numpy as np

SUPPORTED_BENCHMARK = ["sz50s",
                       "hs300s",
                       "zz500s",
                       "zz800s",
                       "zz1000s"]


class StrategyPerformanceEngine:
    def __init__(self,dm):
        self.dm = dm

    def _calc_rets(self,pnl:pd.Series)->pd.Series:
        rets = pnl/pnl.shift(1)-1
        rets.iloc[0] = pnl.iloc[0]-1
        return rets

    def _calc_ret(self,rets:pd.Series,annualize:bool=False)->float:
        nav = (1+rets).prod()
        if annualize:
            return float(nav**(243/len(rets))-1)
        return float(nav-1)

    def _calc_ir(self,rets:pd.Series)->float:
        std = rets.std(ddof=1)
        if pd.isna(std) or std==0:
            return 0.0
        return float(rets.mean()/std*np.sqrt(243))

    def _calc_maxdd(self,rets:pd.Series)->float:
        nav = (1+rets).cumprod()
        return float((nav/nav.cummax()-1).min())

    def _calc_metrics(self,
                      gross_rets:pd.Series,
                      rets:pd.Series,
                      benchmark_rets:pd.Series,
                      turnovers:pd.Series,
                      effs:pd.Series,
                      annualize_ret:bool)->dict[str,float]:
        ex_rets = rets-benchmark_rets
        return {
            "gross_ret":self._calc_ret(gross_rets,annualize=annualize_ret),
            "ret":self._calc_ret(rets,annualize=annualize_ret),
            "ir":self._calc_ir(rets),
            "maxdd":self._calc_maxdd(rets),
            "ex_ret":self._calc_ret(ex_rets,annualize=annualize_ret),
            "ex_ir":self._calc_ir(ex_rets),
            "ex_maxdd":self._calc_maxdd(ex_rets),
            "turnover":float(turnovers.mean()),
            "eff":float(effs.mean()),
        }

    def _calc_group_perf(self,
                         gross_rets:pd.Series,
                         rets:pd.Series,
                         benchmark_rets:pd.Series,
                         turnovers:pd.Series,
                         effs:pd.Series,
                         freq:str)->dict[str,dict[str,float]]:
        result = {}
        periods = rets.index.to_period(freq)
        for period in periods.unique():
            mask = periods==period
            result[str(period)] = self._calc_metrics(
                gross_rets=gross_rets.loc[mask],
                rets=rets.loc[mask],
                benchmark_rets=benchmark_rets.loc[mask],
                turnovers=turnovers.loc[mask],
                effs=effs.loc[mask],
                annualize_ret=False,
            )
        return result

    def calc_perf(self,
                  strategy_result:dict[str,pd.Series],
                  benchmark:str="zz1000s")->dict[str,dict]:
        if benchmark not in SUPPORTED_BENCHMARK:
            raise ValueError(f"unsupported benchmark: {benchmark}")

        gross_pnl = strategy_result["gross_pnl"].sort_index()
        pnl = strategy_result["pnl"].sort_index()
        gross_rets = self._calc_rets(gross_pnl)
        rets = self._calc_rets(pnl)
        benchmark_rets = self.dm.get_data("index_data")[benchmark+"_returns"].reindex(index=rets.index).fillna(0.0)
        turnovers = strategy_result["turnovers"].reindex(index=rets.index).fillna(0.0)
        effs = strategy_result["effs"].reindex(index=rets.index).fillna(0.0)

        return {
            "all":self._calc_metrics(
                gross_rets=gross_rets,
                rets=rets,
                benchmark_rets=benchmark_rets,
                turnovers=turnovers,
                effs=effs,
                annualize_ret=True,
            ),
            "monthly":self._calc_group_perf(
                gross_rets=gross_rets,
                rets=rets,
                benchmark_rets=benchmark_rets,
                turnovers=turnovers,
                effs=effs,
                freq="M",
            ),
            "yearly":self._calc_group_perf(
                gross_rets=gross_rets,
                rets=rets,
                benchmark_rets=benchmark_rets,
                turnovers=turnovers,
                effs=effs,
                freq="Y",
            ),
        }
