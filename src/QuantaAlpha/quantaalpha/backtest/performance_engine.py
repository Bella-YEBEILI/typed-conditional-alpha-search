from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    "start": None,
    "end": None,
    "cost": 0.0012,
    "trading_days": 243,
}


class FactorPerformanceEngine:
    @staticmethod
    def agg_rets(rets: pd.Series, trading_days: int, annualize: bool = True):
        if rets.empty:
            return np.nan, np.nan, np.nan
        nav = (1 + rets).cumprod()
        final_nav = nav.iloc[-1]
        n = len(rets)
        n_years = n / trading_days
        if n_years <= 0 or not np.isfinite(final_nav) or final_nav <= 0:
            ann_ret = np.nan
        else:
            ann_ret = final_nav ** (1 / n_years) - 1
        ret = ann_ret if annualize else final_nav - 1
        vol = np.sqrt(trading_days) * rets.std()
        ir = ann_ret / vol if vol and np.isfinite(vol) else None
        dd = nav / nav.cummax() - 1
        max_dd = -dd.min()
        return ret, ir, max_dd

    def calc_basic_performance(
        self,
        result: dict[str, Any],
        params: dict[str, Any] = DEFAULT_PARAMS,
        annualize: bool = True,
    ) -> dict[str, float]:
        if not isinstance(result, dict) or len(result) == 0:
            raise ValueError("result must be non-empty dict[str,Any]")

        start = params["start"]
        end = params["end"]
        cost = params["cost"]
        trading_days = params["trading_days"]

        long_rets = result.get("long_rets", pd.Series(dtype=float)).loc[start:end]
        long_nums = result.get("long_nums", pd.Series(dtype=float)).loc[start:end]
        long_turnovers = result.get("long_turnovers", pd.Series(dtype=float)).loc[start:end]
        long_netrets = long_rets - (long_turnovers / 2 * cost)

        short_rets = result.get("short_rets", pd.Series(dtype=float)).loc[start:end]
        short_turnovers = result.get("short_turnovers", pd.Series(dtype=float)).loc[start:end]
        short_netrets = short_rets - (short_turnovers / 2 * cost)

        ls_rets = (long_rets + short_rets) / 2
        ls_netrets = (long_netrets + short_netrets) / 2

        rankics = result.get("rankics", pd.Series(dtype=float)).loc[start:end]
        coverages = result.get("coverages", pd.Series(dtype=float)).loc[start:end]

        long_ret, long_ir, long_maxdd = FactorPerformanceEngine.agg_rets(long_rets, trading_days, annualize)
        long_netret, long_netir, long_netmaxdd = FactorPerformanceEngine.agg_rets(long_netrets, trading_days, annualize)

        ls_ret, ls_ir, ls_maxdd = FactorPerformanceEngine.agg_rets(ls_rets, trading_days, annualize)
        ls_netret, ls_netir, ls_netmaxdd = FactorPerformanceEngine.agg_rets(ls_netrets, trading_days, annualize)

        rankic = rankics.mean()
        rankicir = rankic / rankics.std() * np.sqrt(trading_days)

        long_turnover = long_turnovers.mean()
        ls_turnover = (long_turnover + short_turnovers.mean()) / 2

        long_num = long_nums.mean()
        coverage = coverages.mean()

        return {
            "long_ret": long_ret,
            "long_ir": long_ir,
            "long_maxdd": long_maxdd,
            "long_netret": long_netret,
            "long_netir": long_netir,
            "long_netmaxdd": long_netmaxdd,
            "ls_ret": ls_ret,
            "ls_ir": ls_ir,
            "ls_maxdd": ls_maxdd,
            "ls_netret": ls_netret,
            "ls_netir": ls_netir,
            "ls_netmaxdd": ls_netmaxdd,
            "rankic": rankic,
            "rankicir": rankicir,
            "long_turnover": long_turnover,
            "ls_turnover": ls_turnover,
            "long_num": long_num,
            "coverage": coverage,
        }
