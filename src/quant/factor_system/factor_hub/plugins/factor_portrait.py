import os
import pickle
import pandas as pd
from typing import Any

from ..core.factor_result_profiles import DEFAULT_PROFILE_ID
from ..core.factor_performance_engine import DEFAULT_PARAMS,FactorPerformanceEngine
from quant.data_system.data_hub.main import DataManager


class FactorPortraitStore:
    def __init__(self,base_dir:str)->None:
        self.portrait_dir = os.path.join(base_dir,"factor_portraits")
        os.makedirs(self.portrait_dir,exist_ok=True)

    def _path(self,factor_name:str)->str:
        if not isinstance(factor_name,str) or factor_name=="":
            raise ValueError("factor_name must be non-empty str")
        return os.path.join(self.portrait_dir,f"{factor_name}.pkl")

    def has(self,factor_name:str)->bool:
        return os.path.exists(self._path(factor_name))

    def save(self,factor_name:str,portrait:dict)->str:
        if not isinstance(portrait,dict):
            raise ValueError("portrait must be dict")
        path = self._path(factor_name)
        DataManager.safe_to_pickle(portrait,path)
        return path

    def load(self,factor_name:str)->dict:
        path = self._path(factor_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"factor portrait not found: {path}")
        with open(path,"rb") as f:
            portrait = pickle.load(f)
        if not isinstance(portrait,dict):
            raise ValueError("stored portrait must be dict")
        return portrait

    def delete(self,factor_name:str)->bool:
        path = self._path(factor_name)
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True


class FactorPortraitBuilder:
    def __init__(self,simulator,perf_engine:FactorPerformanceEngine):
        self.simulator = simulator
        self.perf_engine = perf_engine

    def _calc_periodic_perf(self,factor_result:dict[str,Any],params:dict,freq:str)->pd.DataFrame:
        rets = factor_result.get("long_rets",pd.Series(dtype=float))
        if rets.empty:
            return pd.DataFrame()
        labels = rets.index.to_period(freq)
        rows = []
        for period,group in rets.groupby(labels):
            start = group.index[0]
            end = group.index[-1]
            p_params = {**params,"start":start,"end":end}
            perf = self.perf_engine.calc_basic_performance(factor_result,p_params,annualize=False)
            perf["period_end"] = end
            rows.append(perf)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("period_end")

    def _build_section(self,factor_result:dict[str,Any],params:dict)->dict:
        return {
            "all":self.perf_engine.calc_basic_performance(factor_result,params),
            "weekly":self._calc_periodic_perf(factor_result,params,"W"),
            "monthly":self._calc_periodic_perf(factor_result,params,"M"),
            "yearly":self._calc_periodic_perf(factor_result,params,"Y"),
        }

    def build(self,
              factor_value:pd.DataFrame,
              factor_result:dict[str,Any],
              profile_id:str=DEFAULT_PROFILE_ID,
              params:dict=DEFAULT_PARAMS)->dict:
        portrait = {}
        portrait["raw"] = self._build_section(factor_result,params)

        for key,ttype,tkey in [("hs300s","subuniverse","hs300s"),
                                ("zz1000s","subuniverse","zz1000s"),
                                ("complete","neutralize","complete")]:
            sim = self.simulator.simulate_transformed(factor_value,ttype,tkey,profile_id,params)
            portrait[key] = self._build_section(sim["factor_result"],params)

        return portrait
