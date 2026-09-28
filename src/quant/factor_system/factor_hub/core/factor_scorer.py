import numpy as np
import pandas as pd
from typing import Callable

from .config import SCORING,_default_scoring


class FactorScorer:
    def get_scoring_func(self,universe:str,domain:str)->Callable:
        return SCORING.get((universe,domain),_default_scoring)

    def score(self,perf:dict,universe:str,domain:str)->float:
        func = self.get_scoring_func(universe,domain)
        try:
            return float(func(perf))
        except Exception:
            return np.nan

    def score_batch(self,perf_df:pd.DataFrame,registry:dict[str,dict])->pd.Series:
        scores = {}
        for name in perf_df.index:
            info = registry.get(name,{})
            universe = info.get("universe","standards")
            domain = info.get("domain","pv")
            perf = perf_df.loc[name].to_dict()
            scores[name] = self.score(perf,universe,domain)
        return pd.Series(scores,name="score")
