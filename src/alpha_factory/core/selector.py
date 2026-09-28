from .factor import FactorResult
import numpy as np


DEFAULT_SELECTOR_CONFIG = {
    "CANDIDATE":{
        "min_rankic":0.045,
        "min_rankicir":0.4,
        "min_ic":0.025,
        "min_turnover":0.01,
        "max_turnover":1.0,
        "min_coverage":0.7,
    },
    "SEED":{
        "min_rankic":0.02,
        "min_coverage":0.7,
    },
    "PERF_SCORE":{
        "rankic":10.0,
        "rankicir":1.0,
    },
}


class Selector:
    def __init__(self,config=DEFAULT_SELECTOR_CONFIG):
        self.config = config

    def select(self,result:FactorResult):
        result.perf_score = self.perf_score(result)
        result.is_candidate = self.is_candidate(result)
        result.is_seed = result.is_candidate or self.is_seed(result)
        return result

    def is_candidate(self,result:FactorResult):
        return self._pass(result,self.config["CANDIDATE"])

    def is_seed(self,result:FactorResult):
        c = self.config.get("SEED",self.config.get("TOPROMOTE"))
        return c is not None and self._pass(result,c)

    def perf_score(self,result:FactorResult):
        c = self.config.get("PERF_SCORE")
        if c is not None:
            return result.rankic*c["rankic"]+result.rankicir*c["rankicir"]
        return np.nan

    def _pass(self,result:FactorResult,c:dict):
        return (
            ("min_rankic" not in c or result.rankic>=c["min_rankic"]) and
            ("min_rankicir" not in c or result.rankicir>=c["min_rankicir"]) and
            ("min_ic" not in c or result.ic>=c["min_ic"]) and
            ("min_turnover" not in c or result.turnover>=c["min_turnover"]) and
            ("max_turnover" not in c or result.turnover<=c["max_turnover"]) and
            ("min_coverage" not in c or result.coverage>=c["min_coverage"])
        )
