from itertools import product
from .factor import Factor,make_factor


WINDOWS = [5,20,60,120,240]

CUT_FLDS = ["ret1","resret","to1","vol5","amplitude","pos_a","illiq1"]

GROUPS = ["sectors","industrys","subindustrys",
          "ret5_g5","vol5_g5","Size_g10","Mom_g10","Vol_g10","Liq_g10","Beta_g10",
          "Bp_g3","OQuality_g3","CQuality_g3","Growth_g3","Improve_g3","Leverage_g3"]

DEFAULT_MATH_CONFIG = {
    "abs":[],
    "power":[],
    "sqrt":[],
    "log":[],
    "diff":[],
    "diff_q":[],
    "tanh":[],
    "midmap":[],
    "rightmap":[],
    "leftmap":[],
    "extreme_leftmap":[],
    "extreme_rightmap":[],
}

DEFAULT_TS_CONFIG = {
    "ts_delta":[WINDOWS],
    "ts_pct":[WINDOWS],
    "ts_rank":[WINDOWS],
    "ts_decay_linear":[WINDOWS],
    "ts_mean":[WINDOWS],
    "ts_std":[WINDOWS],
    "ts_skew":[WINDOWS],
    "ts_kur":[WINDOWS],
    "ts_zscore":[WINDOWS],
    "ts_range":[WINDOWS],
    "ts_median":[WINDOWS],
    "ts_argmax":[WINDOWS],
    "ts_argmin":[WINDOWS],
    "ts_min":[WINDOWS],
    "ts_ir":[WINDOWS],
    "ts_cut_diff":[CUT_FLDS,WINDOWS],
    "ts_cut_normdiff":[CUT_FLDS,WINDOWS],
}

DEFAULT_GROUP_CONFIG = {
    "group_rank":[GROUPS],
    "group_midmap":[GROUPS],
    "group_leftmap":[GROUPS],
    "group_rightmap":[GROUPS],
    "group_neutralize":[GROUPS],
    "group_zscore":[GROUPS],
    "group_scale":[GROUPS],
}


class BaseFactory:
    def __init__(self,config:dict):
        self.config = config

    def expand(self,
               factors:list[Factor],
               atom:tuple[str,...],
               layer:int,
               path:str,
               factory:str)->list[Factor]:
        results = []

        for factor in factors:
            self_formula = f"{factory}_self({factor.spec.formula})"
            results.append(make_factor(
                self_formula,
                atom=atom,
                layer=layer,
                path=path,
                factory=factory,
            ))
            for op,axes in self.config.items():
                for params in product(*axes):
                    args = ",".join([str(p) for p in params])
                    suffix = ","+args if args else ""
                    formula = f"{op}({factor.spec.formula}{suffix})"
                    results.append(make_factor(
                        formula,
                        atom=atom,
                        layer=layer,
                        path=path,
                        factory=factory,
                    ))

        return results


class MathFactory(BaseFactory):
    def __init__(self,config=DEFAULT_MATH_CONFIG):
        super().__init__(config)


class TsFactory(BaseFactory):
    def __init__(self,config=DEFAULT_TS_CONFIG):
        super().__init__(config)


class GroupFactory(BaseFactory):
    def __init__(self,config=DEFAULT_GROUP_CONFIG):
        super().__init__(config)
