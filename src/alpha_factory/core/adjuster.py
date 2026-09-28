from .factor import Factor,make_factor


DEFAULT_ADJUSTER_CONFIG = {
    "min_abs_rankic":0.01,
    "turnover_decay":[
        (0.5,1.0,5),
        (1.0,1.5,10),
        (1.5,2.0,20),
    ],
}


class Adjuster:
    def __init__(self,config:dict=DEFAULT_ADJUSTER_CONFIG):
        self.config = config

    def adjust(self,factors:list[Factor])->list[Factor]:
        adjusted = []

        for factor in factors:
            rankic = factor.result.rankic
            turnover = factor.result.turnover
            formula = factor.spec.formula
            adjust = False

            if abs(rankic)<self.config["min_abs_rankic"]:
                continue

            if rankic<0.0:
                formula = f"neg({formula})"
                adjust = True

            decay = 0
            for lower,upper,d in self.config["turnover_decay"]:
                if turnover>lower and turnover<=upper:
                    decay = d
                    adjust = True
                    break

            if adjust:
                child = make_factor(
                    formula,
                    atom=factor.meta.atom,
                    layer=factor.meta.layer,
                    path=factor.meta.path,
                    factory=factor.meta.factory,
                    decay=decay,
                )
                self._copy_attrs(factor,child)
                adjusted.append(child)

        return adjusted

    @staticmethod
    def _copy_attrs(src:Factor,dst:Factor):
        for name in (
            "interaction_path",
            "genetic_parent_spec_id",
            "genetic_mutation_type",
            "refine_mutation_type",
        ):
            if hasattr(src,name):
                setattr(dst,name,getattr(src,name))
