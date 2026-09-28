from .factor import Factor,FactorMeta,FactorSpec
from .evaluator import Evaluator


DEFAULT_OS_CHECKER_CONFIG = {
    "CANDIDATE":{
        "min_rankic":0.04,
    },
}


class OSChecker:
    def __init__(self,
                 evaluator:Evaluator,
                 config:dict=DEFAULT_OS_CHECKER_CONFIG,
                 threads:int=8):
        self.evaluator = evaluator
        self.config = config
        self.threads = threads

    def check(self,factors:list[Factor])->list[Factor]:
        checked = self.evaluate(factors)
        return [factor for factor in checked if self._pass(factor)]

    def evaluate(self,factors:list[Factor])->list[Factor]:
        checked = [self._clone_factor(factor) for factor in factors]
        checked = self.evaluator.multi_evaluate(checked,self.threads)
        src_by_id = {factor.spec.spec_id:factor for factor in factors}
        for factor in checked:
            src = src_by_id.get(factor.spec.spec_id)
            if src is not None:
                factor.result.perf_score = src.result.perf_score
                factor.result.structure_score = src.result.structure_score
        return checked

    def _pass(self,factor:Factor)->bool:
        c = self.config["CANDIDATE"]
        result = factor.result
        return (
            ("min_rankic" not in c or result.rankic>=c["min_rankic"]) and
            ("min_rankicir" not in c or result.rankicir>=c["min_rankicir"]) and
            ("min_ic" not in c or result.ic>=c["min_ic"]) and
            ("min_turnover" not in c or result.turnover>=c["min_turnover"]) and
            ("max_turnover" not in c or result.turnover<=c["max_turnover"]) and
            ("min_coverage" not in c or result.coverage>=c["min_coverage"])
        )

    @staticmethod
    def _clone_factor(factor:Factor)->Factor:
        spec = FactorSpec(
            formula=factor.spec.formula,
            decay=factor.spec.decay,
            neutralize=factor.spec.neutralize,
            formula_id=factor.spec.formula_id,
            spec_id=factor.spec.spec_id,
        )
        meta = FactorMeta(
            atom=factor.meta.atom,
            layer=factor.meta.layer,
            path=factor.meta.path,
            factory=factor.meta.factory,
        )
        return Factor(meta=meta,spec=spec)


class FullChecker:
    def __init__(self,
                 evaluator:Evaluator,
                 threads:int=8):
        self.evaluator = evaluator
        self.threads = threads

    def check(self,factors:list[Factor])->list[Factor]:
        checked = [OSChecker._clone_factor(factor) for factor in factors]
        return self.evaluator.multi_evaluate(checked,self.threads)
