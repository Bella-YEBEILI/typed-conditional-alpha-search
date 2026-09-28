import numpy as np
from typing import Callable,Union

from .factor import Factor
from .parser import FormulaParser
from .id_manager import IdManager
from ..config.fld_registry import FLD_REGISTRY
from ..factory_ops.numpy_funcs import p2p_corr


DEFAULT_MARKER_CONFIG = {
    "structure":{
        "fundamental_bonus":0.5,
        "mixed_bonus":0.0,
    },
    "fitness":{
        "perf_weight":1.0,
        "structure_weight":1.0,
        "batch_corr_penalty":0.5,
        "prod_corr_penalty":0.5,
    },
    "candidate":{
        "corr_threshold":0.5,
        "score_improve":1.0,
    },
    "seed":{
        "corr_threshold":0.7,
    },
}


class ResultMarker:
    def __init__(self,parser:FormulaParser,config:dict=DEFAULT_MARKER_CONFIG):
        self.parser = parser
        self.config = config

    def mark_structure(self,factors:list[Factor]):
        for factor in factors:
            c = self.config["structure"]
            if factor.node.root is None:
                factor.node = self.parser.parse(factor.spec.formula)
            level_as = {FLD_REGISTRY.get(fld,{}).get("level_a") for fld in factor.node.flds}

            structure_score = 0.0
            if "fundamental" in level_as:
                structure_score += c["fundamental_bonus"]
            if "fundamental" in level_as and "daily_pv" in level_as:
                structure_score += c["mixed_bonus"]

            factor.result.structure_score = structure_score

    def mark_prod_corr(self,factors:list[Factor],product_factors:list[Factor]):
        for factor in factors:
            max_corr,max_id,_ = self._max_corr_factor(factor,product_factors)
            factor.result.prod_corr = max_corr

    def mark_batch_corr(self,factors:list[Factor],accepted_seeds:list[Factor]):
        for factor in factors:
            max_corr,_,_ = self._max_corr_factor(factor,accepted_seeds)
            factor.result.batch_corr = max_corr

    def mark_fitness(self,factors:list[Factor]):
        c = self.config["fitness"]
        for factor in factors:
            fitness = self._base_score(factor)
            if factor.result.batch_corr is not None:
                fitness *= max(0.0,1.0-c["batch_corr_penalty"]*abs(factor.result.batch_corr))
            if factor.result.prod_corr is not None:
                fitness *= max(0.0,1.0-c["prod_corr_penalty"]*abs(factor.result.prod_corr))
            factor.result.fitness = fitness

    def mark_candidate_prod(self,candidates:list[Factor],product_factors:list[Factor]):
        c = self.config["candidate"]
        for factor in candidates:
            factor.result.candidate_prod_prune_passed = False
            factor.result.candidate_prod_max_corr = None
            factor.result.candidate_prod_max_id = None

            max_corr,max_id,other = self._max_corr_factor(factor,product_factors)
            factor.result.prod_corr = max_corr
            factor.result.candidate_prod_max_corr = max_corr
            factor.result.candidate_prod_max_id = max_id

            if max_corr is None or abs(max_corr)<=c["corr_threshold"]:
                factor.result.candidate_prod_prune_passed = True
            elif self._base_score(factor)>=self._base_score(other)*c["score_improve"]:
                factor.result.candidate_prod_prune_passed = True

    def mark_candidate_self(self,candidates:list[Factor]):
        c = self.config["candidate"]
        pool = [factor for factor in candidates if factor.result.candidate_prod_prune_passed==True]

        self._mark_self_prune(
            pool,
            corr_threshold=c["corr_threshold"],
            score_func=self._base_score,
            passed_name="candidate_self_prune_passed",
            corr_name="candidate_self_max_corr",
            id_name="candidate_self_max_id",
        )

        for factor in candidates:
            if factor.result.candidate_prod_prune_passed!=True:
                factor.result.candidate_self_prune_passed = False
                factor.result.candidate_self_max_corr = None
                factor.result.candidate_self_max_id = None

    def mark_seed(self,seeds:list[Factor],whitelist_ops:tuple[str,...]=()):
        c = self.config["seed"]
        self._mark_self_prune(
            seeds,
            corr_threshold=c["corr_threshold"],
            score_func=self._fitness,
            passed_name="seed_prune_passed",
            corr_name="seed_max_corr",
            id_name="seed_max_id",
            whitelist_ops=whitelist_ops,
        )

    def _mark_self_prune(self,
                         factors:list[Factor],
                         corr_threshold:float,
                         score_func:Callable[[Factor],float],
                         passed_name:str,
                         corr_name:str,
                         id_name:str,
                         whitelist_ops:tuple[str,...]=()):
        for factor in factors:
            setattr(factor.result,passed_name,False)
            setattr(factor.result,corr_name,None)
            setattr(factor.result,id_name,None)

        sorted_factors = sorted(factors,key=score_func,reverse=True)
        kept = []

        for factor in sorted_factors:
            max_corr,max_id,_ = self._max_corr_factor(factor,kept)
            setattr(factor.result,corr_name,max_corr)
            setattr(factor.result,id_name,max_id)

            if max_corr is None or abs(max_corr)<=corr_threshold or self._outer_op(factor) in whitelist_ops:
                setattr(factor.result,passed_name,True)
                kept.append(factor)

    def _max_corr_factor(self,
                         factor:Factor,
                         others:list[Factor])->tuple[Union[float,None],Union[str,None],Union[Factor,None]]:
        if others is None or len(others)==0:
            return None,None,None

        spec_id = self._spec_id(factor)
        max_corr = None
        max_id = None
        max_factor = None
        for other in others:
            other_id = self._spec_id(other)
            if other_id==spec_id:
                continue
            corr = p2p_corr(factor.result.rankics,other.result.rankics)
            if not np.isfinite(corr):
                continue
            if max_corr is None or abs(corr)>abs(max_corr):
                max_corr = corr
                max_id = other_id
                max_factor = other

        return max_corr,max_id,max_factor

    def _base_score(self,factor:Factor)->float:
        c = self.config["fitness"]
        perf_score = factor.result.perf_score
        structure_score = factor.result.structure_score
        if perf_score is None:
            perf_score = 0.0
        if structure_score is None:
            structure_score = 0.0
        return perf_score*c["perf_weight"]+structure_score*c["structure_weight"]

    def _fitness(self,factor:Factor)->float:
        if factor.result.fitness is None:
            return self._base_score(factor)
        return factor.result.fitness

    def _outer_op(self,factor:Factor)->Union[str,None]:
        if factor.node.root is None:
            factor.node = self.parser.parse(factor.spec.formula)
        if factor.node.root.kind=="op":
            return factor.node.root.value
        return None

    @staticmethod
    def _spec_id(factor:Factor)->str:
        spec = factor.spec
        if spec.spec_id is not None:
            return spec.spec_id
        return IdManager.get_spec_id(spec.formula,spec.decay,spec.neutralize)
