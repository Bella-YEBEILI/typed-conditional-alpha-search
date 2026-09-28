import random

import numpy as np

from ..core.factor import Factor,make_factor
from ..core.evaluator import Evaluator
from ..core.adjuster import Adjuster
from ..core.result_marker import ResultMarker
from ..core.id_manager import IdManager
from ..factory_ops.numpy_funcs import p2p_corr
from ..genetic.local_mutator import LocalMutator
from ..genetic.population import PopulationSelector
from ..log_and_time.all_factor_recorder import AllFactorRecorder
from .product_manager import ProductManager


class CandidateRefiner:
    def __init__(self,
                 evaluator:Evaluator,
                 adjuster:Adjuster,
                 marker:ResultMarker,
                 mutator:LocalMutator,
                 population_selector:PopulationSelector,
                 product_manager:ProductManager,
                 all_factor_recorder:AllFactorRecorder=None,
                 threads:int=8):
        self.evaluator = evaluator
        self.adjuster = adjuster
        self.marker = marker
        self.mutator = mutator
        self.population_selector = population_selector
        self.product_manager = product_manager
        self.all_factor_recorder = all_factor_recorder if all_factor_recorder is not None else AllFactorRecorder()
        self.threads = threads

    def run(self,formula:str,run_config:dict)->dict:
        rng = random.Random(run_config.get("random_seed",42))
        max_generations = run_config.get("max_generations",10)
        children_per_generation = run_config.get("children_per_generation",1000)
        corr_threshold = run_config.get("corr_threshold",0.5)

        population = [self._init_factor(formula)]
        seen_formula_ids = {IdManager.get_formula_id(formula)}
        best = None

        for generation in range(max_generations):
            product_factors = self.product_manager.load_product_factors()
            self.evaluator.value_engine.clear_value_cache()
            self.evaluator.cache_values(
                population,
                self.threads,
                progress_label=f"refine gen={generation} cache",
            )

            offspring = self.mutator.mutate_batch(
                population=population,
                count=children_per_generation,
                rng=rng,
                seen_formula_ids=seen_formula_ids,
            )
            if len(offspring)==0:
                break

            raw_factors = self.evaluator.multi_evaluate(
                offspring,
                self.threads,
                progress_label=f"refine gen={generation} raw",
            )
            adjusted_factors = self.adjuster.adjust(raw_factors)
            adjusted_factors = self.evaluator.multi_evaluate(
                adjusted_factors,
                self.threads,
                progress_label=f"refine gen={generation} adjusted",
            )
            factors = raw_factors+adjusted_factors
            if len(factors)>0:
                self.marker.mark_structure(factors)
                self._mark_prod_corr_signed(factors,product_factors)
                self.marker.mark_fitness(factors)
                self.all_factor_recorder.save_batch("refine",factors)

            candidates = [factor for factor in factors if factor.result.is_candidate==True]
            if len(candidates)>0:
                self._mark_candidate_prod_signed(candidates,product_factors,corr_threshold)
                self.marker.mark_candidate_self(candidates)
                hit_candidates = [
                    factor for factor in candidates
                    if factor.result.candidate_self_prune_passed==True
                ]
                full_factors = self.product_manager.process_candidates(hit_candidates)
                if len(full_factors)>0:
                    return self._hit_result(generation,hit_candidates,full_factors)

            best = self._best(best,factors)
            population = self.population_selector.select(
                old_population=[],
                passed_seeds=factors,
                evaluated=factors,
                rng=rng,
            )
            if len(population)==0:
                break

        return self._miss_result(best,max_generations)

    @staticmethod
    def _init_factor(formula:str)->Factor:
        factor = make_factor(
            formula,
            atom=("refine",),
            layer=None,
            path="init",
            factory=None,
            decay=0,
        )
        factor.spec.formula_id = IdManager.get_formula_id(formula)
        factor.spec.spec_id = IdManager.get_spec_id(formula,0,None)
        return factor

    def _mark_candidate_prod_signed(self,
                                    candidates:list[Factor],
                                    product_factors:list[Factor],
                                    corr_threshold:float):
        for factor in candidates:
            max_corr,max_id = self._max_signed_corr(factor,product_factors)
            factor.result.prod_corr = max_corr
            factor.result.candidate_prod_max_corr = max_corr
            factor.result.candidate_prod_max_id = max_id
            factor.result.candidate_prod_prune_passed = (
                max_corr is None or max_corr<=corr_threshold
            )

    def _mark_prod_corr_signed(self,factors:list[Factor],product_factors:list[Factor]):
        for factor in factors:
            max_corr,_ = self._max_signed_corr(factor,product_factors)
            factor.result.prod_corr = max_corr

    @staticmethod
    def _max_signed_corr(factor:Factor,others:list[Factor]):
        max_corr = None
        max_id = None
        spec_id = IdManager.get_spec_id(factor.spec.formula,factor.spec.decay,factor.spec.neutralize)
        for other in others:
            if other.spec.spec_id==spec_id:
                continue
            corr = p2p_corr(factor.result.rankics,other.result.rankics)
            if not np.isfinite(corr):
                continue
            corr = float(corr)
            if max_corr is None or corr>max_corr:
                max_corr = corr
                max_id = other.spec.spec_id
        return max_corr,max_id

    @staticmethod
    def _best(current,factors:list[Factor]):
        pool = [factor for factor in factors if factor.result.fitness is not None and np.isfinite(factor.result.fitness)]
        if len(pool)==0:
            return current
        best = max(pool,key=lambda f:f.result.fitness)
        if current is None or best.result.fitness>current.result.fitness:
            return best
        return current

    @staticmethod
    def _hit_result(generation:int,hit_candidates:list[Factor],full_factors:list[Factor])->dict:
        product_ids = [f.spec.spec_id for f in full_factors]
        factor = next(
            (f for f in hit_candidates if f.spec.spec_id in product_ids),
            hit_candidates[0],
        )
        return {
            "hit":True,
            "generation":generation,
            "product_ids":product_ids,
            "formula":factor.spec.formula,
            "rankic":factor.result.rankic,
            "prod_corr":factor.result.prod_corr,
        }

    @staticmethod
    def _miss_result(best,max_generations:int)->dict:
        if best is None:
            return {"hit":False,"generation":max_generations,"best":None}
        return {
            "hit":False,
            "generation":max_generations,
            "best":{
                "formula":best.spec.formula,
                "rankic":best.result.rankic,
                "fitness":best.result.fitness,
                "prod_corr":best.result.prod_corr,
            },
        }
