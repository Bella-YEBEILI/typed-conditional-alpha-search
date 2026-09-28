import ctypes
import gc
import os
import random
import time
from datetime import datetime

import numpy as np

from ..core.factor import Factor,make_factor
from ..core.evaluator import Evaluator
from ..core.adjuster import Adjuster
from ..core.result_marker import ResultMarker
from ..core.id_manager import IdManager
from ..factory_ops.numpy_funcs import p2p_corr
from ..genetic.mutator import GeneticMutator
from ..genetic.population import PopulationSelector
from ..repository.genetic_repository import GeneticRepository
from ..log_and_time.all_factor_recorder import AllFactorRecorder
from .product_manager import ProductManager


class GeneticMiner:
    def __init__(self,
                 evaluator:Evaluator,
                 adjuster:Adjuster,
                 marker:ResultMarker,
                 mutator:GeneticMutator,
                 population_selector:PopulationSelector,
                 genetic_repository:GeneticRepository,
                 product_manager:ProductManager,
                 all_factor_recorder:AllFactorRecorder=None,
                 threads:int=8):
        self.evaluator = evaluator
        self.adjuster = adjuster
        self.marker = marker
        self.mutator = mutator
        self.population_selector = population_selector
        self.genetic_repository = genetic_repository
        self.product_manager = product_manager
        self.all_factor_recorder = all_factor_recorder if all_factor_recorder is not None else AllFactorRecorder()
        self.threads = threads

    def run(self,
            config_path:str,
            atoms:list[str],
            run_config:dict)->dict:
        run_id = self.genetic_repository.new_run_id()
        rng = random.Random(run_config.get("random_seed",42))
        max_generations = run_config.get("max_generations",20)
        children_per_generation = run_config.get("children_per_generation",1000)
        stop_when_product_hit = run_config.get("stop_when_product_hit",True)

        population = self._init_population(atoms)
        accepted_seed_pool = []
        seen_formula_ids = {IdManager.get_formula_id(f.spec.formula) for f in population}
        product_ids = []
        total_evaluated = 0
        total_seed = 0
        total_candidate = 0
        stopped_reason = "max_generations"

        for generation in range(max_generations):
            started_at = datetime.now()
            t0 = time.perf_counter()
            product_factors = self.product_manager.load_product_factors()

            self.evaluator.value_engine.clear_value_cache()
            self._release_memory()
            self.evaluator.cache_values(
                population,
                self.threads,
                progress_label=f"genetic gen={generation} cache",
            )

            offspring = self.mutator.mutate_batch(
                population=population,
                count=children_per_generation,
                rng=rng,
                seen_formula_ids=seen_formula_ids,
            )
            if len(offspring)==0:
                stopped_reason = "empty_offspring"
                self.evaluator.value_engine.clear_value_cache()
                self._release_memory()
                break
            raw_factors = self.evaluator.multi_evaluate(
                offspring,
                self.threads,
                progress_label=f"genetic gen={generation} raw",
            )
            adjusted_factors = self.adjuster.adjust(raw_factors)
            adjusted_factors = self.evaluator.multi_evaluate(
                adjusted_factors,
                self.threads,
                progress_label=f"genetic gen={generation} adjusted",
            )

            factors = raw_factors+adjusted_factors
            total_evaluated += len(factors)
            if len(factors)>0:
                self.marker.mark_structure(factors)
                self._mark_prod_corr_signed(factors,product_factors)
                self.marker.mark_fitness(factors)

            candidates = [factor for factor in factors if factor.result.is_candidate==True]
            total_candidate += len(candidates)
            candidate_prod_passed = []
            candidate_self_passed = []
            full_factors = []

            if len(candidates)>0:
                self._mark_candidate_prod_signed(candidates,product_factors)
                self.marker.mark_candidate_self(candidates)
                candidate_prod_passed = [
                    factor for factor in candidates
                    if factor.result.candidate_prod_prune_passed==True
                ]
                candidate_self_passed = [
                    factor for factor in candidates
                    if factor.result.candidate_self_prune_passed==True
                ]
                full_factors = self.product_manager.process_candidates(candidate_self_passed)
                product_ids.extend([factor.spec.spec_id for factor in full_factors])

            seeds = [factor for factor in factors if factor.result.is_seed==True]
            total_seed += len(seeds)
            passed_seeds = []
            if len(seeds)>0:
                self.marker.mark_structure(seeds)
                self.marker.mark_batch_corr(seeds,accepted_seed_pool)
                self.marker.mark_fitness(seeds)
                self.marker.mark_seed(seeds)
                passed_seeds = [
                    factor for factor in seeds
                    if factor.result.seed_prune_passed==True
                ]
                accepted_seed_pool.extend(passed_seeds)
                self.genetic_repository.save_seed_batch(run_id,generation,seeds)

            self.all_factor_recorder.save_batch("genetic",factors)
            best = self._best_factor(seeds)
            hit = len(full_factors)>0
            if hit:
                stopped_reason = "product_hit"

            self.genetic_repository.append_run_log(
                run_id=run_id,
                config_path=config_path,
                generation=generation,
                population_count=len(population),
                offspring_count=len(offspring),
                evaluated_count=len(factors),
                seed_count=len(seeds),
                passed_seed_count=len(passed_seeds),
                candidate_count=len(candidates),
                candidate_prod_passed_count=len(candidate_prod_passed),
                candidate_self_passed_count=len(candidate_self_passed),
                product_count=len(full_factors),
                best_spec_id=best.spec.spec_id if best is not None else None,
                best_fitness=best.result.fitness if best is not None else None,
                best_rankic=best.result.rankic if best is not None else None,
                started_at=started_at,
                total_sec=time.perf_counter()-t0,
                stopped_reason=stopped_reason if hit or generation==max_generations-1 else None,
            )

            if hit and stop_when_product_hit:
                self.evaluator.value_engine.clear_value_cache()
                del offspring,raw_factors,adjusted_factors,factors,candidates
                del candidate_prod_passed,candidate_self_passed,full_factors,seeds,passed_seeds,product_factors
                self._release_memory()
                break

            next_population = self.population_selector.select(
                old_population=population,
                passed_seeds=passed_seeds,
                evaluated=factors,
                rng=rng,
            )
            self.evaluator.value_engine.clear_value_cache()
            del offspring,raw_factors,adjusted_factors,factors,candidates
            del candidate_prod_passed,candidate_self_passed,full_factors,seeds,passed_seeds,product_factors
            self._release_memory()

            population = next_population
            if len(population)==0:
                stopped_reason = "empty_population"
                break

        return {
            "run_id":run_id,
            "hit":len(product_ids)>0,
            "product_ids":product_ids,
            "stopped_reason":stopped_reason,
            "evaluated_count":total_evaluated,
            "seed_count":total_seed,
            "candidate_count":total_candidate,
        }

    @staticmethod
    def _init_population(atoms:list[str])->list[Factor]:
        population = []
        for atom in atoms:
            factor = make_factor(
                atom,
                atom=("genetic",),
                layer=0,
                path="init",
                factory=None,
                decay=0,
            )
            factor.genetic_parent_spec_id = None
            factor.genetic_mutation_type = "init"
            population.append(factor)
        return population

    def _mark_candidate_prod_signed(self,candidates:list[Factor],product_factors:list[Factor]):
        threshold = self.marker.config["candidate"]["corr_threshold"]
        for factor in candidates:
            max_corr,max_id = self._max_signed_corr(factor,product_factors)
            factor.result.prod_corr = max_corr
            factor.result.candidate_prod_max_corr = max_corr
            factor.result.candidate_prod_max_id = max_id
            factor.result.candidate_prod_prune_passed = (
                max_corr is None or max_corr<=threshold
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
    def _best_factor(factors:list[Factor]):
        if len(factors)==0:
            return None
        return max(factors,key=lambda f:-np.inf if f.result.fitness is None else f.result.fitness)

    @staticmethod
    def _release_memory():
        gc.collect()
        if os.name=="posix":
            ctypes.CDLL("libc.so.6").malloc_trim(0)
