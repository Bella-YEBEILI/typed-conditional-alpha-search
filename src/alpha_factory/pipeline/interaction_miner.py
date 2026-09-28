import time
from typing import Callable

from ..core.factor import Factor
from ..core.evaluator import Evaluator
from ..core.adjuster import Adjuster
from ..core.result_marker import ResultMarker
from ..repository.seed_repository import SeedRepository
from ..repository.interaction_repository import InteractionRepository
from ..log_and_time.run_recorder import RunRecorder
from ..log_and_time.all_factor_recorder import AllFactorRecorder
from .product_manager import ProductManager


class InteractionMiner:
    def __init__(self,
                 evaluator:Evaluator,
                 adjuster:Adjuster,
                 marker:ResultMarker,
                 seed_repository:SeedRepository,
                 interaction_repository:InteractionRepository,
                 product_manager:ProductManager,
                 run_recorder:RunRecorder=None,
                 all_factor_recorder:AllFactorRecorder=None,
                 threads:int=8):
        self.evaluator = evaluator
        self.adjuster = adjuster
        self.marker = marker
        self.seed_repository = seed_repository
        self.interaction_repository = interaction_repository
        self.product_manager = product_manager
        self.product_factors = product_manager.load_product_factors()
        self.run_recorder = run_recorder if run_recorder is not None else RunRecorder()
        self.all_factor_recorder = all_factor_recorder if all_factor_recorder is not None else AllFactorRecorder()
        self.threads = threads

    def run(self,
            atoms:list[str],
            layers:list[int],
            paths:list[str],
            factories:list[str],
            strategy_func:Callable,
            strategy_params:dict,
            seed_prune_passed:bool=True,
            low_rankic=None,
            reprune:bool=False,
            limit=None,
            config_path=None,
            run_type:str="interaction")->list[Factor]:
        started_at,start_perf = self.run_recorder.start()
        seeds = self.seed_repository.load_seeds(
            atoms=atoms,
            layers=layers,
            paths=paths,
            factories=factories,
            seed_prune_passed=seed_prune_passed,
            low_rankic=low_rankic,
            limit=limit,
        )

        if reprune:
            seeds = self._reprune(seeds)

        self.product_factors = self.product_manager.load_product_factors()
        # 刷新生产相关性
        self.marker.mark_prod_corr(seeds,self.product_factors)

        self.evaluator.value_engine.clear_value_cache()
        self.evaluator.cache_values(
            seeds,
            self.threads,
            progress_label="interaction seeds cache",
        )

        done_jobs = self.interaction_repository.load_done_jobs()
        strategy_t0 = time.perf_counter()
        print("[strategy] run strategy...",flush=True)
        generated,records = strategy_func(
            factors=seeds,
            done_jobs=done_jobs,
            params=strategy_params,
        )
        print(
            f"[strategy] generated={len(generated)} records={len(records)} sec={time.perf_counter()-strategy_t0:.2f}",
            flush=True,
        )

        raw_factors = self.evaluator.multi_evaluate(
            generated,
            self.threads,
            progress_label="interaction raw",
        )
        adjusted_factors = self.adjuster.adjust(raw_factors)
        adjusted_factors = self.evaluator.multi_evaluate(
            adjusted_factors,
            self.threads,
            progress_label="interaction adjusted",
        )
        factors = raw_factors+adjusted_factors
        self.interaction_repository.save_eval_batch(factors)

        seed_count = len([factor for factor in factors if factor.result.is_seed==True])
        candidates = [factor for factor in factors if factor.result.is_candidate==True]
        self.marker.mark_structure(candidates)
        self.marker.mark_candidate_prod(candidates,self.product_factors)
        self.marker.mark_candidate_self(candidates)
        self.marker.mark_fitness(candidates)
        self.interaction_repository.save_eval_batch(candidates)
        self.all_factor_recorder.save_batch(run_type,factors)

        output_factors = [
            factor for factor in candidates
            if factor.result.candidate_self_prune_passed==True
        ]
        full_factors = self.product_manager.process_candidates(output_factors)
        self.product_factors = self.product_manager.load_product_factors()
        self.run_recorder.append(
            run_type=run_type,
            config_path=config_path,
            atom=atoms,
            evaluated_count=len(factors),
            seed_count=seed_count,
            candidate_count=len(candidates),
            final_seed_count=None,
            full_product_count=len(full_factors),
            started_at=started_at,
            start_perf=start_perf,
        )
        return output_factors

    def _reprune(self,seeds:list[Factor])->list[Factor]:
        self.marker.mark_seed(seeds)
        return [factor for factor in seeds if factor.result.seed_prune_passed==True]
