from ..core.factor import Factor,make_factor
from ..core.evaluator import Evaluator
from ..core.machine_lib import MathFactory,TsFactory,GroupFactory
from ..core.adjuster import Adjuster
from ..core.result_marker import ResultMarker
from ..repository.seed_repository import SeedRepository
from ..log_and_time.run_recorder import RunRecorder
from ..log_and_time.all_factor_recorder import AllFactorRecorder
from .product_manager import ProductManager


DEFAULT_LAYERS = {
    1:("math","ts","group"),
    2:("math","ts","group"),
    3:("math","ts","group"),
}


class SeedMiner:
    def __init__(
        self,
        evaluator:Evaluator,
        math_factory:MathFactory,
        ts_factory:TsFactory,
        group_factory:GroupFactory,
        adjuster:Adjuster,
        marker:ResultMarker,
        seed_repository:SeedRepository,
        product_manager:ProductManager,
        layers:dict=DEFAULT_LAYERS,
        seed_prune_whitelist_ops:tuple[str,...]=(),
        last_layer_blacklist_ops:tuple[str,...]=(),
        run_recorder:RunRecorder=None,
        all_factor_recorder:AllFactorRecorder=None,
        threads:int=8,
    ):
        self.evaluator = evaluator
        self.math_factory = math_factory
        self.ts_factory = ts_factory
        self.group_factory = group_factory
        self.adjuster = adjuster
        self.marker = marker
        self.seed_repository = seed_repository
        self.product_manager = product_manager
        self.product_factors = product_manager.load_product_factors()
        self.layers = layers
        self.seed_prune_whitelist_ops = seed_prune_whitelist_ops
        self.last_layer_blacklist_ops = last_layer_blacklist_ops
        self.run_recorder = run_recorder if run_recorder is not None else RunRecorder()
        self.all_factor_recorder = all_factor_recorder if all_factor_recorder is not None else AllFactorRecorder()
        self.threads = threads
        self.accepted_seeds = {}

    def run(self,atoms:list[str],config_path=None)->list[Factor]:
        self.accepted_seeds = {}
        factories = {
            "math":self.math_factory,
            "ts":self.ts_factory,
            "group":self.group_factory,
        }

        for atom in atoms:
            started_at,start_perf = self.run_recorder.start()
            evaluated_count = 0
            seed_count = 0
            candidate_count = 0
            final_seed_count = 0
            full_product_count = 0

            self.accepted_seeds[atom] = []
            atom_tuple = (atom,)
            layer_inputs = [make_factor(atom,atom=atom_tuple,layer=0,path="",factory=None)]
            sorted_layers = sorted(self.layers)

            for layer in sorted_layers:
                new_factors = []
                for factor in layer_inputs:
                    used = set(factor.meta.path.split(">")) if factor.meta.path else set()
                    for factory_name in self.layers[layer]:
                        if factory_name in used:
                            continue
                        path = factory_name if not factor.meta.path else factor.meta.path+">"+factory_name
                        new_factors.extend(factories[factory_name].expand(
                            [factor],
                            atom=atom_tuple,
                            layer=layer,
                            path=path,
                            factory=factory_name,
                        ))
                if layer==sorted_layers[-1] and len(self.last_layer_blacklist_ops)>0:
                    new_factors = [
                        factor for factor in new_factors
                        if self._outer_op(factor) not in self.last_layer_blacklist_ops
                    ]

                raw_factors = self.evaluator.multi_evaluate(
                    new_factors,
                    self.threads,
                    progress_label=f"seed atom={atom} layer={layer} raw",
                )
                adjusted_factors = self.adjuster.adjust(raw_factors)
                adjusted_factors = self.evaluator.multi_evaluate(
                    adjusted_factors,
                    self.threads,
                    progress_label=f"seed atom={atom} layer={layer} adjusted",
                )
                evaluated = raw_factors+adjusted_factors
                evaluated_count += len(evaluated)

                seeds = [factor for factor in evaluated if factor.result.is_seed==True]
                seed_count += len(seeds)
                if len(seeds)>0:
                    self.marker.mark_structure(seeds)
                    candidates = [factor for factor in seeds if factor.result.is_candidate==True]
                    candidate_count += len(candidates)

                    if len(candidates)>0:
                        self.marker.mark_candidate_prod(candidates,self.product_factors)
                        self.marker.mark_candidate_self(candidates)
                        passed = [
                            factor for factor in candidates
                            if factor.result.candidate_self_prune_passed==True
                        ]
                        full_factors = self.product_manager.process_candidates(passed)
                        full_product_count += len(full_factors)
                        self.product_factors = self.product_manager.load_product_factors()

                    self.marker.mark_batch_corr(seeds,self.accepted_seeds[atom])
                    self.marker.mark_prod_corr(seeds,self.product_factors)
                    self.marker.mark_fitness(seeds)
                    self.marker.mark_seed(seeds,self.seed_prune_whitelist_ops)
                    self.seed_repository.save_eval_batch(seeds)

                    layer_inputs = [
                        factor for factor in seeds
                        if factor.result.seed_prune_passed==True
                    ]
                    final_seed_count += len(layer_inputs)
                    self.accepted_seeds[atom].extend(layer_inputs)
                else:
                    layer_inputs = []

                self.all_factor_recorder.save_batch("seed",evaluated)
                self.evaluator.value_engine.clear_value_cache()
                if layer!=sorted_layers[-1]:
                    self.evaluator.cache_values(
                        layer_inputs,
                        self.threads,
                        progress_label=f"seed atom={atom} layer={layer} cache",
                    )

                if len(layer_inputs)==0:
                    break

            self.run_recorder.append(
                run_type="seed",
                config_path=config_path,
                atom=atom,
                evaluated_count=evaluated_count,
                seed_count=seed_count,
                candidate_count=candidate_count,
                final_seed_count=final_seed_count,
                full_product_count=full_product_count,
                started_at=started_at,
                start_perf=start_perf,
            )

        return [
            factor
            for atom in atoms
            for factor in self.accepted_seeds[atom]
        ]

    @staticmethod
    def _outer_op(factor:Factor):
        formula = factor.spec.formula
        pos = formula.find("(")
        if pos<0:
            return None
        return formula[:pos]
