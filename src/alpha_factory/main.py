import importlib
import importlib.util
import os
from pathlib import Path

from .factory_data.data_manager import DataManager
from .core.parser import FormulaParser
from .core.value_engine import ValueEngine
from .core.post_processor import PostProcessor
from .core.result_engine import ResultEngine
from .core.selector import Selector
from .core.evaluator import Evaluator
from .core.machine_lib import MathFactory,TsFactory,GroupFactory
from .core.adjuster import Adjuster
from .core.result_marker import ResultMarker
from .repository.seed_repository import SeedRepository
from .repository.interaction_repository import InteractionRepository
from .repository.product_repository import ProductRepository
from .repository.genetic_repository import GeneticRepository
from .pipeline.seed_miner import SeedMiner
from .pipeline.interaction_miner import InteractionMiner
from .pipeline.product_manager import ProductManager
from .pipeline.genetic_miner import GeneticMiner
from .pipeline.candidate_refiner import CandidateRefiner
from .core.checker import OSChecker,FullChecker
from .log_and_time.run_recorder import RunRecorder
from .log_and_time.all_factor_recorder import AllFactorRecorder
from .genetic.context import GeneticContext
from .genetic.mutator import GeneticMutator
from .genetic.local_mutator import LocalMutator
from .genetic.population import PopulationSelector


class AlphaFactory:
    def __init__(self):
        self.dm = DataManager()
        self.parser = FormulaParser(dm=self.dm)
        self.value_engine = ValueEngine(dm=self.dm)
        self.post_processor = PostProcessor(dm=self.dm)
        self.is_result_engine = ResultEngine(dm=self.dm,period="is")
        self.os_result_engine = ResultEngine(dm=self.dm,period="os")
        self.full_result_engine = ResultEngine(dm=self.dm,period="full")
        self.seed_repository = SeedRepository()
        self.interaction_repository = InteractionRepository()
        self.product_repository = ProductRepository()
        self.genetic_repository = GeneticRepository()
        self.run_recorder = RunRecorder()
        self.all_factor_recorder = AllFactorRecorder()

    def run_seed(self,config_path:str):
        self._print_pid("seed")
        config = self.load_config(config_path)
        marker = ResultMarker(self.parser,config=config["marker"])
        seed_search = config.get("seed_search",{})

        is_evaluator = self._make_evaluator(self.is_result_engine,Selector(config["selector"]))
        os_evaluator = self._make_evaluator(self.os_result_engine,Selector(config["os_checker"]))
        full_evaluator = self._make_evaluator(self.full_result_engine,Selector(config["os_checker"]))
        product_manager = self._make_product_manager(
            is_evaluator=is_evaluator,
            os_evaluator=os_evaluator,
            full_evaluator=full_evaluator,
            marker=marker,
            config=config,
        )

        miner = SeedMiner(
            evaluator=is_evaluator,
            math_factory=MathFactory(config["factories"]["math"]),
            ts_factory=TsFactory(config["factories"]["ts"]),
            group_factory=GroupFactory(config["factories"]["group"]),
            adjuster=Adjuster(config["adjuster"]),
            marker=marker,
            seed_repository=self.seed_repository,
            product_manager=product_manager,
            layers=config["layers"],
            seed_prune_whitelist_ops=tuple(seed_search.get("seed_prune_whitelist_ops",())),
            last_layer_blacklist_ops=tuple(seed_search.get("last_layer_blacklist_ops",())),
            run_recorder=self.run_recorder,
            all_factor_recorder=self.all_factor_recorder,
            threads=config["threads"],
        )

        return miner.run(
            config["atoms"],
            config_path=str(Path(config_path).resolve()),
        )

    def run_interaction(self,config_path:str):
        self._print_pid("interaction")
        config = self.load_config(config_path)
        strategy_func = self.load_strategy(
            config["strategy"]["module"],
            config["strategy"].get("func","generate"),
        )
        marker = ResultMarker(self.parser,config=config["marker"])

        is_evaluator = self._make_evaluator(self.is_result_engine,Selector(config["selector"]))
        os_evaluator = self._make_evaluator(self.os_result_engine,Selector(config["os_checker"]))
        full_evaluator = self._make_evaluator(self.full_result_engine,Selector(config["os_checker"]))
        product_manager = self._make_product_manager(
            is_evaluator=is_evaluator,
            os_evaluator=os_evaluator,
            full_evaluator=full_evaluator,
            marker=marker,
            config=config,
        )

        miner = InteractionMiner(
            evaluator=is_evaluator,
            adjuster=Adjuster(config["adjuster"]) if config.get("adjuster") else Adjuster(),
            marker=marker,
            seed_repository=self.seed_repository,
            interaction_repository=self.interaction_repository,
            product_manager=product_manager,
            run_recorder=self.run_recorder,
            all_factor_recorder=self.all_factor_recorder,
            threads=config["threads"],
        )
        source = config["source"]
        run_type = self._interaction_run_type(config["strategy"]["module"])
        atoms = config["seeds"]
        if run_type=="interaction_condition":
            output_factors = []
            for atom in atoms:
                output_factors.extend(miner.run(
                    atoms=[atom],
                    layers=source.get("layers"),
                    paths=source.get("paths"),
                    factories=source.get("factories"),
                    strategy_func=strategy_func,
                    strategy_params=config["strategy"]["params"],
                    seed_prune_passed=source.get("seed_prune_passed",True),
                    low_rankic=source.get("low_rankic"),
                    reprune=bool(source.get("reprune",False)),
                    limit=source.get("limit"),
                    config_path=str(Path(config_path).resolve()),
                    run_type=run_type,
                ))
            return output_factors

        return miner.run(
            atoms=atoms,
            layers=source.get("layers"),
            paths=source.get("paths"),
            factories=source.get("factories"),
            strategy_func=strategy_func,
            strategy_params=config["strategy"]["params"],
            seed_prune_passed=source.get("seed_prune_passed",True),
            low_rankic=source.get("low_rankic"),
            reprune=bool(source.get("reprune",False)),
            limit=source.get("limit"),
            config_path=str(Path(config_path).resolve()),
            run_type=run_type,
        )

    def run_genetic(self,config_path:str):
        self._print_pid("genetic")
        config = self.load_config(config_path)
        marker = ResultMarker(self.parser,config=config["marker"])
        is_evaluator = self._make_evaluator(self.is_result_engine,Selector(config["selector"]))
        os_evaluator = self._make_evaluator(self.os_result_engine,Selector(config["os_checker"]))
        full_evaluator = self._make_evaluator(self.full_result_engine,Selector(config["os_checker"]))
        product_manager = self._make_product_manager(
            is_evaluator=is_evaluator,
            os_evaluator=os_evaluator,
            full_evaluator=full_evaluator,
            marker=marker,
            config=config,
        )
        context = GeneticContext(
            parser=self.parser,
            config=config["genetic"],
        )
        miner = GeneticMiner(
            evaluator=is_evaluator,
            adjuster=Adjuster(config["adjuster"]),
            marker=marker,
            mutator=GeneticMutator(context,config["genetic"]),
            population_selector=PopulationSelector(config["population"]),
            genetic_repository=self.genetic_repository,
            product_manager=product_manager,
            all_factor_recorder=self.all_factor_recorder,
            threads=config["threads"],
        )
        return miner.run(
            config_path=str(Path(config_path).resolve()),
            atoms=config["source"]["atoms"],
            run_config=config["run"],
        )

    def run_refine(self,formula:str,config_path:str):
        self._print_pid("refine")
        config = self.load_config(config_path)
        marker = ResultMarker(self.parser,config=config["marker"])
        is_evaluator = self._make_evaluator(self.is_result_engine,Selector(config["selector"]))
        os_evaluator = self._make_evaluator(self.os_result_engine,Selector(config["os_checker"]))
        full_evaluator = self._make_evaluator(self.full_result_engine,Selector(config["os_checker"]))
        product_manager = self._make_product_manager(
            is_evaluator=is_evaluator,
            os_evaluator=os_evaluator,
            full_evaluator=full_evaluator,
            marker=marker,
            config=config,
        )
        context = GeneticContext(
            parser=self.parser,
            config=config["refine"],
        )
        refiner = CandidateRefiner(
            evaluator=is_evaluator,
            adjuster=Adjuster(config["adjuster"]),
            marker=marker,
            mutator=LocalMutator(context,config["refine"]),
            population_selector=PopulationSelector(config["population"]),
            product_manager=product_manager,
            all_factor_recorder=self.all_factor_recorder,
            threads=config["threads"],
        )
        return refiner.run(
            formula=formula,
            run_config=config["refine"],
        )

    def check_submission(self,target,config_path=None)->dict:
        product_manager = self._make_manual_product_manager(config_path)
        return product_manager.check_submission(target)

    def submit(self,target,config_path=None)->dict:
        product_manager = self._make_manual_product_manager(config_path)
        return product_manager.submit(target)

    def delete(self,spec_id:str,config_path=None)->dict:
        product_manager = self._make_manual_product_manager(config_path)
        return product_manager.delete(spec_id)

    def optimize_product_exact(self,corr_threshold:float=0.5,apply:bool=False,config_path=None)->dict:
        product_manager = self._make_manual_product_manager(config_path)
        return product_manager.optimize_product_exact(
            corr_threshold=corr_threshold,
            apply=apply,
        )

    def load_seeds(self,
                   atoms:list[str],
                   layers:list[int]=None,
                   paths:list[str]=None,
                   factories:list[str]=None,
                   seed_prune_passed:bool=True,
                   low_rankic=None,
                   limit=None):
        return self.seed_repository.load_seeds(
            atoms=atoms,
            layers=layers,
            paths=paths,
            factories=factories,
            seed_prune_passed=seed_prune_passed,
            low_rankic=low_rankic,
            limit=limit,
        )

    def load_seeds_from_config(self,config_path:str,summary:bool=False):
        config = self.load_config(config_path)
        source = config["source"]
        atoms = config["seeds"]
        seeds = self.load_seeds(
            atoms=atoms,
            layers=source.get("layers"),
            paths=source.get("paths"),
            factories=source.get("factories"),
            seed_prune_passed=source.get("seed_prune_passed",True),
            low_rankic=source.get("low_rankic"),
            limit=source.get("limit"),
        )
        if not summary:
            return seeds

        strategy_func = self.load_strategy(
            config["strategy"]["module"],
            config["strategy"].get("func","generate"),
        )
        done_jobs = self.interaction_repository.load_done_jobs()
        generated,_ = strategy_func(
            factors=seeds,
            done_jobs=done_jobs,
            params=config["strategy"]["params"],
        )
        unlimited_params = self._remove_strategy_limits(config["strategy"]["params"])
        unlimited_generated,_ = strategy_func(
            factors=seeds,
            done_jobs=done_jobs,
            params=unlimited_params,
        )
        return {
            "seeds":seeds,
            "seed_count":len(seeds),
            "done_job_count":len(done_jobs),
            "generated_count":len(generated),
            "unlimited_generated_count":len(unlimited_generated),
        }

    def _remove_strategy_limits(self,value):
        if isinstance(value,dict):
            out = {}
            for k,v in value.items():
                if k=="limit":
                    out[k] = None
                else:
                    out[k] = self._remove_strategy_limits(v)
            return out
        if isinstance(value,list):
            return [self._remove_strategy_limits(v) for v in value]
        if isinstance(value,tuple):
            return tuple(self._remove_strategy_limits(v) for v in value)
        return value

    @staticmethod
    def _print_pid(run_name:str):
        print(f"[run] {run_name} pid={os.getpid()}",flush=True)

    def _interaction_run_type(self,module_name:str):
        name = module_name.split(".")[-1]
        if "condition" in name:
            return "interaction_condition"
        if "cross_single" in name:
            return "cross_single"
        if "cross_multi" in name:
            return "cross_multi"
        return "interaction"

    def _make_evaluator(self,result_engine:ResultEngine,selector:Selector)->Evaluator:
        return Evaluator(
            parser=self.parser,
            value_engine=self.value_engine,
            post_processor=self.post_processor,
            result_engine=result_engine,
            selector=selector,
        )

    def _make_product_manager(self,
                              is_evaluator:Evaluator,
                              os_evaluator:Evaluator,
                              full_evaluator:Evaluator,
                              marker:ResultMarker,
                              config:dict)->ProductManager:
        threads = config["threads"]
        return ProductManager(
            is_evaluator=is_evaluator,
            os_checker=OSChecker(
                evaluator=os_evaluator,
                config=config["os_checker"],
                threads=threads,
            ),
            full_checker=FullChecker(
                evaluator=full_evaluator,
                threads=threads,
            ),
            marker=marker,
            product_repository=self.product_repository,
            threads=threads,
        )

    def _make_manual_product_manager(self,config_path=None)->ProductManager:
        if config_path is None:
            config_path = Path(__file__).resolve().parent/"config"/"seed.py"
        config = self.load_config(str(config_path))
        marker = ResultMarker(self.parser,config=config["marker"])
        is_evaluator = self._make_evaluator(self.is_result_engine,Selector(config["selector"]))
        os_evaluator = self._make_evaluator(self.os_result_engine,Selector(config["os_checker"]))
        full_evaluator = self._make_evaluator(self.full_result_engine,Selector(config["os_checker"]))
        return self._make_product_manager(
            is_evaluator=is_evaluator,
            os_evaluator=os_evaluator,
            full_evaluator=full_evaluator,
            marker=marker,
            config=config,
        )

    def load_strategy(self,module_name:str,func_name:str):
        if "." not in module_name:
            module_name = "alpha_factory.interaction_strategy."+module_name
        module = importlib.import_module(module_name)
        return getattr(module,func_name)

    def load_config(self,config_path:str)->dict:
        path = Path(config_path)
        spec = importlib.util.spec_from_file_location("alpha_factory_config",path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.CONFIG
