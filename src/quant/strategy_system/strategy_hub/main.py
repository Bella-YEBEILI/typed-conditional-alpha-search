import pandas as pd
import os
from typing import Literal

from quant.data_system.data_hub.main import DataManager

from .core.strategy_registry import StrategyRegistry
from .core.strategy_signal_engine import StrategySignalEngine
from .core.strategy_signal_store import StrategySignalStore
from .core.strategy_weight_engine import StrategyWeightEngine
from .core.strategy_weight_store import StrategyWeightStore
from .core.strategy_result_engine import StrategyResultEngine
from .core.strategy_result_store import StrategyResultStore
from .core.strategy_performance_engine import StrategyPerformanceEngine

from .services.access_service import AccessService
from .services.manage_service import ManageService
from .services.research_service import ResearchService

from .tools.strategy_plotter import StrategyPlotter


class StrategyManager:
    def __init__(self,dm=None,base_dir="/home/workspace/common/quant/quant_strategy"):
        dm = DataManager() if dm is None else dm
        self.dm = dm

        registry_path = os.path.join(base_dir,"strategy_registry.json")
        self.registry = StrategyRegistry(registry_path)

        self.signal_engine = StrategySignalEngine(self.registry)
        self.signal_store = StrategySignalStore(base_dir)
        self.weight_engine = StrategyWeightEngine(self.dm,base_dir)
        self.weight_store = StrategyWeightStore(base_dir)
        self.result_engine = StrategyResultEngine(dm)
        self.result_store = StrategyResultStore(base_dir)
        self.performance_engine = StrategyPerformanceEngine(dm)
        self.plotter = StrategyPlotter(dm=dm)

        self.access = AccessService()
        self.manage = ManageService(
            self.registry,
            self.signal_engine,
            self.signal_store,
            self.weight_engine,
            self.weight_store,
            self.result_engine,
            self.result_store,
            base_dir
        )
        self.research = ResearchService(
            self.dm,
            self.signal_engine,
            self.weight_engine,
            self.result_engine,
            self.performance_engine,
            self.plotter
        )

    # access


    # manage
    def submit(self,file_path:str)->dict:
        return self.manage.submit(file_path)
    
    # research
    def evaluate_strategy(self,
                          file_path:str,
                          start:str|None=None,
                          end:str|None=None,
                          trade_point:str="open_vwaps_30m",
                          buy_fr:float=0.0001,
                          sell_fr:float=0.0006,
                          benchmark:str="zz1000s",
                          plot=True):
        return self.research.evaluate_strategy(file_path,start,end,trade_point,buy_fr,sell_fr,benchmark,plot)

    def evaluate_signal(self,
                        signal:str|pd.DataFrame,
                        universe:str="standards",
                        method:Literal["custom","optimizer"]="custom",
                        weight_config_name:str="n100_simple_eqw",
                        start:str|None=None,
                        end:str|None=None,
                        trade_point:str="open_vwaps_30m",
                        buy_fr:float=0.0001,
                        sell_fr:float=0.0006,
                        benchmark:str="zz1000s",
                        plot=True):
        return self.research.evaluate_signal(signal,universe,method,weight_config_name,start,end,trade_point,buy_fr,sell_fr,benchmark,plot)

    def evaluate_weight(self,
                        weight:str|pd.DataFrame,
                        start:str|None=None,
                        end:str|None=None,
                        trade_point:str="open_vwaps_30m",
                        buy_fr:float=0.0001,
                        sell_fr:float=0.0006,
                        benchmark:str="zz1000s",
                        plot=True):
        return self.research.evaluate_weight(weight,start,end,trade_point,buy_fr,sell_fr,benchmark,plot)

