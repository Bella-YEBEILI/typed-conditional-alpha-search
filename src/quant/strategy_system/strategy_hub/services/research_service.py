import os
import time
import pandas as pd

from typing import Literal

from quant.data_system.data_hub.main import DataManager

from ..core.strategy_loader import load_strategy_module
from ..core.strategy_performance_engine import StrategyPerformanceEngine
from ..core.strategy_result_engine import StrategyResultEngine
from ..core.strategy_signal_engine import StrategySignalEngine
from ..core.strategy_weight_engine import StrategyWeightEngine
from ..tools.strategy_plotter import StrategyPlotter


DEFAULT_TRADE_POINT = "open_vwaps_30m"
DEFAULT_BENCHMARK = "zz1000s"
DEFAULT_BUY_FR = 0.0001
DEFAULT_SELL_FR = 0.0006


class ResearchService:
    def __init__(self,
                 dm:DataManager,
                 signal_engine:StrategySignalEngine,
                 weight_engine:StrategyWeightEngine,
                 result_engine:StrategyResultEngine,
                 performance_engine:StrategyPerformanceEngine,
                 plotter:StrategyPlotter):
        self.dm = dm
        self.signal_engine = signal_engine 
        self.weight_engine = weight_engine 
        self.result_engine = result_engine 
        self.performance_engine = performance_engine 
        self.plotter = plotter

    def py_to_signal(self,
                   file_path:str)->pd.DataFrame:
        
        if file_path.endswith(".py"):
            return self.signal_engine.run_by_path(file_path)
        raise ValueError("unsupported file type")
        
    def signal_to_weight(self,
                         signal:str|pd.DataFrame,
                         universe:str="standards",
                         method:Literal["custom","optimizer"]="custom",
                         weight_config_name:str="n100_simple_eqw",
                         start:str|None=None,
                         end:str|None=None):
        if isinstance(signal,str):
            signal = pd.read_pickle(signal)

        weight = self.weight_engine.generate_weight(signal,universe,method,weight_config_name,start,end)
        return weight

    def weight_to_result(self,
                         weight:str|pd.DataFrame,
                         trade_point:str="open_vwaps_30m",
                         buy_fr:float=0.0001,
                         sell_fr:float=0.0006,
                         start:str|None=None,
                         end:str|None=None):
        if isinstance(weight,str):
            weight = pd.read_pickle(weight)

        result = self.result_engine.calc_result(weight,trade_point,buy_fr,sell_fr,start,end)
        return result

    def result_to_performance(self,
                              result:str|dict[str,pd.Series],
                              benchmark:str="zz1000s"):
        if isinstance(result,str):
            result = pd.read_pickle(result)
        
        return self.performance_engine.calc_perf(result,benchmark)

    def evaluate_strategy(self,
                          file_path:str,
                          start:str|None=None,
                          end:str|None=None,
                          trade_point:str="open_vwaps_30m",
                          buy_fr:float=0.0001,
                          sell_fr:float=0.0006,
                          benchmark:str="zz1000s",
                          plot:bool=True,
                          )->dict:
        m = load_strategy_module(file_path)
        universe = m.SETTING["universe"]
        method = m.SETTING["method"] 
        weight_config_name = m.SETTTING["weight_config_name"]

        t1 = time.perf_counter()
        signal = self.py_to_signal(file_path)
        t2 = time.perf_counter()
        print("generate signal time: ",t2-t1)

        t1 = time.perf_counter()
        weight = self.signal_to_weight(signal,universe,method,weight_config_name,start,end)
        t2 = time.perf_counter()
        print("generate weight time: ",t2-t1)
        
        t1 = time.perf_counter()
        result = self.weight_to_result(weight,trade_point,buy_fr,sell_fr,start,end)
        t2 = time.perf_counter()
        print("generate result time: ",t2-t1)

        t1 = time.perf_counter()
        performance = self.result_to_performance(result,benchmark)
        t2 = time.perf_counter()
        print("generate performance time: ",t2-t1)

        if plot:
            self.plotter.plot_strategy(result,start,end,benchmark)

        return {
            "signal":signal,
            "weight":weight,
            "result":result,
            "performance":performance
        }


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
                        plot:bool=True)->dict:
        if isinstance(signal,str):
            signal = pd.read_pickle(signal)
        if not isinstance(signal,pd.DataFrame):
            raise ValueError("signal must be dataframe")
        
        t1 = time.perf_counter()
        weight = self.signal_to_weight(signal,universe,method,weight_config_name,start,end)
        t2 = time.perf_counter()
        print("generate weight time: ",t2-t1)
        
        t1 = time.perf_counter()
        result = self.weight_to_result(weight,trade_point,buy_fr,sell_fr,start,end)
        t2 = time.perf_counter()
        print("generate result time: ",t2-t1)

        t1 = time.perf_counter()
        performance = self.result_to_performance(result,benchmark)
        t2 = time.perf_counter()
        print("generate performance time: ",t2-t1)

        if plot:
            self.plotter.plot_strategy(result,start,end,benchmark)

        return {
            "signal":signal,
            "weight":weight,
            "result":result,
            "performance":performance
        }

    def evaluate_weight(self,
                        weight:str|pd.DataFrame,
                        start:str|None=None,
                        end:str|None=None,
                        trade_point:str="open_vwaps_30m",
                        buy_fr:float=0.0001,
                        sell_fr:float=0.0006,
                        benchmark:str="zz1000s",
                        plot:bool=True):
        if isinstance(weight,str):
            weight = pd.read_pickle(weight)
        if not isinstance(weight,pd.DataFrame):
            raise ValueError("weight must be dataframe")
        
        t1 = time.perf_counter()
        result = self.weight_to_result(weight,trade_point,buy_fr,sell_fr,start,end)
        t2 = time.perf_counter()
        print("generate result time: ",t2-t1)

        t1 = time.perf_counter()
        performance = self.result_to_performance(result,benchmark)
        t2 = time.perf_counter()
        print("generate performance time: ",t2-t1)

        if plot:
            self.plotter.plot_strategy(result,start,end,benchmark)

        return {
            "weight":weight,
            "result":result,
            "performance":performance
        }
        
