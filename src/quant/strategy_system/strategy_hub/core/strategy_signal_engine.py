import pandas as pd
import os

from .config import STRATEGY_START_DATE,STRATEGY_END_DATE
from .strategy_loader import load_strategy_module

class StrategySignalEngine:
    def __init__(self,registry):
        self.registry = registry

    def run_by_name(self,
                    strategy_name:str,
                    start:str|None=None,
                    end:str|None=None)->pd.DataFrame:
        item = self.registry.get_strategy(strategy_name)
        if item is None:
            raise ValueError(f"strategy not found: {strategy_name}")
        return self.run_by_path(item["file_path"],start,end)

    def run_by_path(self,
                    file_path:str,
                    start:str|None=None,
                    end:str|None=None)->pd.DataFrame:
        if start is None:
            start = STRATEGY_START_DATE
        if end is None:
            end = STRATEGY_END_DATE

        m = load_strategy_module(file_path)
        signal = m.run(start=start,end=end)
        
        return signal.loc[start:end]

        


