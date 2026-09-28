import os
import pandas as pd
from quant.data_system.data_hub.main import DataManager


class StrategySignalStore:
    def __init__(self,base_dir:str)->None:
        self.signal_dir = os.path.join(base_dir,"strategy_signals")
        os.makedirs(self.signal_dir,exist_ok=True)

    def _path(self,strategy_name:str)->str:
        if not isinstance(strategy_name,str) or strategy_name=="":
            raise ValueError("strategy_name must be non-empty str")
        return os.path.join(self.signal_dir,f"{strategy_name}.pkl")

    def has(self,strategy_name:str)->bool:
        return os.path.exists(self._path(strategy_name))
    
    def save(self,strategy_name:str,signal:pd.DataFrame)->str:
        if not isinstance(signal,pd.DataFrame):
            raise ValueError("signal must be DataFrame")

        path = self._path(strategy_name)
        DataManager.safe_to_pickle(signal,path)
        return path

    def load(self,strategy_name:str)->pd.DataFrame:
        path = self._path(strategy_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"strategy signal not found: {path}")
        data = pd.read_pickle(path)
        if not isinstance(data,pd.DataFrame):
            raise ValueError("stored strategy signal must be DataFrame")
        return data

    def delete(self,strategy_name:str)->bool:
        path = self._path(strategy_name)
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True

    def get_path(self,strategy_name:str)->str:
        return self._path(strategy_name)