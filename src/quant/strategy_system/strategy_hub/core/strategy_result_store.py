import os
import pickle
import pandas as pd
from typing import Any
from quant.data_system.data_hub.main import DataManager

class StrategyResultStore:
    def __init__(self,base_dir:str)->None:
        self.base_dir = base_dir
        self.root_dir = os.path.join(base_dir,"strategy_results")
        os.makedirs(self.root_dir,exist_ok=True)

    def _check_name(self,x:str,field:str)->None:
        if not isinstance(x,str) or x=="":
            raise ValueError(f"{field} must be non-empty str")

    def _strategy_dir(self,strategy_name:str)->str:
        self._check_name(strategy_name,"strategyname")
        d = os.path.join(self.root_dir,strategy_name)
        os.makedirs(d,exist_ok=True)
        return d

    def _path(self,strategy_name:str,trade_point:str)->str:
        self._check_name(trade_point,"trade_point")
        return os.path.join(self._strategy_dir(strategy_name),f"{trade_point}.pkl")

    def _validate_result(self,result:dict[str,Any])->None:
        if not isinstance(result,dict) or len(result)==0:
            raise ValueError("result must be non-empty dict[str,Any]")

        for k,v in result.items():
            if not isinstance(k,str) or k=="":
                raise ValueError("result keys must be non-empty str")
            if not isinstance(v,(pd.Series,pd.DataFrame)):
                raise ValueError(f"result[{k}] must be pd.Series or pd.DataFrame")

    def has(self,strategy_name:str,trade_point:str)->bool:
        return os.path.exists(self._path(strategy_name,trade_point))

    def save(self,strategy_name:str,trade_point:str,result:dict[str,Any])->str:
        self._validate_result(result)
        path = self._path(strategy_name,trade_point)
        DataManager.safe_to_pickle(result,path)
        return path

    def load(self,strategy_name:str,trade_point:str)->dict[str,Any]:
        path = self._path(strategy_name,trade_point)
        if not os.path.exists(path):
            raise FileNotFoundError(f"strategy result not found: {path}")

        with open(path,"rb") as f:
            result = pickle.load(f)

        self._validate_result(result)
        return result

    def delete(self,strategy_name:str,trade_point:str)->bool:
        path = self._path(strategy_name,trade_point)
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True

    def get_path(self,strategy_name:str,trade_point:str)->str:
        return self._path(strategy_name,trade_point)
    
    def list_trade_points(self,strategy_name:str):
        d = os.path.join(self.root_dir,strategy_name)
        if not os.path.isdir(d):
            return []
        return [f[:-4] for f in os.listdir(d) if f.endswith(".pkl")]
