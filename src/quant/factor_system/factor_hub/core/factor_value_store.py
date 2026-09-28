import os
import pandas as pd
from quant.data_system.data_hub.main import DataManager


class FactorValueStore:
    def __init__(self,base_dir:str)->None:
        self.value_dir = os.path.join(base_dir,"factor_values")
        os.makedirs(self.value_dir,exist_ok=True)

    def _path(self,factor_name:str)->str:
        if not isinstance(factor_name,str) or factor_name=="":
            raise ValueError("factor_name must be non-empty str")
        return os.path.join(self.value_dir,f"{factor_name}.pkl")

    def has(self,factor_name:str)->bool:
        return os.path.exists(self._path(factor_name))
    
    def save(self,factor_name:str,value:pd.DataFrame)->str:
        if not isinstance(value,pd.DataFrame):
            raise ValueError("value must be DataFrame")

        path = self._path(factor_name)
        DataManager.safe_to_pickle(value,path)
        return path

    def load(self,factor_name:str)->pd.DataFrame:
        path = self._path(factor_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"factor value not found: {path}")
        data = pd.read_pickle(path)
        if not isinstance(data,pd.DataFrame):
            raise ValueError("stored factor value must be DataFrame")
        return data

    def delete(self,factor_name:str)->bool:
        path = self._path(factor_name)
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True

    def get_path(self,factor_name:str)->str:
        return self._path(factor_name)