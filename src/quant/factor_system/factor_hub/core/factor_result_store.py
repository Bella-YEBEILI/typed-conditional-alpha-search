import os
import pickle
import pandas as pd
from typing import Any
from quant.data_system.data_hub.main import DataManager

class FactorResultStore:
    def __init__(self,base_dir:str)->None:
        self.base_dir = base_dir
        self.root_dir = os.path.join(base_dir,"factor_results")
        os.makedirs(self.root_dir,exist_ok=True)

    def _check_name(self,x:str,field:str)->None:
        if not isinstance(x,str) or x=="":
            raise ValueError(f"{field} must be non-empty str")

    def _factor_dir(self,factor_name:str)->str:
        self._check_name(factor_name,"factor_name")
        d = os.path.join(self.root_dir,factor_name)
        os.makedirs(d,exist_ok=True)
        return d

    def _path(self,factor_name:str,profile_id:str)->str:
        self._check_name(profile_id,"profile_id")
        return os.path.join(self._factor_dir(factor_name),f"{profile_id}.pkl")

    def _validate_result(self,result:dict[str,Any])->None:
        if not isinstance(result,dict) or len(result)==0:
            raise ValueError("result must be non-empty dict[str,Any]")

        for k,v in result.items():
            if not isinstance(k,str) or k=="":
                raise ValueError("result keys must be non-empty str")
            if not isinstance(v,(pd.Series,pd.DataFrame)):
                raise ValueError(f"result[{k}] must be pd.Series or pd.DataFrame")

    def has(self,factor_name:str,profile_id:str)->bool:
        return os.path.exists(self._path(factor_name,profile_id))

    def save(self,factor_name:str,profile_id:str,result:dict[str,Any])->str:
        self._validate_result(result)
        path = self._path(factor_name,profile_id)
        DataManager.safe_to_pickle(result,path)
        return path

    def load(self,factor_name:str,profile_id:str)->dict[str,Any]:
        path = self._path(factor_name,profile_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"factor result not found: {path}")

        with open(path,"rb") as f:
            result = pickle.load(f)

        self._validate_result(result)
        return result

    def delete(self,factor_name:str,profile_id:str)->bool:
        path = self._path(factor_name,profile_id)
        if not os.path.exists(path):
            return False
        os.remove(path)
        return True

    def get_path(self,factor_name:str,profile_id:str)->str:
        return self._path(factor_name,profile_id)
    
    def list_profile_ids(self,factor_name:str):
        d = os.path.join(self.root_dir,factor_name)
        if not os.path.isdir(d):
            return []
        return [f[:-4] for f in os.listdir(d) if f.endswith(".pkl")]
