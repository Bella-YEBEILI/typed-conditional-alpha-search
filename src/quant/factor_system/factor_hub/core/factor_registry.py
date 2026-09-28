import json
import os
from datetime import datetime
from .factor_loader import load_factor_module

class FactorRegistry:
    def __init__(self,registry_path:str)->None:
        self.registry_path = registry_path
        self._ensure_registry_file()

    def _ensure_registry_file(self)->None:
        folder = os.path.dirname(self.registry_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder,exist_ok=True)
        if not os.path.exists(self.registry_path) or os.path.getsize(self.registry_path)==0:
            with open(self.registry_path,"w",encoding="utf-8") as f:
                json.dump({},f,ensure_ascii=False,indent=2)

    def _read_all(self)->dict:
        with open(self.registry_path,"r",encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data,dict):
            raise ValueError("registry file must contain a dict")
        for factor_name,item in data.items():
            if not isinstance(item,dict):
                raise ValueError("registry item must be dict")
        return data

    def _write_all(self,data:dict)->None:
        with open(self.registry_path,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2)

    def register_factor(self,file_path:str)->dict:
        m = load_factor_module(file_path)
        meta = m.META

        factor_name = meta.get("factor_name")
        if not factor_name:
            raise ValueError("META.factor_name is required")

        all_data = self._read_all()
        if factor_name in all_data:
            raise ValueError(f"factor already exists: {factor_name}")
        
        all_data[factor_name] = {
            "factor_name":factor_name,
            "type":m.TYPE,
            "author":meta.get("author"),
            "level":meta.get("level"),
            "tag":meta.get("tag",""),
            "category":meta.get("category","unknown"),
            "domain":str(meta.get("domain","pv")),
            "file_path":os.path.abspath(file_path),
            "data_needed":m.SETTING.get("data_needed"),
            "factor_needed":m.SETTING.get("factor_needed",[]),
            "universe":m.SETTING.get("universe"),
            "pasteurization":m.SETTING.get("pasteurization"),
            "decay":m.SETTING.get("decay",0),
            "neutralize":m.SETTING.get("neutralize",None),
            "submitted_at":datetime.now().strftime("%Y%m%d%H%M%S"),
        }

        self._write_all(all_data)
        return all_data[factor_name]

    def list_factors(self)->dict:
        return self._read_all()

    def get_factor(self,factor_name:str)->dict|None:
        all_data = self._read_all()
        return all_data.get(factor_name)
    
    def has_factor(self,factor_name:str)->bool:
        data = self._read_all()
        return factor_name in data
    
    def delete_factor(self,factor_name:str)->dict|None:
        all_data = self._read_all()
        for k in all_data.keys():
            if k==factor_name:
                item = all_data[k]
                del all_data[k]
                self._write_all(all_data)
                return item
        return None
