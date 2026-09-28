import json
import os
from datetime import datetime
from .strategy_loader import load_strategy_module

class StrategyRegistry:
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
        for _,item in data.items():
            if not isinstance(item,dict):
                raise ValueError("registry item must be dict")
        return data

    def _write_all(self,data:dict)->None:
        tmp_path = f"{self.registry_path}.tmp"
        with open(tmp_path,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2)
        os.replace(tmp_path,self.registry_path)

    def register_strategy(self,file_path:str)->dict:
        m = load_strategy_module(file_path)
        meta = m.META

        if hasattr(m,"PRODUCT"):
            product = m.PRODUCT
        else:
            product = "unknown"

        strategy_name = meta.get("strategy_name")
        if not strategy_name:
            raise ValueError("META.strategy_name is required")

        all_data = self._read_all()
        if strategy_name in all_data:
            raise ValueError(f"strategy already exists: {strategy_name}")
        
        all_data[strategy_name] = {
            "strategy_name":strategy_name,
            "product":product,
            "author":meta.get("author","unknown"),
            "file_path":os.path.abspath(file_path),
            "strategy_needed":m.SETTING.get("strategy_needed",[]),
            "universe":m.SETTING.get("universe"),
            "method":m.SETTING.get("method"),
            "weight_config_name":m.SETTING.get("weight_config_name"),
            "submitted_at":datetime.now().strftime("%Y%m%d%H%M%S"),
        }

        self._write_all(all_data)
        return all_data[strategy_name]

    def list_strategys(self)->dict:
        return self._read_all()
    
    def list_strategy_names(self)->list[str]:
        return list(self._read_all().keys())

    def get_strategy(self,strategy_name:str)->dict|None:
        all_data = self._read_all()
        return all_data.get(strategy_name)
    
    def has_strategy(self,strategy_name:str)->bool:
        data = self._read_all()
        return strategy_name in data
    
    def delete_strategy(self,strategy_name:str)->dict|None:
        all_data = self._read_all()
        for k in all_data.keys():
            if k==strategy_name:
                item = all_data[k]
                del all_data[k]
                self._write_all(all_data)
                return item
        return None