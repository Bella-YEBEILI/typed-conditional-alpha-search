import json
import os


class FactorSelector:
    def __init__(self,selector_path:str)->None:
        self.selector_path = selector_path
        self._ensure_file()

    def _ensure_file(self)->None:
        folder = os.path.dirname(self.selector_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder,exist_ok=True)
        if not os.path.exists(self.selector_path) or os.path.getsize(self.selector_path)==0:
            self._write([])

    def _read(self)->list[str]:
        with open(self.selector_path,"r",encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data,list):
            raise ValueError("selected_factors file must contain a list")
        return data

    def _write(self,data:list[str])->None:
        with open(self.selector_path,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2)

    def list_selected(self)->list[str]:
        return self._read()

    def is_selected(self,factor_name:str)->bool:
        return factor_name in self._read()

    def add(self,factor_name:str)->None:
        data = self._read()
        if factor_name in data:
            return
        data.append(factor_name)
        self._write(data)

    def remove(self,factor_name:str)->None:
        data = self._read()
        if factor_name not in data:
            return
        data.remove(factor_name)
        self._write(data)

    def set(self,factor_names:list[str])->None:
        self._write(list(dict.fromkeys(factor_names)))
