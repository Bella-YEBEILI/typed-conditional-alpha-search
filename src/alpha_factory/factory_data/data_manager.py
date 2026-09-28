import os
import pickle
import numpy as np
from pathlib import Path


class DataManager:
    def __init__(self,
                 mmap:bool=False):
        self.mmap_mode = "r" if mmap else None
        self.base_dir = (Path(__file__).resolve().parent/"data").resolve()
        self.data_index:dict[str,str] = {}
        for root,_,files in os.walk(self.base_dir):
            for f in files:
                if f.endswith(".npy"):
                    name = f[:-4]
                    self.data_index[name] = os.path.join(root,f)
        

    def list_datas(self)->list[str]:
        return sorted(list(self.data_index.keys()))

    def get_data(self,fld:str)->np.ndarray:
        path = self.data_index.get(fld)
        if path is None:
            raise KeyError(f"field '{fld}' not found in {self.base_dir}")

        return np.load(path,mmap_mode=self.mmap_mode,allow_pickle=False)

    def get_datas(self,flds:list[str])->dict[str,np.ndarray]:
        datas = {}
        for fld in flds:
            datas[fld] = self.get_data(fld)
        return datas

    def get_axis(self)->tuple[np.ndarray,np.ndarray]:
        with open(self.base_dir/"axis"/"dates.pkl","rb") as f:
            dates = pickle.load(f)
        with open(self.base_dir/"axis"/"stocks.pkl","rb") as f:
            stocks = pickle.load(f)
        return dates,stocks
