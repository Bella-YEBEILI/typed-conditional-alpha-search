import os
import pandas as pd

class AccessService:
    def __init__(self,
                 base_dir:str):
        self.base_dir = base_dir
        self.data_index:dict[str,str] = {}
        for root,_,files in os.walk(self.base_dir):
            for f in files:
                if f.endswith(".pkl"):
                    name = f[:-4]
                    self.data_index[name] = os.path.join(root,f)

    def list_datas(self)->list[str]:
        return list(self.data_index.keys())

    def get_data(self,
                 fld:str,
                 start:str|None="20150101",
                 end:str|None=None)->pd.DataFrame:
        path = self.data_index.get(fld)
        if path is None:
            raise KeyError(f"field '{fld}' not found in {self.base_dir}")
        
        data = pd.read_pickle(path)

        if not isinstance(data.index,pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index.astype(str))
        
        return data.loc[start:end]

    def get_datas(self,
                  flds:list[str],
                  start:str|None="20150101",
                  end:str|None=None)->dict[str,pd.DataFrame]:
        datas = {}
        for fld in flds:
            datas[fld] = self.get_data(fld,start,end)
        return datas