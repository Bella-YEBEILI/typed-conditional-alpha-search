import pandas as pd

from .core.safe_pickle_writer import SafePickleWriter
from .core.fundamental_updater import FundamentalUpdater
from .core.fundamental_neutralizer import FundamentalNeutralizer
from .service.update_service import UpdateService
from .service.access_service import AccessService

class DataManager:
    def __init__(self,
                 source_dir:str="/home/workspace/common/product_pkl/data",
                 fundamental_input_dir:str="/home/workspace/common/product_pkl/fundamentals",
                 base_dir:str="/home/workspace/common/quant_data"):  
        self.access = AccessService(base_dir=base_dir)
        fundamental_updater = FundamentalUpdater(source_dir,fundamental_input_dir,base_dir)
        fundamental_neutralizer = FundamentalNeutralizer(base_dir)
        self.updater = UpdateService(source_dir,base_dir,fundamental_updater,fundamental_neutralizer) 

    def list_datas(self):
        return self.access.list_datas()

    def get_data(self,
                 fld:str,
                 start:str|None="20150101",
                 end:str|None=None)->pd.DataFrame:
        return self.access.get_data(fld,start,end)
    
    def get_datas(self,
                  flds:list[str],
                  start:str|None="20150101",
                  end:str|None=None)->dict[str,pd.DataFrame]:
        return self.access.get_datas(flds,start,end)
    
    def update(self):
        self.updater.update()

    @staticmethod
    def safe_to_pickle(obj,path:str):
        SafePickleWriter.safe_to_pickle(obj,path)