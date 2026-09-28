import pandas as pd
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from quant.data_system.data_hub.main import DataManager
from quant.factor_system.factor_hub.main import FactorManager
from quant.strategy_system.strategy_hub.main import StrategyManager

class QuantPipeline:
    def __init__(self):
        self.data_manager = DataManager()
        self.factor_manager = FactorManager()
        self.strategy_manager = StrategyManager()

    def update(self,end:str):
        print(f"update to end: {end}")
        t1 = time.perf_counter()
        self.data_manager.update()
        t2 = time.perf_counter()
        print("update data time ",t2-t1,"sec")

        t1 = time.perf_counter()
        self.factor_manager.manage.update_batch(end=end)
        t2 = time.perf_counter()
        print("update factor time ",t2-t1,"sec")

        t1 = time.perf_counter()
        self.strategy_manager.manage.update_batch(end=end)
        t2 = time.perf_counter()
        print("update strategy time ",t2-t1,"sec")

if __name__=="__main__":
    all_dates = pd.read_pickle("/home/workspace/common/product_pkl/data/all_dates.pkl")
    end = all_dates[-1].strftime("%Y%m%d")
    QuantPipeline().update(end)