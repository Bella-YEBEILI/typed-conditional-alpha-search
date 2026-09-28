import os
import shutil
import uuid

import pandas as pd
import numpy as np

from ..core.strategy_loader import load_strategy_module
from ..core.strategy_registry import StrategyRegistry
from ..core.strategy_result_engine import StrategyResultEngine
from ..core.strategy_result_store import StrategyResultStore
from ..core.strategy_signal_engine import StrategySignalEngine
from ..core.strategy_signal_store import StrategySignalStore
from ..core.strategy_weight_engine import StrategyWeightEngine
from ..core.strategy_weight_store import StrategyWeightStore


DEFAULT_TRADE_POINT = "open_vwaps_30m"


class ManageService:
    def __init__(self,
                 registry:StrategyRegistry,
                 signal_engine:StrategySignalEngine,
                 signal_store:StrategySignalStore,
                 weight_engine:StrategyWeightEngine,
                 weight_store:StrategyWeightStore,
                 result_engine:StrategyResultEngine,
                 result_store:StrategyResultStore,
                 base_dir:str):
        self.registry = registry
        self.signal_engine = signal_engine
        self.signal_store = signal_store
        self.weight_engine = weight_engine
        self.weight_store = weight_store
        self.result_engine = result_engine
        self.result_store = result_store
        self.base_dir = base_dir
        os.makedirs(self.base_dir,exist_ok=True)

    def _backup_file(self,path:str)->str|None:
        if not os.path.exists(path):
            return None
        backup_path = f"{path}.bak.{uuid.uuid4().hex}"
        shutil.copy2(path,backup_path)
        return backup_path

    def _restore_file(self,path:str,backup_path:str|None)->None:
        if backup_path is None:
            if os.path.exists(path):
                os.remove(path)
            return
        os.replace(backup_path,path)

    def _cleanup_backup(self,backup_path:str|None)->None:
        if backup_path is not None and os.path.exists(backup_path):
            os.remove(backup_path)

    def submit(self,
               file_path:str)->dict:
        module = load_strategy_module(file_path)
        if module.TYPE != "regular":
            # TODO: super strategy
            raise ValueError("unsupported type")
        
        universe = module.SETTING["universe"]
        method = module.SETTING["method"]
        weight_config_name = module.SETTING["weight_config_name"]

        strategy_name = module.META["strategy_name"]
        if self.registry.has_strategy(strategy_name):
            raise ValueError(f"strategy already exists: {strategy_name}")

        dst_dir = os.path.join(self.base_dir,"strategy_repo")
        os.makedirs(dst_dir,exist_ok=True)
        dst_path = os.path.join(dst_dir,strategy_name+".py")
        if os.path.exists(dst_path):
            raise FileExistsError(f"strategy file already exists: {dst_path}")

        signal_done = False
        weight_done = False
        result_done = False
        registered = False
        copied = False

        try:
            shutil.copy2(file_path,dst_path)
            copied = True

            print("generate signal...")
            signal = self.signal_engine.run_by_path(dst_path)
            signal_path = self.signal_store.save(strategy_name,signal)
            signal_done = True

            print("generate weight...")
            weight = self.weight_engine.generate_weight(signal,universe,method,weight_config_name)
            weight_path = self.weight_store.save(strategy_name,weight)
            weight_done = True

            print("generate result...")
            result = self.result_engine.calc_result(weight,
                                                    trade_point=DEFAULT_TRADE_POINT)
            result_path = self.result_store.save(strategy_name,DEFAULT_TRADE_POINT,result)
            result_done = True

            registry_item = self.registry.register_strategy(dst_path)
            registered = True

            return {
                "strategy_name":strategy_name,
                "status":"submitted",
                "strategy_file_path":dst_path,
                "registry_item":registry_item,
                "signal_path":signal_path,
                "weight_path":weight_path,
                "result_path":result_path,
                "trade_point":DEFAULT_TRADE_POINT,
            }
        
        except Exception:
            if registered:
                self.registry.delete_strategy(strategy_name)
            if result_done:
                self.result_store.delete(strategy_name,DEFAULT_TRADE_POINT)
            if weight_done:
                self.weight_store.delete(strategy_name)
            if signal_done:
                self.signal_store.delete(strategy_name)
            if copied and os.path.exists(dst_path):
                os.remove(dst_path)
            raise

    def update(self,
               strategy_name:str,
               end:str|None=None)->dict:
        item = self.registry.get_strategy(strategy_name)
        if item is None:
            raise ValueError(f"strategy not found: {strategy_name}")
        if not self.signal_store.has(strategy_name):
            raise FileNotFoundError(f"strategy signal not found: {strategy_name}")

        file_path = item["file_path"]
        module = load_strategy_module(file_path)
        if module.TYPE != "regular":
            # TODO: super strategy
            raise ValueError("unsupported type")
        
        universe = module.SETTING["universe"]
        method = module.SETTING["method"]
        weight_config_name = module.SETTING["weight_config_name"]

        old_signal = self.signal_store.load(strategy_name)
        if old_signal.empty:
            raise ValueError(f"stored signal is empty: {strategy_name}")

        last_signal_dt = pd.Timestamp(old_signal.index.max())
        if pd.Timestamp(end)<last_signal_dt:
            raise ValueError(f"end must be >= last signal date: {last_signal_dt.strftime('%Y%m%d')}")
        
        print(f"update signal: {last_signal_dt.strftime('%Y%m%d')}->{end}")
        update_start = last_signal_dt.strftime("%Y%m%d")
        new_signal = self.signal_engine.run_by_path(file_path,start=update_start,end=end)
        total_signal = pd.concat([old_signal.astype(np.float64,copy=False),
                                  new_signal.astype(np.float64,copy=False)],axis=0)
        total_signal = total_signal[~total_signal.index.duplicated(keep="last")].sort_index()

        weight = self.weight_engine.generate_weight(total_signal,
                                                    universe,
                                                    method,
                                                    weight_config_name,
                                                    start=total_signal.index.min().strftime("%Y%m%d"),
                                                    end=end)
        
        result = self.result_engine.calc_result(weight,
                                                trade_point=DEFAULT_TRADE_POINT,
                                                end=end)

        signal_path = self.signal_store.get_path(strategy_name)
        weight_path = self.weight_store.get_path(strategy_name)
        result_path = self.result_store.get_path(strategy_name,DEFAULT_TRADE_POINT)
        signal_backup = None
        weight_backup = None
        result_backup = None

        try:
            signal_backup = self._backup_file(signal_path)
            saved_signal_path = self.signal_store.save(strategy_name,total_signal)

            weight_backup = self._backup_file(weight_path)
            saved_weight_path = self.weight_store.save(strategy_name,weight)

            result_backup = self._backup_file(result_path)
            saved_result_path = self.result_store.save(strategy_name,DEFAULT_TRADE_POINT,result)
        except Exception:
            self._restore_file(signal_path,signal_backup)
            self._restore_file(weight_path,weight_backup)
            self._restore_file(result_path,result_backup)
            raise
        else:
            self._cleanup_backup(signal_backup)
            self._cleanup_backup(weight_backup)
            self._cleanup_backup(result_backup)

        return {
            "strategy_name":strategy_name,
            "status":"updated",
            "updated_from":update_start,
            "updated_to":end,
            "signal_path":saved_signal_path,
            "weight_path":saved_weight_path,
            "result_path":saved_result_path,
            "trade_point":DEFAULT_TRADE_POINT,
        }
    
    def update_batch(self,
                     strategy_names:list[str]|None=None,
                     end:str|None=None)->dict:
        update_results = {}

        if strategy_names is None:
            strategy_names = self.registry.list_strategy_names()
        
        for strategy_name in strategy_names:
            print("update strategy: ",strategy_name)
            update_results[strategy_name] = self.update(strategy_name,end)
        
        return update_results

    def delete(self,strategy_name:str)->dict:
        item = self.registry.delete_strategy(strategy_name)
        if item is None:
            raise ValueError(f"strategy not found: {strategy_name}")

        file_path = item["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)

        signal_deleted = self.signal_store.delete(strategy_name)
        weight_deleted = self.weight_store.delete(strategy_name)

        result_dir = os.path.join(self.base_dir,"strategy_results",strategy_name)
        result_deleted = os.path.isdir(result_dir)
        if result_deleted:
            shutil.rmtree(result_dir)

        return {
            "strategy_name":strategy_name,
            "status":"deleted",
            "registry_item":item,
            "strategy_file_deleted":not os.path.exists(file_path),
            "signal_deleted":signal_deleted,
            "weight_deleted":weight_deleted,
            "result_deleted":result_deleted,
        }
