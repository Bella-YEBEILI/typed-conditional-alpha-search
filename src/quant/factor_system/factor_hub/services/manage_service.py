import os
import hashlib
import shutil

import numpy as np
import pandas as pd
import time

from ..core.factor_loader import load_factor_module
from ..core.factor_result_profiles import DEFAULT_PROFILE_ID
from ..core.factor_performance_engine import DEFAULT_PARAMS
from ..core.factor_criteria import FactorCriteria

from ..components.lib_interactor import LibInteractor
from ..components.factor_comparator import FactorComparator
from ..components.factor_simulator import FactorSimulator

from ..plugins.factor_portrait import FactorPortraitBuilder
from ..plugins.factor_portrait import FactorPortraitStore

def _file_code_hash(file_path:str)->str:
    with open(file_path,"rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class ManageService:
    def __init__(self,
                 interactor:LibInteractor,
                 comparator:FactorComparator,
                 simulator:FactorSimulator,
                 factors_out_dir:str,
                 portrait_builder:FactorPortraitBuilder,
                 portrait_store:FactorPortraitStore,
                 criteria:FactorCriteria):
        self.interactor = interactor
        self.comparator = comparator
        self.simulator = simulator
        self.factors_out_dir = factors_out_dir
        self.portrait_builder = portrait_builder
        self.portrait_store = portrait_store
        self.criteria = criteria
        os.makedirs(self.factors_out_dir,exist_ok=True)
        self._check_cache:dict[str,dict] = {}

    # library factor
    def has_factor(self,factor_name:str)->bool:
        return self.interactor.has_factor(factor_name)

    def has_value(self,factor_name:str)->bool:
        return self.interactor.has_value(factor_name)

    def has_result(self,factor_name:str,profile_id:str=DEFAULT_PROFILE_ID)->bool:
        return self.interactor.has_result(factor_name,profile_id)
    
    def get_value(self,factor_name:str)->pd.DataFrame:
        return self.interactor.get_value(factor_name)
    
    def get_result(self,factor_name:str,profile_id:str=DEFAULT_PROFILE_ID)->dict:
        return self.interactor.get_result(factor_name,profile_id)

    # materialize single
    def materialize_value(self,factor_name:str,end:str|None=None):
        return self.interactor.materialize_value(factor_name,end)

    def materialize_result(self,factor_name:str,profile_id:str=DEFAULT_PROFILE_ID):
        return self.interactor.materialize_result(factor_name,profile_id)

    def materialize_portrait(self,factor_name:str):
        factor_value = self.get_value(factor_name)
        factor_result = self.get_result(factor_name)
        factor_portrait = self.portrait_builder.build(factor_value,factor_result)
        portrait_path = self.portrait_store.save(factor_name,factor_portrait)
        return portrait_path

    # materialize batch
    def materialize_value_batch(self,factor_names:list[str]|None=None)->dict[str,str]:
        if factor_names is None:
            factor_names = self.interactor.list_factor_names()

        path_dict = {}
        for factor_name in factor_names:
            try:
                path_dict[factor_name] = self.materialize_value(factor_name)
            except Exception as e:
                print(f"materialize value failed: factor_name={factor_name}, error: {e}")
        return path_dict

    def materialize_result_batch(self,
                                 factor_names:list[str]|None=None,
                                 profile_id:str=DEFAULT_PROFILE_ID)->dict[str,str]:
        if factor_names is None:
            factor_names = self.interactor.list_factor_names()

        path_dict = {}
        for factor_name in factor_names:
            try:
                path_dict[factor_name] = self.materialize_result(factor_name,profile_id)
            except Exception as e:
                print(f"materialize result failed: factor_name={factor_name}, profile_id={profile_id}, error: {e}")
        return path_dict
    
    def materialize_portrait_batch(self,factor_names:list[str]|None=None)->dict[str,str]:
        if factor_names is None:
            factor_names = self.interactor.list_factor_names()
        
        progress = 1
        total = len(factor_names)

        path_dict = {}
        for factor_name in factor_names:
            print(f"materialize_portrait: {factor_name}, progress:{progress}, total:{total}")
            try:
                path_dict[factor_name] = self.materialize_portrait(factor_name)
            except Exception as e:
                print(f"materialize portrait failed: factor_name={factor_name}, error: {e}")
            progress += 1
        return path_dict

    # submit
    def _check_factor_dependencies(self,module):
        if module.TYPE!="super":
            return

        factor_needed = module.SETTING["factor_needed"]
        for factor_name in factor_needed:
            if not self.interactor.has_factor(factor_name):
                raise ValueError(f"dependent factor not found in registry: {factor_name}")
            if not self.interactor.has_value(factor_name):
                raise FileNotFoundError(f"dependent factor value not found: {factor_name}")

    def _compute_check(self,file_path:str)->dict:
        module = load_factor_module(file_path)
        self._check_factor_dependencies(module)

        factor_name = module.META["factor_name"]
        universe = module.SETTING.get("universe")
        domain = str(module.META.get("domain","pv"))
        tag = module.META.get("tag","")

        sim_result = self.simulator.simulate_py_file(file_path,DEFAULT_PROFILE_ID,DEFAULT_PARAMS)
        factor_value = sim_result["factor_value"]
        factor_result = sim_result["factor_result"]

        # portrait
        portrait = {}
        portrait = self.portrait_builder.build(factor_value,factor_result,DEFAULT_PROFILE_ID,DEFAULT_PARAMS)

        raw = portrait.get("raw",{})
        full_perf = raw.get("all",sim_result["factor_performance"]) if isinstance(raw,dict) else sim_result["factor_performance"]
        hs300s_perf = portrait.get("hs300s",{}).get("all",{}) if isinstance(portrait.get("hs300s"),dict) else {}
        zz1000s_perf = portrait.get("zz1000s",{}).get("all",{}) if isinstance(portrait.get("zz1000s"),dict) else {}
        complete_perf = portrait.get("complete",{}).get("all",{}) if isinstance(portrait.get("complete"),dict) else {}

        # prod corr
        rankics_series = factor_result.get("rankics",pd.Series(dtype=float))
        rankics_data = self.interactor.get_rankics_data(profile_id=DEFAULT_PROFILE_ID,
                                                        tier="selected",
                                                        universe=universe,
                                                        domain=domain)
        
        prod_corr_result = self.comparator.check_prod_corr(lib_data=rankics_data,new_data=rankics_series)
        if factor_name in prod_corr_result.index:
            prod_corr_result = prod_corr_result.drop(index=factor_name)
        prod_corr_result = prod_corr_result.dropna()

        if len(prod_corr_result)>0:
            max_prod_corr = prod_corr_result.iloc[0]
            max_prod_corr_factor = prod_corr_result.index[0]
            avg_prod_corr = prod_corr_result.mean()
        else:
            max_prod_corr = np.nan
            max_prod_corr_factor = ""
            avg_prod_corr = np.nan

        full_perf = dict(full_perf)
        full_perf["max_prod_corr"] = max_prod_corr
        full_perf["max_prod_corr_factor"] = max_prod_corr_factor
        full_perf["avg_prod_corr"] = avg_prod_corr

        nan_corr = {k:np.nan for k in ("max_prod_corr","max_prod_corr_factor","avg_prod_corr")}
        hs300s_row = {**dict(hs300s_perf),**nan_corr}
        zz1000s_row = {**dict(zz1000s_perf),**nan_corr}
        complete_row = {**dict(complete_perf),**nan_corr}

        rows = {
            "raw":full_perf,
            "hs300s":hs300s_row,
            "zz1000s":zz1000s_row,
            "complete":complete_row,
        }

        # max corr factor perf
        if max_prod_corr_factor:
            try:
                corr_perf = self.interactor.get_performance(
                    max_prod_corr_factor,
                    profile_id=DEFAULT_PROFILE_ID,
                    params=DEFAULT_PARAMS,
                    persist_result=False,
                )
                corr_row = dict(corr_perf)
                corr_row["max_prod_corr"] = np.nan
                corr_row["max_prod_corr_factor"] = np.nan
                corr_row["avg_prod_corr"] = np.nan
                rows[max_prod_corr_factor] = corr_row
            except Exception:
                pass

        df = pd.DataFrame(rows).T
        df.index.name = "factor"

        if tag=="smart_styles":
            return {
                "factor_name":factor_name,
                "universe":universe,
                "domain":domain,
                "tag":tag,
                "summary":df,
                "factor_value":factor_value,
                "factor_result":factor_result,
                "portrait":portrait,
                "section_perfs":{"raw":dict(full_perf),"zz1000s":dict(zz1000s_perf)},
            }
        else:
            return {
                "factor_name":factor_name,
                "universe":universe,
                "domain":domain,
                "tag":tag,
                "summary":df,
                "factor_value":factor_value,
                "factor_result":factor_result,
                "portrait":portrait,
                "section_perfs":{"raw":dict(full_perf),"zz1000s":dict(zz1000s_perf),"complete":dict(complete_perf)},
            }


    def check_submission(self,file_path:str)->tuple[pd.DataFrame,dict]:
        code_hash = _file_code_hash(file_path)
        cached = self._check_cache.get(file_path)
        if cached is not None and cached["code_hash"]==code_hash:
            print(f"[check_submission] cache hit: {file_path}")
            result = cached["result"]
        else:
            print(f"[check_submission] computing: {file_path}")
            result = self._compute_check(file_path)
            self._check_cache[file_path] = {"code_hash":code_hash,"result":result}

        criteria_result = self.criteria.check(result["section_perfs"],result["universe"],result["domain"])
        return result["summary"],criteria_result

    def submit(self,file_path:str)->dict:
        module = load_factor_module(file_path)
        self._check_factor_dependencies(module)

        factor_name = module.META["factor_name"]
        if self.has_factor(factor_name):
            raise ValueError(f"factor already exists: {factor_name}")

        code_hash = _file_code_hash(file_path)
        cached = self._check_cache.get(file_path)
        if cached is not None and cached["code_hash"]==code_hash:
            print(f"[submit] reusing check_submission cache for {factor_name}")
            cached_data = cached["result"]
        else:
            print(f"[submit] computing: {factor_name}")
            cached_data = self._compute_check(file_path)
            self._check_cache[file_path] = {"code_hash":code_hash,"result":cached_data}

        factor_value = cached_data["factor_value"]
        factor_result = cached_data["factor_result"]
        portrait = cached_data["portrait"]
        section_perfs = cached_data["section_perfs"]
        universe = cached_data["universe"]
        domain = cached_data["domain"]

        criteria_result = self.criteria.check(section_perfs,universe,domain)
        if not criteria_result["passed"]:
            return {
                "factor_name":factor_name,
                "status":"rejected",
                "criteria_details":criteria_result["details"],
            }

        dst_path = os.path.join(self.factors_out_dir,f"{factor_name}.py")
        copied = False
        registered = False
        value_done = False
        result_done = False
        portrait_done = False

        try:
            shutil.copy2(file_path,dst_path)
            copied = True

            reg_item = self.interactor.register_factor(dst_path)
            registered = True

            if factor_value is not None:
                value_path = self.interactor.value_store.save(factor_name,factor_value)
            else:
                value_path = self.interactor.materialize_value(factor_name)
            value_done = True

            if factor_result is not None:
                result_path = self.interactor.result_store.save(factor_name,DEFAULT_PROFILE_ID,factor_result)
            else:
                result_path = self.interactor.materialize_result(factor_name,DEFAULT_PROFILE_ID)
            result_done = True

            portrait_path = None
            if self.portrait_builder is not None and self.portrait_store is not None:
                if portrait is None:
                    fv = self.interactor.get_value(factor_name)
                    fr = self.interactor.get_result(factor_name,DEFAULT_PROFILE_ID)
                    portrait = self.portrait_builder.build(fv,fr,DEFAULT_PROFILE_ID,DEFAULT_PARAMS)
                portrait_path = self.portrait_store.save(factor_name,portrait)
                portrait_done = True

            # clear cache after successful submit
            self._check_cache.pop(file_path,None)

            return {
                "factor_name":factor_name,
                "status":"submitted",
                "factor_file_path":dst_path,
                "registry_item":reg_item,
                "value_path":value_path,
                "result_path":result_path,
                "portrait_path":portrait_path,
            }
        except Exception as e:
            print(f"submit failed: factor_name={factor_name}, error={e}")
            if portrait_done and self.portrait_store is not None:
                self.portrait_store.delete(factor_name)
            if result_done:
                self.interactor.delete_result(factor_name,DEFAULT_PROFILE_ID)
            if value_done:
                self.interactor.delete_value(factor_name)
            if registered:
                self.interactor.delete_factor(factor_name)
            if copied and os.path.exists(dst_path):
                os.remove(dst_path)
            raise

    def submit_batch(self,folder_path:str)->dict[str,dict]:
        if not os.path.isdir(folder_path):
            raise ValueError(f"folder not found: {folder_path}")

        file_names = sorted(fn for fn in os.listdir(folder_path) if fn.endswith(".py"))
        total = len(file_names)
        results = {}

        for i,file_name in enumerate(file_names,1):
            file_path = os.path.join(folder_path,file_name)
            print(f"submit progress: {i}/{total} {file_name}")
            try:
                results[file_name] = self.submit(file_path)
                print(f"{file_name}: {results[file_name]['status']}")
            except Exception as e:
                results[file_name] = {"error":str(e)}
                print(e)
        return results

    # delete
    def delete(self,factor_name:str)->dict:
        if not self.has_factor(factor_name):
            raise ValueError(f"factor not found in registry: {factor_name}")

        factor_info = self.interactor.get_factor(factor_name)
        factor_file = factor_info.get("file_path")

        if self.portrait_store is not None and self.portrait_store.has(factor_name):
            self.portrait_store.delete(factor_name)

        if self.has_result(factor_name):
            self.interactor.delete_result(factor_name,DEFAULT_PROFILE_ID)

        if self.has_value(factor_name):
            self.interactor.delete_value(factor_name)

        self.interactor.delete_factor(factor_name)
        self.interactor.select_remove(factor_name)

        if factor_file and os.path.exists(factor_file):
            os.remove(factor_file)

        return {"factor_name":factor_name,"status":"deleted"}
    
    def _update_result(self,factor_name:str):
        result_paths = []
        result_paths.append(self.materialize_result(factor_name))
        for profile_id in [k for k in self.interactor.list_profile_ids(factor_name) if k != DEFAULT_PROFILE_ID]:
            result_paths.append(self.materialize_result(factor_name,profile_id))
        return result_paths
    
    # update
    def update(self,
               factor_name:str,
               end:str|None=None,
               update_result:bool=True,
               update_portrait:bool=True)->dict:
        
        t1 = time.perf_counter()
        value_path,final_dt = self.materialize_value(factor_name,end)
        t2 = time.perf_counter()
        print("update value time: ",t2-t1)
        print(f"update value to: {final_dt}")
        result_paths = None
        portrait_path = None

        t1 = time.perf_counter()
        if update_result or update_portrait:
            result_paths = self._update_result(factor_name)
        t2 = time.perf_counter()
        print("update result time: ",t2-t1)
        
        if update_portrait:
            t1 = time.perf_counter()
            portrait_path = self.materialize_portrait(factor_name)
            t2 = time.perf_counter()
            print("update portrait time: ",t2-t1)

        return {
            "factor_name":factor_name,
            "status":"updated",
            "value_path":value_path,
            "value_final_dt":final_dt,
            "result_paths":result_paths,
            "portrait_path":portrait_path
        }

    def update_batch(self,
                     factor_names:list[str]|None=None,
                     end:str|None=None,
                     update_result:bool=True,
                     update_portrait:bool=True,
                     **kwargs)->dict:
        # TODO: super alpha的兼容问题
        update_results = {}

        factor_names = self.interactor.filter_names(factor_names,**kwargs)
        
        for factor_name in factor_names:
            print("update factor: ",factor_name)
            update_results[factor_name] = self.update(factor_name,end,update_result,update_portrait)
        
        return update_results
