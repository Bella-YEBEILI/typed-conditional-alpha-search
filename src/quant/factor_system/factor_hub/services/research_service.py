import os
import sys
import threading
from functools import lru_cache

import pandas as pd

from ..core.factor_result_profiles import DEFAULT_PROFILE_ID
from ..core.factor_performance_engine import DEFAULT_PARAMS

from ..components.lib_interactor import LibInteractor
from ..components.factor_comparator import FactorComparator
from ..components.factor_simulator import FactorSimulator
from ..tools.factor_plotter import FactorPlotter
from ..tools.factor_template_builder import FactorTemplateBuilder

from quant.quant_lib.quantEnum import UniverseType
from quant.quant_lib.quantEnum import DomainType


_SILENT_LOCK = threading.Lock()


class ResearchService:
    def __init__(self,
                 interactor:LibInteractor,
                 comparator:FactorComparator,
                 simulator:FactorSimulator,
                 plotter:FactorPlotter,
                 template_builder:FactorTemplateBuilder):
        self.interactor = interactor
        self.comparator = comparator
        self.simulator = simulator
        self.plotter = plotter
        self.template_builder = template_builder

    def create_template(self,factor_type:str="regular",out_dir:str|None=None,**kwargs)->str:
        if self.template_builder is None:
            raise ValueError("template_builder is required for create_template")
        return self.template_builder.create_template(factor_type,out_dir,**kwargs)

    def create_spec(self,
                    factor_type:str="regular",
                    **kwargs)->dict:
        spec = {
            "type":factor_type,
            "factor_name":"unknown",
            "author":"unknown",
            "level":"days",
            "tag":"",
            "category":"unknown",
            "universe":"standards",
            "pasteurization":False,
            "decay":0,
            "neutralize":None,
            "formula":"",
        }
        spec.update(kwargs)
        return spec

    # evaluate
    def _apply_transforms(self,bundle:dict,transform_spec:dict[str,list|str]|None,profile_id:str,params:dict):
        fv = bundle["factor_value"]
        result_dict = {"raw":bundle["factor_result"]}
        perf_dict = {"raw":bundle["factor_performance"]}
        if transform_spec:
            for field in ("subuniverse","neutralize"):
                val = transform_spec.get(field,[])
                if isinstance(val,str):
                    val = [val]
                for key in val:
                    tb = self.simulator.simulate_transformed(fv,field,key,profile_id,params)
                    result_dict[key] = tb["factor_result"]
                    perf_dict[key] = tb["factor_performance"]
        bundle["factor_result"] = result_dict
        bundle["factor_performance"] = perf_dict

    def _process_single(self,
                        bundle:dict,
                        profile_id:str,
                        params:dict,
                        prod_corr:bool,
                        prod_corr_universe:str|None,
                        prod_corr_domain:str|None,
                        plot:bool):
        if prod_corr:
            rankics_series = bundle["factor_result"].get("rankics",pd.Series(dtype=float))

            universe = prod_corr_universe
            if universe is None:
                universe = bundle.get("universe")
                if universe is None:
                    universe = UniverseType.standard
            
            domain = prod_corr_domain
            if domain is None:
                domain = bundle.get("domain")
                if domain is None:
                    domain = DomainType.pv
                
            rankics_data = self.interactor.get_rankics_data(profile_id=profile_id,
                                                            tier="selected",
                                                            universe=str(universe),
                                                            domain=str(domain))
                
            prod_corr_result = self.comparator.check_prod_corr(lib_data=rankics_data,new_data=rankics_series)
            bundle["prod_corr_result"] = prod_corr_result
        
        if plot:
            self.plotter.plot_result(
                id=bundle["id"],
                factor_result=bundle["factor_result"],
                params=params,
            )
    
    def _evaluate_value(self,
                        factor_value:pd.DataFrame,
                        profile_id:str=DEFAULT_PROFILE_ID,
                        params:dict=DEFAULT_PARAMS,
                        prod_corr:bool=False,
                        prod_corr_universe:str|None=None,
                        prod_corr_domain:str|None=None,
                        plot:bool=True)->dict:
        bundle = self.simulator.simulate_value(factor_value=factor_value,
                                               factor_name=None,
                                               profile_id=profile_id,
                                               params=params)
        self._process_single(bundle,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,plot)
        return bundle
    
    def _evaluate_pkl_file(self,
                           pkl_path:str,
                           profile_id:str=DEFAULT_PROFILE_ID,
                           params:dict=DEFAULT_PARAMS,
                           prod_corr:bool=False,
                           prod_corr_universe:str|None=None,
                           prod_corr_domain:str|None=None,
                           plot:bool=True)->dict:
        bundle = self.simulator.simulate_pkl_file(pkl_path,profile_id,params)
        self._process_single(bundle,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,plot)
        return bundle

    def _evaluate_py_file(self,
                          py_path:str,
                          profile_id:str=DEFAULT_PROFILE_ID,
                          params:dict=DEFAULT_PARAMS,
                          prod_corr:bool=False,
                          prod_corr_universe:str|None=None,
                          prod_corr_domain:str|None=None,
                          plot:bool=True)->dict:
        bundle = self.simulator.simulate_py_file(py_path,profile_id,params)
        self._process_single(bundle,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,plot)
        return bundle
    
    def _evaluate_spec(self,
                       spec:dict,
                       profile_id:str=DEFAULT_PROFILE_ID,
                       params:dict=DEFAULT_PARAMS,
                       prod_corr:bool=False,
                       prod_corr_universe:str|None=None,
                       prod_corr_domain:str|None=None,
                       plot:bool=True)->dict:
        bundle = self.simulator.simulate_spec(spec,profile_id,params)
        self._process_single(bundle,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,plot)
        return bundle
    
    def _evaluate_formula(self,
                          formula:str,
                          profile_id:str=DEFAULT_PROFILE_ID,
                          params:dict=DEFAULT_PARAMS,
                          prod_corr:bool=False,
                          prod_corr_universe:str|None=None,
                          prod_corr_domain:str|None=None,
                          plot:bool=True)->dict:
        bundle = self.simulator.simulate_formula(formula,profile_id,params)
        self._process_single(bundle,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,plot)
        return bundle    
    
    def evaluate(self,
                 target,
                 profile_id:str=DEFAULT_PROFILE_ID,
                 params:dict=DEFAULT_PARAMS,
                 prod_corr:bool=False,
                 prod_corr_universe:str|None=None,
                 prod_corr_domain:str|None=None,
                 plot:bool=True,
                 transform_spec:dict[str,list|str]|None=None,
                 silent:bool=False)->dict:
        if silent:
            _SILENT_LOCK.acquire()
            _stdout = sys.stdout
            _devnull = open(os.devnull,"w")
            sys.stdout = _devnull
        try:
            if isinstance(target,pd.DataFrame):
                bundle = self._evaluate_value(target,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,plot)
            elif isinstance(target,dict):
                bundle = self._evaluate_spec(target,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,plot)
            elif isinstance(target,str):
                if os.path.isfile(target) and target.endswith(".py"):
                    bundle = self._evaluate_py_file(target,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,plot)
                elif os.path.isfile(target) and target.endswith(".pkl"):
                    bundle = self._evaluate_pkl_file(target,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,plot)
                else:
                    bundle = self._evaluate_formula(target,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,plot)
            else:
                raise ValueError("unsupported target type")
            self._apply_transforms(bundle,transform_spec,profile_id,params)
            return bundle
        finally:
            if silent:
                sys.stdout = _stdout
                _devnull.close()
                _SILENT_LOCK.release()
        
    # evaluate batch

    def _batch_process(self,
                       bundle:dict,
                       profile_id:str,
                       params:dict,
                       prod_corr:bool,
                       prod_corr_universe:str|None,
                       prod_corr_domain:str|None,
                       rows:list,
                       self_rankic_data:dict,
                       transform_spec:dict[str,list|str]|None=None):
        self._process_single(bundle,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,plot=False)
        id = bundle["id"]
        row = dict(bundle["factor_performance"])
        row["id"] = id
        if transform_spec is not None:
            row["transform_key"] = "raw"
        if prod_corr:
            prod_corr_result = bundle.get("prod_corr_result",pd.Series(dtype=float))
            if len(prod_corr_result)>0:
                row["max_prod_corr"] = prod_corr_result.iloc[0]
                row["max_prod_corr_factor"] = prod_corr_result.index[0]
                row["avg_prod_corr"] = prod_corr_result.mean()
            else:
                row["max_prod_corr"] = pd.NA
                row["max_prod_corr_factor"] = pd.NA
                row["avg_prod_corr"] = pd.NA
        rows.append(row)
        if self_rankic_data is not None:
            self_rankic_data[id] = bundle["factor_result"].get("rankics",pd.Series(dtype=float))
        if transform_spec:
            fv = bundle["factor_value"]
            for field in ("subuniverse","neutralize"):
                val = transform_spec.get(field,[])
                if isinstance(val,str):
                    val = [val]
                for key in val:
                    tb = self.simulator.simulate_transformed(fv,field,key,profile_id,params)
                    trow = dict(tb["factor_performance"])
                    trow["id"] = id
                    trow["transform_key"] = key
                    rows.append(trow)

    def _batch_finalize(self,
                        rows:list,
                        self_rankic_data:dict|None,
                        has_transform:bool=False)->pd.DataFrame|tuple[pd.DataFrame,pd.DataFrame]:
        if len(rows)==0:
            return (pd.DataFrame(),pd.DataFrame()) if self_rankic_data is not None else pd.DataFrame()
        perf_df = pd.DataFrame(rows)
        if has_transform:
            perf_df = perf_df.set_index(["id","transform_key"])
        else:
            perf_df = perf_df.set_index("id")
        if self_rankic_data is None:
            return perf_df

        self_corr_df,self_corr_stats_df = self.comparator.check_self_corr(self_rankic_data)
        corr_cols = ["max_self_corr","max_self_corr_factor","avg_self_corr"]

        if has_transform:
            raw_corr = self_corr_stats_df[corr_cols]
            raw_corr.index = pd.MultiIndex.from_arrays([raw_corr.index,["raw"]*len(raw_corr)],names=["id","transform_key"])
            perf_df = perf_df.join(raw_corr,how="left")
        else:
            perf_df = perf_df.join(self_corr_stats_df[corr_cols],how="left")

        return perf_df,self_corr_df

    def _evaluate_batch_folder(self,
                               folder_path:str,
                               profile_id:str,
                               params:dict,
                               prod_corr:bool,
                               prod_corr_universe:str|None,
                               prod_corr_domain:str|None,
                               self_corr:bool,
                               transform_spec:dict[str,list|str]|None=None)->pd.DataFrame|tuple[pd.DataFrame,pd.DataFrame]:
        files = sorted(os.listdir(folder_path))
        py_files = [f for f in files if f.endswith(".py")]
        pkl_files = [f for f in files if f.endswith(".pkl")]

        if py_files and pkl_files:
            raise ValueError("folder contains both .py and .pkl files, must be one type only")
        if not py_files and not pkl_files:
            return (pd.DataFrame(),pd.DataFrame()) if self_corr else pd.DataFrame()
        rows = []
        self_rankic_data = {} if self_corr else None

        cnt = 0
        n = len(py_files)+len(pkl_files)

        for fn in (py_files or pkl_files):
            file_path = os.path.join(folder_path,fn)
            cnt += 1
            print(f"evaluate factor: {file_path}, progress: {cnt}, total:{n}")
            try:
                if fn.endswith(".py"):
                    bundle = self.simulator.simulate_py_file(file_path,profile_id,params)
                else:
                    bundle = self.simulator.simulate_pkl_file(file_path,profile_id,params)
                self._batch_process(bundle,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,rows,self_rankic_data,transform_spec)
            except Exception as e:
                print(f"evaluate failed: {file_path}, error: {e}")
                
        return self._batch_finalize(rows,self_rankic_data,has_transform=transform_spec is not None)

    def _evaluate_batch_values(self,
                               values:dict[str,pd.DataFrame],
                               profile_id:str,
                               params:dict,
                               prod_corr:bool,
                               prod_corr_universe:str|None,
                               prod_corr_domain:str|None,
                               self_corr:bool,
                               transform_spec:dict[str,list|str]|None=None)->pd.DataFrame|tuple[pd.DataFrame,pd.DataFrame]:
        rows = []
        self_rankic_data = {} if self_corr else None

        cnt = 0
        n = len(values)
        for name,df in values.items():
            cnt += 1
            print(f"evaluate factor value: {name}, progress:{cnt}, total:{n}")
            try:
                bundle = self.simulator.simulate_value(factor_value=df,factor_name=name,profile_id=profile_id,params=params)
                self._batch_process(bundle,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,rows,self_rankic_data,transform_spec)
            except Exception as e:
                print(f"evaluate failed: {name}, error: {e}")
        return self._batch_finalize(rows,self_rankic_data,has_transform=transform_spec is not None)

    def _evaluate_batch_specs(self,
                              specs:list[dict],
                              profile_id:str,
                              params:dict,
                              prod_corr:bool,
                              prod_corr_universe:str|None,
                              prod_corr_domain:str|None,
                              self_corr:bool,
                              transform_spec:dict[str,list|str]|None=None)->pd.DataFrame|tuple[pd.DataFrame,pd.DataFrame]:
        rows = []
        self_rankic_data = {} if self_corr else None

        cnt = 0
        n = len(specs)
        for i,spec in enumerate(specs):
            cnt += 1
            print(f"evaluate factor spec: {i}, progress:{cnt}, total:{n}")
            try:
                bundle = self.simulator.simulate_spec(spec,profile_id,params)
                self._batch_process(bundle,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,rows,self_rankic_data,transform_spec)
            except Exception as e:
                print(f"evaluate spec failed: index={i}, error: {e}")
        return self._batch_finalize(rows,self_rankic_data,has_transform=transform_spec is not None)

    def _evaluate_batch_formulas(self,
                                 formulas:list[str],
                                 profile_id:str,
                                 params:dict,
                                 prod_corr:bool,
                                 prod_corr_universe:str|None,
                                 prod_corr_domain:str|None,
                                 self_corr:bool,
                                 transform_spec:dict[str,list|str]|None=None)->pd.DataFrame|tuple[pd.DataFrame,pd.DataFrame]:
        rows = []
        self_rankic_data = {} if self_corr else None

        cnt = 0
        n = len(formulas)
        for i,formula in enumerate(formulas):
            print(f"evaluate factor formula: {i}, progress:{cnt}, total:{n}")
            try:
                bundle = self.simulator.simulate_formula(formula,profile_id,params)
                self._batch_process(bundle,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,rows,self_rankic_data,transform_spec)
            except Exception as e:
                print(f"evaluate formula failed: index={i}, error: {e}")
        return self._batch_finalize(rows,self_rankic_data,has_transform=transform_spec is not None)

    def evaluate_batch(self,
                       target,
                       profile_id:str=DEFAULT_PROFILE_ID,
                       params:dict=DEFAULT_PARAMS,
                       prod_corr:bool=False,
                       prod_corr_universe:str|None=None,
                       prod_corr_domain:str|None=None,
                       self_corr:bool=False,
                       transform_spec:dict[str,list|str]|None=None)->pd.DataFrame|tuple[pd.DataFrame,pd.DataFrame]:
        if isinstance(target,str):
            if os.path.isdir(target):
                return self._evaluate_batch_folder(target,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,self_corr,transform_spec)
            raise ValueError("evaluate_batch str target must be a folder path")

        if isinstance(target,dict):
            if all(isinstance(v,pd.DataFrame) for v in target.values()):
                return self._evaluate_batch_values(target,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,self_corr,transform_spec)
            raise ValueError("evaluate_batch dict target must be dict[str, DataFrame]")

        if isinstance(target,list):
            if len(target)==0:
                return (pd.DataFrame(),pd.DataFrame()) if self_corr else pd.DataFrame()
            if all(isinstance(item,dict) for item in target):
                return self._evaluate_batch_specs(target,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,self_corr,transform_spec)
            if all(isinstance(item,str) for item in target):
                return self._evaluate_batch_formulas(target,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,self_corr,transform_spec)
            raise ValueError("evaluate_batch list must be all dicts (specs) or all strings (formulas)")

        raise ValueError("unsupported target type for evaluate_batch")
