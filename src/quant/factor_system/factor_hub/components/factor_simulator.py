import pandas as pd
import time
import os

from quant.factor_system.factor_hub.core.factor_result_engine import FactorResultEngine

from ..core.factor_loader import load_factor_module
from ..core.factor_value_engine import FactorValueEngine
from ..core.factor_result_profiles import DEFAULT_PROFILE_ID
from ..core.factor_performance_engine import FactorPerformanceEngine,DEFAULT_PARAMS
from ..core.factor_value_transformer import FactorValueTransformer
from ..tools.factor_spec_builder import FactorSpecBuilder
from ..tools.factor_spec_renderer import FactorSpecRenderer

class FactorSimulator:
    def __init__(self,
                 value_engine:FactorValueEngine,
                 result_engine:FactorResultEngine,
                 perf_engine:FactorPerformanceEngine,
                 spec_builder:FactorSpecBuilder,
                 spec_renderer:FactorSpecRenderer,
                 transformer:FactorValueTransformer):
        self.value_engine = value_engine
        self.result_engine = result_engine
        self.perf_engine = perf_engine
        self.spec_builder = spec_builder
        self.spec_renderer = spec_renderer
        self.transformer = transformer

    def simulate_value(self,
                       factor_value:pd.DataFrame,
                       factor_name:str|None=None,
                       profile_id:str=DEFAULT_PROFILE_ID,
                       params:dict=DEFAULT_PARAMS)->dict:
        if not isinstance(factor_value,pd.DataFrame):
            raise ValueError("factor value must be dataframe")
        
        t1 = time.perf_counter()
        factor_result = self.result_engine.calc_result(factor_value,profile_id)
        t2 = time.perf_counter()
        print("calc factor result time: ",t2-t1,"sec")
        
        t1 = time.perf_counter()
        factor_perf = self.perf_engine.calc_basic_performance(factor_result,params)
        t2 = time.perf_counter()
        print("calc factor performance time: ",t2-t1,"sec")

        return {
            "id":"unknown" if factor_name is None else factor_name,
            "factor_value":factor_value,
            "factor_result":factor_result,
            "factor_performance":factor_perf,
        }
    
    def simulate_pkl_file(self,
                          pkl_path:str,
                          profile_id:str=DEFAULT_PROFILE_ID,
                          params:dict=DEFAULT_PARAMS)->dict:
        if not pkl_path.endswith(".pkl"):
            raise ValueError("pkl path must ends with .pkl")
        
        t1 = time.perf_counter()
        factor_value = pd.read_pickle(pkl_path)
        t2 = time.perf_counter()
        print("load factor value time: ",t2-t1,"sec")

        bundle = self.simulate_value(factor_value=factor_value,
                                     profile_id=profile_id,
                                     params=params)
        
        return{
            "id":os.path.splitext(os.path.basename(pkl_path))[0],
            "factor_value":factor_value,
            "factor_result":bundle["factor_result"],
            "factor_performance":bundle["factor_performance"]
        }
        
    
    def simulate_py_file(self,
                         py_path:str,
                         profile_id:str=DEFAULT_PROFILE_ID,
                         params:dict=DEFAULT_PARAMS)->dict:
        m = load_factor_module(py_path)
        factor_name = m.META["factor_name"]
        universe = m.SETTING.get("universe")
        domain = m.SETTING.get("domain")

        t1 = time.perf_counter()
        factor_value = self.value_engine.calc_value_by_path(py_path)
        t2 = time.perf_counter()
        print("calc factor value time: ",t2-t1,"sec")

        t1 = time.perf_counter()
        factor_result = self.result_engine.calc_result(factor_value,profile_id)
        t2 = time.perf_counter()
        print("calc factor result time: ",t2-t1,"sec")
        
        t1 = time.perf_counter()
        factor_perf = self.perf_engine.calc_basic_performance(factor_result,params)
        t2 = time.perf_counter()
        print("calc factor performance time: ",t2-t1,"sec")

        return {
            "id":factor_name,
            "universe":universe,
            "domain":domain,
            "factor_value":factor_value,
            "factor_result":factor_result,
            "factor_performance":factor_perf,
        }

    def simulate_spec(self,
                      spec:dict,
                      profile_id:str=DEFAULT_PROFILE_ID,
                      params:dict=DEFAULT_PARAMS)->dict:
        built_spec = self.spec_builder.build_from_dict(spec)
        temp_file_path = self.spec_renderer.render_temp_file(built_spec)

        try:
            return self.simulate_py_file(temp_file_path,profile_id,params)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def simulate_formula(self,
                         formula:str,
                         profile_id:str=DEFAULT_PROFILE_ID,
                         params:dict=DEFAULT_PARAMS)->dict:
        built_spec = self.spec_builder.build_from_formula(formula)
        temp_file_path = self.spec_renderer.render_temp_file(built_spec)

        try:
            return self.simulate_py_file(temp_file_path,profile_id,params)
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def simulate_transformed(self,
                             factor_value:pd.DataFrame,
                             transform_type:str,
                             transform_key:str,
                             profile_id:str=DEFAULT_PROFILE_ID,
                             params:dict=DEFAULT_PARAMS)->dict:
        if self.transformer is None:
            raise ValueError("transformer is required for simulate_transformed")
        if transform_type=="subuniverse":
            tv = self.transformer.subuniverse_transform(factor_value,transform_key)
        elif transform_type=="neutralize":
            tv = self.transformer.neutralize_transform(factor_value,transform_key)
        else:
            raise ValueError(f"unsupported transform_type: {transform_type}")
        factor_result = self.result_engine.calc_result(tv,profile_id)
        factor_perf = self.perf_engine.calc_basic_performance(factor_result,params)
        return {
            "factor_result":factor_result,
            "factor_performance":factor_perf,
        }
