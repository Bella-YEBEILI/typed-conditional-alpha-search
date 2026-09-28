import pandas as pd
from functools import lru_cache

from ..core.config import FACTOR_END_DATE
from ..core.factor_result_profiles import DEFAULT_PROFILE_ID
from ..core.factor_performance_engine import DEFAULT_PARAMS

from ..core.factor_registry import FactorRegistry
from ..core.factor_value_engine import FactorValueEngine
from ..core.factor_value_store import FactorValueStore
from ..core.factor_result_engine import FactorResultEngine
from ..core.factor_result_store import FactorResultStore
from ..core.factor_performance_engine import FactorPerformanceEngine
from ..core.factor_selector import FactorSelector

class LibInteractor:
    def __init__(self,
                 dm,
                 registry:FactorRegistry,
                 value_engine:FactorValueEngine,
                 value_store:FactorValueStore,
                 result_engine:FactorResultEngine,
                 result_store:FactorResultStore,
                 perf_engine:FactorPerformanceEngine,
                 selector:FactorSelector):
        self.dm = dm
        self.registry = registry
        self.value_engine = value_engine
        self.value_store = value_store
        self.result_engine = result_engine
        self.result_store = result_store
        self.perf_engine = perf_engine
        self.selector = selector

    """
    参数语义:
    get: 加载
    materialize: 落盘
    """

    # supplement
    def filter_names(self,factor_names:list[str]|None=None,**kwargs)->list[str]:
        if factor_names is None:
            factor_names = self.list_factor_names()

        # tier filter
        tier = kwargs.get("tier")
        if tier=="selected":
            selected = set(self.selector.list_selected())
            factor_names = [n for n in factor_names if n in selected]
        elif tier=="pool":
            selected = set(self.selector.list_selected())
            factor_names = [n for n in factor_names if n not in selected]

        # registry field filters
        registry_keys = {"type","author","level","tag","domain","category","universe"}
        filters = {k:v for k,v in kwargs.items() if k in registry_keys and v is not None}
        if filters:
            registry = self.list_factors()
            factor_names = [
                n for n in factor_names
                if n in registry and all(str(registry[n].get(k,""))==str(v) for k,v in filters.items())
            ]

        factor_names.sort()
        return factor_names

    # registry
    # TODO: 处理文件重名问题以及因子重名问题
    def register_factor(self,file_path:str)->dict: 
        return self.registry.register_factor(file_path)
    
    def list_factors(self)->dict[str,dict]:
        return self.registry.list_factors()

    def list_factor_names(self)->list[str]:
        data = self.list_factors()
        names = list(data.keys())
        names.sort()
        return names
    
    def has_factor(self,factor_name:str)->bool:
        return self.registry.has_factor(factor_name)

    def get_factor(self,factor_name:str)->dict:
        if not self.has_factor(factor_name):
            raise ValueError(f"factor not found in registry: {factor_name}")
        return self.registry.get_factor(factor_name)
    
    def delete_factor(self,factor_name:str):
        return self.registry.delete_factor(factor_name)

    # selector
    def list_selected(self)->list[str]:
        return self.selector.list_selected()

    def is_selected(self,factor_name:str)->bool:
        return self.selector.is_selected(factor_name)

    def select_add(self,factor_name:str)->None:
        if not self.has_factor(factor_name):
            raise ValueError(f"factor not found in pool: {factor_name}")
        self.selector.add(factor_name)
        print(f"successfully add {factor_name}")

    def select_remove(self,factor_name:str)->None:
        self.selector.remove(factor_name)

    def select_set(self,factor_names:list[str])->None:
        for name in factor_names:
            if not self.has_factor(name):
                raise ValueError(f"factor not found in pool: {name}")
        self.selector.set(factor_names)

    # value
    def has_value(self,factor_name:str)->bool:
        return self.value_store.has(factor_name)
    
    def get_value(self,factor_name:str)->pd.DataFrame:
        if not self.has_value(factor_name):
            raise FileNotFoundError(f"factor value not found: factor_name={factor_name}")
        return self.value_store.load(factor_name)

    def materialize_value(self,factor_name:str,end:str|None=None)->tuple[str,str]:
        if end is None:
            end = FACTOR_END_DATE

        if self.has_value(factor_name):
            v = self.get_value(factor_name)
            if v.index[-1]>=pd.Timestamp(end):
                print(f"factor {factor_name} already update to end {end}")
                return 

        factor_value = self.value_engine.calc_value_by_name(factor_name=factor_name,end=end)
        return self.value_store.save(factor_name,factor_value),pd.Timestamp(factor_value.index[-1]).strftime("%Y%m%d")
    
    def delete_value(self,factor_name:str):
        return self.value_store.delete(factor_name)

    # result
    def list_profile_ids(self,factor_name:str):
        return self.result_store.list_profile_ids(factor_name)
    
    def has_result(self,factor_name:str,profile_id:str=DEFAULT_PROFILE_ID)->bool:
        return self.result_store.has(factor_name,profile_id)
    
    def get_result(self,
                   factor_name:str,
                   profile_id:str=DEFAULT_PROFILE_ID,
                   persist_result:bool=False):
        if self.has_result(factor_name,profile_id):
            return self.result_store.load(factor_name,profile_id)
        
        factor_value = self.get_value(factor_name)
        factor_result = self.result_engine.calc_result(factor_value,profile_id)
        if persist_result:
            self.result_store.save(factor_name,profile_id,factor_result)
        return factor_result
    
    def materialize_result(self,factor_name:str,profile_id:str=DEFAULT_PROFILE_ID):
        factor_value = self.get_value(factor_name)
        factor_result = self.result_engine.calc_result(factor_value,profile_id)
        return self.result_store.save(factor_name,profile_id,factor_result)
    
    def delete_result(self,factor_name:str,profile_id:str=DEFAULT_PROFILE_ID):
        return self.result_store.delete(factor_name,profile_id)

    # performance
    def get_performance(self,
                        factor_name:str,
                        profile_id:str=DEFAULT_PROFILE_ID,
                        params:dict=DEFAULT_PARAMS,
                        persist_result:bool=False)->dict[str,float]:
        factor_result = self.get_result(factor_name,profile_id,persist_result)
        return self.perf_engine.calc_basic_performance(factor_result,params)

    # others
    @lru_cache(maxsize=16)
    def get_rankics_data(self,
                         factor_names:tuple[str,...]|None=None,
                         profile_id:str=DEFAULT_PROFILE_ID,
                         persist_result:bool=False,
                         tier:str|None=None,
                         universe:str|None=None,
                         domain:str|None=None)->dict[str,pd.Series]:
        """
        返回rankic序列字典
        这种用lru_cache修饰的函数不能传**kwargs!
        """
        kwargs = {}
        if tier is not None: kwargs["tier"] = tier
        if universe is not None: kwargs["universe"] = universe
        if domain is not None: kwargs["domain"] = domain

        names = list(factor_names) if factor_names is not None else None
        names = self.filter_names(names,**kwargs)
        
        data = {}
        for factor_name in names:
            result = self.get_result(factor_name,profile_id,persist_result)
            data[factor_name] = result.get("rankics",pd.Series(dtype=float))

        return data
    
    @lru_cache(maxsize=16)
    def get_exrets_data(self,
                        factor_names:tuple[str,...]|None=None,
                        profile_id:str=DEFAULT_PROFILE_ID,
                        persist_result:bool=False,
                        tier:str|None=None,
                        universe:str|None=None,
                        domain:str|None=None,
                        benchmark:str="zz1000s")->dict[str,pd.Series]:
        """
        返回毛超额收益序列字典
        """
        kwargs = {}
        if tier is not None: kwargs["tier"] = tier
        if universe is not None: kwargs["universe"] = universe
        if domain is not None: kwargs["domain"] = domain

        names = list(factor_names) if factor_names is not None else None
        names = self.filter_names(names,**kwargs)

        index_returns = self.dm.get_data("index_data")[benchmark+"_returns"]
        
        data = {}
        for factor_name in names:
            result = self.get_result(factor_name,profile_id,persist_result)
            ret = result.get("long_rets",pd.Series(dtype=float))
            index_ret,ret= index_returns.align(ret,join="inner")
            exret = ret-index_ret
            data[factor_name] = exret
        
        return data
    
    @lru_cache(maxsize=16)
    def get_exnetrets_data(self,
                           factor_names:tuple[str,...]|None=None,
                           profile_id:str=DEFAULT_PROFILE_ID,
                           persist_result:bool=False,
                           tier:str|None=None,
                           universe:str|None=None,
                           domain:str|None=None,
                           benchmark:str="zz1000s")->dict[str,pd.Series]:
        """
        返回净超额收益序列字典
        """
        kwargs = {}
        if tier is not None: kwargs["tier"] = tier
        if universe is not None: kwargs["universe"] = universe
        if domain is not None: kwargs["domain"] = domain

        names = list(factor_names) if factor_names is not None else None
        names = self.filter_names(names,**kwargs)

        index_returns = self.dm.get_data("index_data")[benchmark+"_returns"]
        
        data = {}
        for factor_name in names:
            result = self.get_result(factor_name,profile_id,persist_result)
            ret = result.get("long_rets",pd.Series(dtype=float))
            to = result.get("long_turnovers",pd.Series(dtype=float))
            netret = ret-to*0.0012/2
            index_ret,netret= index_returns.align(netret,join="inner")
            exret = netret-index_ret
            data[factor_name] = exret
        
        return data
    
    @lru_cache(maxsize=16)
    def get_ifdrawdown_data(self,
                            factor_names:tuple[str,...]|None=None,
                            profile_id:str=DEFAULT_PROFILE_ID,
                            persist_result:bool=False,
                            tier:str|None=None,
                            universe:str|None=None,
                            domain:str|None=None,
                            benchmark:str="zz1000s")->dict[str,pd.Series]:
        """
        毛超额收益是否处于回撤段
        """
        kwargs = {}
        if tier is not None: kwargs["tier"] = tier
        if universe is not None: kwargs["universe"] = universe
        if domain is not None: kwargs["domain"] = domain

        names = list(factor_names) if factor_names is not None else None
        names = self.filter_names(names,**kwargs)
        
        index_returns = self.dm.get_data("index_data")[benchmark+"_returns"]

        data = {}
        for factor_name in names:
            result = self.get_result(factor_name,profile_id,persist_result)
            ret = result.get("long_rets",pd.Series(dtype=float))
            index_ret,ret= index_returns.align(ret,join="inner")
            exret = ret-index_ret
            expnl = (1+exret).cumprod()
            dd = expnl/expnl.cummax()-1
            data[factor_name] = dd<0
        
        return data
    
    @lru_cache(maxsize=16)
    def get_ifnetdrawdown_data(self,
                               factor_names:tuple[str,...]|None=None,
                               profile_id:str=DEFAULT_PROFILE_ID,
                               persist_result:bool=False,
                               tier:str|None=None,
                               universe:str|None=None,
                               domain:str|None=None,
                               benchmark:str="zz1000s")->dict[str,pd.Series]:
        """
        净超额收益是否处于回撤段
        """
        kwargs = {}
        if tier is not None: kwargs["tier"] = tier
        if universe is not None: kwargs["universe"] = universe
        if domain is not None: kwargs["domain"] = domain

        names = list(factor_names) if factor_names is not None else None
        names = self.filter_names(names,**kwargs)
        
        index_returns = self.dm.get_data("index_data")[benchmark+"_returns"]

        data = {}
        for factor_name in names:
            result = self.get_result(factor_name,profile_id,persist_result)
            ret = result.get("long_rets",pd.Series(dtype=float))
            to = result.get("long_turnovers",pd.Series(dtype=float))
            netret = ret-to*0.0012/2
            index_ret,netret= index_returns.align(netret,join="inner")
            exret = netret-index_ret
            expnl = (1+exret).cumprod()
            dd = expnl/expnl.cummax()-1
            data[factor_name] = dd<0
        
        return data
    
    @lru_cache(maxsize=16)
    def get_ifretinvalid_data(self,
                              factor_names:tuple[str,...]|None=None,
                              profile_id:str=DEFAULT_PROFILE_ID,
                              persist_result:bool=False,
                              tier:str|None=None,
                              universe:str|None=None,
                              domain:str|None=None,
                              benchmark:str="zz1000s")->dict[str,pd.Series]:
        """
        毛超额收益是否小于0
        """
        kwargs = {}
        if tier is not None: kwargs["tier"] = tier
        if universe is not None: kwargs["universe"] = universe
        if domain is not None: kwargs["domain"] = domain

        names = list(factor_names) if factor_names is not None else None
        names = self.filter_names(names,**kwargs)
        
        index_returns = self.dm.get_data("index_data")[benchmark+"_returns"]

        data = {}
        for factor_name in names:
            result = self.get_result(factor_name,profile_id,persist_result)
            ret = result.get("long_rets",pd.Series(dtype=float))
            index_ret,ret= index_returns.align(ret,join="inner")
            exret = ret-index_ret
            data[factor_name] = exret<0
        
        return data
    
    @lru_cache(maxsize=16)
    def get_ifnetinvalid_data(self,
                              factor_names:tuple[str,...]|None=None,
                              profile_id:str=DEFAULT_PROFILE_ID,
                              persist_result:bool=False,
                              tier:str|None=None,
                              universe:str|None=None,
                              domain:str|None=None,
                              benchmark:str="zz1000s")->dict[str,pd.Series]:
        """
        净超额收益是否小于0
        """
        kwargs = {}
        if tier is not None: kwargs["tier"] = tier
        if universe is not None: kwargs["universe"] = universe
        if domain is not None: kwargs["domain"] = domain

        names = list(factor_names) if factor_names is not None else None
        names = self.filter_names(names,**kwargs)
        
        index_returns = self.dm.get_data("index_data")[benchmark+"_returns"]

        data = {}
        for factor_name in names:
            result = self.get_result(factor_name,profile_id,persist_result)
            ret = result.get("long_rets",pd.Series(dtype=float))
            to = result.get("long_turnovers",pd.Series(dtype=float))
            netret = ret-to*0.0012/2
            index_ret,netret= index_returns.align(netret,join="inner")
            exret = netret-index_ret
            data[factor_name] = exret<0
        
        return data
    



    def clear_rankics_cache(self):
        self.get_rankics_data.cache_clear()


    

    

    

