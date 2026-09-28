import os
import pandas as pd
import copy

from .core.factor_registry import FactorRegistry
from .core.factor_value_engine import FactorValueEngine
from .core.factor_value_store import FactorValueStore
from .core.factor_result_engine import FactorResultEngine
from .core.factor_result_store import FactorResultStore
from .core.factor_result_profiles import DEFAULT_PROFILE_ID
from .core.factor_performance_engine import DEFAULT_PARAMS,FactorPerformanceEngine
from .core.factor_value_transformer import FactorValueTransformer
from .core.factor_selector import FactorSelector
from .core.factor_criteria import FactorCriteria
from .core.factor_scorer import FactorScorer

from .components.factor_comparator import FactorComparator
from .components.factor_simulator import FactorSimulator
from .components.lib_interactor import LibInteractor
from .tools.factor_plotter import FactorPlotter
from .tools.factor_spec_builder import FactorSpecBuilder
from .tools.factor_spec_renderer import FactorSpecRenderer
from .tools.factor_template_builder import FactorTemplateBuilder

from .services.access_service import AccessService
from .services.research_service import ResearchService
from .services.manage_service import ManageService
from .services.select_service import SelectService

from .plugins.factor_portrait import FactorPortraitStore,FactorPortraitBuilder
from .plugins.factor_weekly_report import FactorWeeklyReporter
from .plugins.factor_pruner import FactorPruner

from quant.data_system.data_hub.main import DataManager

class FactorManager:
    def __init__(self,dm=None,base_dir="/home/workspace/common/quant/quant_factor"):
        """
        dm: 数据管理器类,必须提供get_data方法
        base_dir: 因子存储目录地址
        """
        
        dm = DataManager() if dm is None else dm
        self.dm = dm

        self.default_profile_id = DEFAULT_PROFILE_ID
        self.default_params = DEFAULT_PARAMS

        registry_path = os.path.join(base_dir,"factor_registry.json")
        registry = FactorRegistry(registry_path)

        value_engine = FactorValueEngine(dm,base_dir,registry)
        value_store = FactorValueStore(base_dir)
        result_engine = FactorResultEngine(dm)
        result_store = FactorResultStore(base_dir)
        perf_engine = FactorPerformanceEngine()

        selector_path = os.path.join(base_dir,"select_list.json")
        selector = FactorSelector(selector_path)

        criteria = FactorCriteria()
        scorer = FactorScorer()

        spec_builder = FactorSpecBuilder(dm,registry)
        spec_renderer = FactorSpecRenderer()
        template_builder = FactorTemplateBuilder()
        transformer = FactorValueTransformer(dm)
        portrait_store = FactorPortraitStore(base_dir)

        self.interactor = LibInteractor(
            dm,
            registry,
            value_engine,
            value_store,
            result_engine,
            result_store,
            perf_engine,
            selector
        )
        self.comparator = FactorComparator()
        plotter = FactorPlotter()
        self.simulator = FactorSimulator(
            value_engine,
            result_engine,
            perf_engine,
            spec_builder,
            spec_renderer,
            transformer,
        )

        portrait_builder = FactorPortraitBuilder(self.simulator,perf_engine)
        reporter = FactorWeeklyReporter(self.interactor,portrait_store,dm)

        self.access = AccessService(
            self.interactor,
            self.comparator,
            plotter,
            portrait_store,
            scorer)
        
        self.research = ResearchService(
            self.interactor,
            self.comparator,
            self.simulator,
            plotter,
            template_builder)
        
        factors_out_dir = os.path.join(base_dir,"factor_repo")
        self.manage = ManageService(
            self.interactor,
            self.comparator,
            self.simulator,
            factors_out_dir,
            portrait_builder,
            portrait_store,
            criteria)
        
        self.select = SelectService(
            self.interactor,
            self.comparator,
        )

        self.reporter = reporter
        
        self.pruner = FactorPruner(
            self.research,
            criteria,
            scorer
        )
    
    # property
    def get_default_profile_id(self):
        return self.default_profile_id
    
    def get_default_params(self):
        return copy.deepcopy(self.default_params)
    
    # research
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
        return self.research.evaluate(target,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,plot,transform_spec,silent)

    def evaluate_batch(self,
                       target,
                       profile_id:str=DEFAULT_PROFILE_ID,
                       params:dict=DEFAULT_PARAMS,
                       prod_corr:bool=False,
                       prod_corr_universe:str|None=None,
                       prod_corr_domain:str|None=None,
                       self_corr:bool=False,
                       transform_spec:dict[str,list|str]|None=None):
        return self.research.evaluate_batch(target,profile_id,params,prod_corr,prod_corr_universe,prod_corr_domain,self_corr,transform_spec)

    def create_template(self,
                        factor_type:str="regular",
                        out_dir:str|None=None,
                        **kwargs)->str:
        return self.research.create_template(factor_type,out_dir,**kwargs)

    def create_spec(self,factor_type:str="regular",**kwargs)->dict:
        return self.research.create_spec(factor_type,**kwargs)
    

    # access
    def list_factor_names(self,**kwargs)->list:
        return self.access.list_factor_names(**kwargs)

    def get_value(self,factor_name)->pd.DataFrame:
        return self.access.get_value(factor_name)
    
    def get_result(self,factor_name,profile_id:str=DEFAULT_PROFILE_ID)->dict:
        return self.access.get_result(factor_name,profile_id)

    def get_performance(self,
                        factor_name:str,
                        profile_id:str=DEFAULT_PROFILE_ID,
                        params:dict=DEFAULT_PARAMS,
                        prod_corr:bool=False):
        return self.access.get_performance(factor_name,profile_id,params,prod_corr)
    
    def get_portrait(self,factor_name:str)->dict:
        return self.access.get_portrait(factor_name)
    
    def plot_factor(self,
                    factor_name:str,
                    profile_id:str=DEFAULT_PROFILE_ID,
                    params:str=DEFAULT_PARAMS,
                    plot_size:tuple[float,float]=(6,4.5)):
        return self.access.plot_factor(factor_name,profile_id,params,plot_size)

    """
        batch method **kwargs:
        tier:Literal["selected","pool","all"]
        domain:Literal["pv","fundamental","hybrid"]
        type:Literal["regular","super"]
        level:Literal["minutes","days"]
        tag:str
        universe:QuantEnum.UniverseType
        category:QuantEnum.CategoryType
    """

    def get_value_batch(self,
                        factor_names:list[str]|None=None,
                        **kwargs)->dict:
        return self.access.get_value_batch(factor_names,**kwargs)

    def get_result_batch(self,
                         factor_names:list[str]|None=None,
                         profile_id:str=DEFAULT_PROFILE_ID,
                         **kwargs):
        return self.access.get_result_batch(factor_names,profile_id,**kwargs)

    def get_performance_batch(self,
                              factor_names:list[str]|None=None,
                              profile_id:str=DEFAULT_PROFILE_ID,
                              params:dict=DEFAULT_PARAMS,
                              prod_corr:bool=False,
                              **kwargs):
        return self.access.get_performance_batch(factor_names=factor_names,profile_id=profile_id,params=params,prod_corr=prod_corr,**kwargs)

    def get_corr(self,
                 factor_names:list[str]|None=None,
                 profile_id:str=DEFAULT_PROFILE_ID,
                 corr_method:str="rankics_corr",
                 plot:bool=True,
                 **kwargs):
        return self.access.get_corr(factor_names,profile_id,corr_method,plot,**kwargs)


    # manage
    def check_submission(self,file_path:str)->pd.DataFrame:
        return self.manage.check_submission(file_path)

    def submit(self,file_path:str)->dict:
        return self.manage.submit(file_path)

    def submit_batch(self,folder_path:str)->dict[str,dict]:
        return self.manage.submit_batch(folder_path)
    

    # select
    def select_add(self,factor_name:str):
        self.select.select_add(factor_name)

    def select_remove(self,factor_name:str):
        self.select.select_remove(factor_name)
    

    # other
    def generate_weekly_report(self,end_date:str|None=None,save_path:str|None=None)->dict:
        report = self.reporter.generate(end_date)
        if save_path:
            self.reporter.save(report,save_path)
        return report
    
    def clear_rankics_cache(self):
        self.interactor.clear_rankics_cache()
