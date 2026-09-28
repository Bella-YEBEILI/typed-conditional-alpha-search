import matplotlib.pyplot as plt
import pandas as pd
from typing import Any

from ..core.factor_result_profiles import DEFAULT_PROFILE_ID
from ..core.factor_performance_engine import DEFAULT_PARAMS

from ..components.lib_interactor import LibInteractor
from ..components.factor_comparator import FactorComparator
from ..tools.factor_plotter import FactorPlotter
from ..core.factor_scorer import FactorScorer
from ..plugins.factor_portrait import FactorPortraitStore

CORR_METHODS = ["rankics_corr",
                "exrets_corr",
                "exnetrets_corr",
                "exdd_overlap",
                "exnetdd_overlap",
                "exinvalid_overlap",
                "exnetinvalid_overlap"
                ]

class AccessService:
    def __init__(self,
                 interactor:LibInteractor,
                 comparator:FactorComparator,
                 plotter:FactorPlotter,
                 portrait_store:FactorPortraitStore,
                 scorer:FactorScorer):
        self.interactor = interactor
        self.comparator = comparator
        self.plotter = plotter
        self.portrait_store = portrait_store
        self.scorer = scorer
    
    # info
    def list_factors(self)->dict[str,dict]:
        return self.interactor.list_factors()

    def list_factor_names(self,**kwargs)->list[str]:
        factor_names = self.interactor.list_factor_names()
        return self.interactor.filter_names(factor_names,**kwargs)
    
    def get_factor(self,factor_name:str)->dict:
        return self.interactor.get_factor(factor_name)
    
    # value
    def get_value(self,factor_name:str)->pd.DataFrame:
        return self.interactor.get_value(factor_name)
    
    def get_value_batch(self,
                        factor_names:list[str]|None=None,
                        **kwargs)->dict[str,pd.DataFrame]:
        value_dict = {}
        factor_names = self.interactor.filter_names(factor_names,**kwargs)
        for factor_name in factor_names:
            try:
                value_dict[factor_name] = self.get_value(factor_name)
            except Exception as e:
                print(f"get value failed: factor_name={factor_name}, error: {e}")
        return value_dict
    
    # result
    def get_result(self,factor_name:str,profile_id:str=DEFAULT_PROFILE_ID)->dict[str,Any]:
        return self.interactor.get_result(factor_name,profile_id)
    
    def get_result_batch(self,
                         factor_names:list[str]|None=None,
                         profile_id:str=DEFAULT_PROFILE_ID,
                         **kwargs)->dict[str,dict]:
        result_dict = {}
        factor_names = self.interactor.filter_names(factor_names,**kwargs)
        for factor_name in factor_names:
            try:
                result_dict[factor_name] = self.get_result(factor_name,profile_id)
            except Exception as e:
                print(f"get result failed: factor_name={factor_name}, profile_id={profile_id}, error: {e}")
        return result_dict
    
    # performance
    def get_performance(self,
                        factor_name:str,
                        profile_id:str=DEFAULT_PROFILE_ID,
                        params:dict=DEFAULT_PARAMS,
                        prod_corr:bool=False)->dict[str,Any]:
        perf = self.interactor.get_performance(
            factor_name,
            profile_id=profile_id,
            params=params,
            persist_result=False,
        )

        info = self.get_factor(factor_name)
        universe = info["universe"]
        domain = info["domain"]
        perf["score"] = self.scorer.score(perf,universe,domain)

        if not prod_corr:
            return perf

        factor_result = self.interactor.get_result(
            factor_name,
            profile_id=profile_id,
            persist_result=False,
        )
        rankics_series = factor_result.get("rankics",pd.Series(dtype=float))

        rankics_data = self.interactor.get_rankics_data(
            profile_id=profile_id,
            persist_result=False,
            tier="selected",
            universe=universe,
            domain=domain
        )

        corr_s = self.comparator.check_prod_corr(lib_data=rankics_data,new_data=rankics_series)
        if factor_name in corr_s.index:
            corr_s = corr_s.drop(index=factor_name)
        corr_s = corr_s.dropna()
        if len(corr_s)==0:
            perf["max_corr"] = pd.NA
            perf["max_corr_factor_name"] = pd.NA
            perf["avg_corr"] = pd.NA
            return perf
        
        perf["max_corr"] = corr_s.iloc[0]
        perf["max_corr_factor_name"] = corr_s.index[0]
        perf["avg_corr"] = corr_s.mean()
        return perf
    
    def get_performance_batch(self,
                              factor_names:list[str]|None=None,
                              profile_id:str=DEFAULT_PROFILE_ID,
                              params:dict=DEFAULT_PARAMS,
                              prod_corr:bool=False,
                              **kwargs)->pd.DataFrame:
        factor_names = self.interactor.filter_names(factor_names,**kwargs)

        rows = []
        for factor_name in factor_names:
            perf = self.get_performance(factor_name,profile_id,params,prod_corr)
            row = dict(perf)
            row["factor_name"] = factor_name
            rows.append(row)

        if len(rows)==0:
            return pd.DataFrame()

        perf_df = pd.DataFrame(rows).set_index("factor_name")
        return perf_df
    
    # portrait
    def get_portrait(self,factor_name:str)->dict:
        return self.portrait_store.load(factor_name)

    # other
    def plot_factor(self,
                    factor_name:str,
                    profile_id:str=DEFAULT_PROFILE_ID,
                    params:dict=DEFAULT_PARAMS,
                    plot_size:tuple[float,float]=(6,4.5)):
        factor_result = self.get_result(factor_name,profile_id)
        self.plotter.plot_result(id=factor_name,factor_result=factor_result,params=params,plot_size=plot_size)

    # TODO: 1. 不同角度刻画相关性 2. 热力图绘制 3. 区分benchmark
    def get_corr(self,
                 factor_names:list[str]|None=None,
                 profile_id=DEFAULT_PROFILE_ID,
                 corr_method:str="rankics_corr",
                 plot:bool=True,
                 **kwargs)->tuple[pd.DataFrame,pd.DataFrame]:
        factor_names = self.interactor.filter_names(factor_names,**kwargs)
        factor_names = tuple(factor_names)
        tier = kwargs.get("tier",None)
        universe = kwargs.get("universe",None)
        domain = kwargs.get("domain",None)
        benchmark = kwargs.get("benchmark","zz1000s")

        if corr_method not in CORR_METHODS:
            raise ValueError("unsupported corr method")

        if corr_method=="rankics_corr":
            data = self.interactor.get_rankics_data(factor_names=factor_names,
                                                    profile_id=profile_id,
                                                    tier=tier,
                                                    universe=universe,
                                                    domain=domain)
            corr_df,corr_stats = self.comparator.check_self_corr(data,"PEARSON")

        elif corr_method=="exrets_corr":
            data = self.interactor.get_exrets_data(factor_names=factor_names,
                                                   profile_id=profile_id,
                                                   tier=tier,
                                                   universe=universe,
                                                   domain=domain,
                                                   benchmark=benchmark)
            corr_df,corr_stats = self.comparator.check_self_corr(data,"PEARSON")

        elif corr_method=="exnetrets_corr":
            data = self.interactor.get_exnetrets_data(factor_names=factor_names,
                                                      profile_id=profile_id,
                                                      tier=tier,
                                                      universe=universe,
                                                      domain=domain,
                                                      benchmark=benchmark)
            corr_df,corr_stats = self.comparator.check_self_corr(data,"PEARSON")

        elif corr_method=="exdd_overlap":
            data = self.interactor.get_ifdrawdown_data(factor_names=factor_names,
                                                       profile_id=profile_id,
                                                       tier=tier,
                                                       universe=universe,
                                                       domain=domain,
                                                       benchmark=benchmark)
            corr_df,corr_stats = self.comparator.check_self_corr(data,"JACCARD")

        elif corr_method=="exnetdd_overlap":
            data = self.interactor.get_ifnetdrawdown_data(factor_names=factor_names,
                                                          profile_id=profile_id,
                                                          tier=tier,
                                                          universe=universe,
                                                          domain=domain,
                                                          benchmark=benchmark)
            corr_df,corr_stats = self.comparator.check_self_corr(data,"JACCARD")

        elif corr_method=="exinvalid_overlap":
            data = self.interactor.get_ifretinvalid_data(factor_names=factor_names,
                                                         profile_id=profile_id,
                                                         tier=tier,
                                                         universe=universe,
                                                         domain=domain,
                                                         benchmark=benchmark)
            corr_df,corr_stats = self.comparator.check_self_corr(data,"JACCARD")

        elif corr_method=="exnetinvalid_overlap":
            data = self.interactor.get_ifnetinvalid_data(factor_names=factor_names,
                                                         profile_id=profile_id,
                                                         tier=tier,
                                                         universe=universe,
                                                         domain=domain,
                                                         benchmark=benchmark)
            corr_df,corr_stats = self.comparator.check_self_corr(data,"JACCARD")

        if plot and len(corr_df)>0:
            n = len(corr_df)
            fig_w = min(18,max(6,n*0.35))
            fig_h = min(18,max(5,n*0.35))
            fig,ax = plt.subplots(figsize=(fig_w,fig_h),constrained_layout=True)
            if corr_method.endswith("_corr"):
                im = ax.imshow(corr_df.to_numpy(dtype=float),vmin=-1,vmax=1,cmap="coolwarm")
                cbar = fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
            else:
                plot_values = corr_df.to_numpy(dtype=float)*2-1
                im = ax.imshow(plot_values,vmin=-1,vmax=1,cmap="coolwarm")
                cbar = fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04,ticks=[-1,0,1])
                cbar.ax.set_yticklabels(["0","0.5","1"])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(corr_method)
            plt.show()
        
        return corr_df,corr_stats
