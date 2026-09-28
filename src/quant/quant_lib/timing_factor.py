import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from quant.data_system.data_hub.main import DataManager

DEFAULT_PARAMS = {
    "start":"20160101",
    "end":"20251231"
}

INDEX_NAMES = {"hs300s","zz500s","zz800s","zz1000s"}


class TimingFactor:
    DEFAULT_PROPERTY = {
        "factor_name":None,
        "signal_type":"",# c,b
        "main_index":"zz1000s",
        "aux_index":"hs300s",
        "basedir":"/home/workspace/users/liangxiwen/tqstrategyserver/TQStrategyServer/timing_factors",
        "data_needed":[],
    }

    def __init__(self,dm:DataManager|None=None):
        self.dm = DataManager() if dm is None else dm
        self.factor_property = dict(self.DEFAULT_PROPERTY)
        self.factor_property.update(self.set_factor_property())

        if self.factor_property.get("factor_name") is None:
            self.factor_property["factor_name"] = self.__class__.__name__
        self.factor_name = self.factor_property["factor_name"]

        signal_type = self.factor_property.get("signal_type")
        if signal_type not in {"c","b"}:
            raise ValueError("signal_type must be 'c' or 'b'")
        self.signal_type = signal_type

        self.main_index = self.factor_property["main_index"]
        self.aux_index = self.factor_property["aux_index"]

        self.basedir = self.factor_property["basedir"]
        os.makedirs(os.path.join(self.basedir,"c","values"),exist_ok=True)
        os.makedirs(os.path.join(self.basedir,"b","values"),exist_ok=True)

        self.data_needed = {}

    def set_factor_property(self)->dict:
        return {}

    def calc_factor(self):
        raise NotImplementedError
    
    def load_data(self):
        index_returns = self.dm.get_data("index_data")
        self.main_returns = index_returns[self.main_index+"_returns"]
        self.aux_returns = index_returns[self.aux_index+"_returns"]

        self.data_needed = {}
        for fld in self.factor_property["data_needed"]:
            self.data_needed[fld] = self.dm.get_data(fld)

    def run(self,params:dict|None=None,plot:bool=True):
        params = {**DEFAULT_PARAMS,**(params or {})}
        start = params["start"]
        end = params["end"]

        self.load_data()
        self.calc_factor()

        self.factor_weekly = self.factor_data.resample("W-FRI").last().dropna()
        # TODO: 默认每周最后一个交易日收盘出信号后可以立即调仓（无指数开盘价格）
        self.main_returns_weekly = ((1+self.main_returns).resample("W-FRI").prod(min_count=1)-1).dropna()
        self.aux_returns_weekly = ((1+self.aux_returns).resample("W-FRI").prod(min_count=1)-1).dropna()

        self.calc_factor_performance(start,end)
        if plot:
            self.plot_factor()


    def _run_binary(self,start:str,end:str):
        signal = self.factor_weekly.fillna(0).astype(bool)

        future_main = self.main_weekly_ret.shift(-1)
        future_aux = self.aux_weekly_ret.shift(-1)

        df = pd.concat(
            {
                "signal":signal,
                "future_main":future_main,
                "future_aux":future_aux
            },
            axis=1
        ).dropna()

        df = df.loc[start:end]
        sig = df["signal"].astype(bool)

        self.factor_weekly = sig
        self.benchmark_returns = df["future_main"]
        self.strategy_returns = df["future_main"]+0.5*sig.astype(float)*(df["future_aux"]-df["future_main"])
        self.ex_returns = self.strategy_returns-self.benchmark_returns

        self.benchmark_pnl = (1+self.benchmark_returns).cumprod()
        self.strategy_pnl = (1+self.strategy_returns).cumprod()
        self.ex_pnl = (1+self.ex_returns).cumprod()

    def calc_factor_performance(self,start,end):
        factor = self.factor_weekly.shift(1).loc[start:end]

        main_rets = self.main_returns_weekly.loc[start:end]
        aux_rets = self.aux_returns_weekly.loc[start:end]
        diff_rets = aux_rets-main_rets

        df = pd.concat({"factor":factor,
                        "main_rets":main_rets,
                        "aux_rets":aux_rets,
                        "diff_rets":diff_rets},axis=1).dropna(how="any")

        if self.signal_type=="c":
            avg_ic = df["factor"].corr(df["diff_rets"])
            yearly_ic = df.groupby(df.index.year).apply(lambda x:x["factor"].corr(x["diff_rets"]))

            n_group = min(10,len(df))
            groups = pd.qcut(df["factor"].rank(method="first"),q=n_group,labels=False)
            group_returns = df.groupby(groups)["diff_rets"].mean()
            group_returns.index = [f"group_{int(i)+1}" for i in group_returns.index]

            self.factor_performance = {"avg_ic":avg_ic,
                                       "yearly_ic":yearly_ic,
                                       "group_returns":group_returns}
        
        elif self.signal_type=="b":
            self.benchmark_rets = df["main_rets"]
            self.strategy_rets = df["future_main"]+0.5*df["factor"].astype(float)*(df["future_diff"])
            self.ex_rets = self.strategy_rets-self.benchmark_rets
            
            n = len(self.ex_rets)
            if n==0:
                self.factor_performance = {}
                return

            ex_pnl = (1+self.ex_rets).cumprod()
            ex_nav = ex_pnl.iloc[-1]

            ann_ex_ret = ex_nav**(52/n)-1
            ex_std = self.ex_rets.std()
            ex_ir = np.nan if ex_std==0 else ann_ex_ret/ex_std*np.sqrt(52)
            dds = ex_pnl/ex_pnl.cummax()-1
            ex_maxdd = -dds.min()
            win_rate = (self.ex_rets>0).mean()

            self.factor_performance = {
                "ex_ret":ann_ex_ret,
                "ex_ir":ex_ir,
                "ex_maxdd":ex_maxdd,
                "win_rate":win_rate,
            }


    def plot_factor(self):
        if self.signal_type=="c":
            yearly_ic = self.factor_performance["yearly_ic"]
            group_returns = self.factor_performance["group_returns"]

            _,axes = plt.subplots(2,1,figsize=(4,3),constrained_layout=True)
            yearly_ic.plot(kind="bar",ax=axes[0],title="Yearly IC")
            group_returns.plot(kind="bar",ax=axes[1],title="Group Future Diff Return")

            for ax in axes:
                ax.grid(True,alpha=0.3)
                ax.tick_params(axis="x",labelrotation=30,labelsize=8)
                ax.tick_params(axis="y",labelsize=8)
            plt.show()

        elif self.signal_type=="b":
            self.benchmark_pnl = (1+self.benchmark_rets).cumprod()
            self.strategy_pnl = (1+self.strategy_rets).cumprod()
            self.ex_pnl = (1+self.ex_rets).cumprod()

            _,ax = plt.subplots(1,1,figsize=(4,3),constrained_layout=True)
            ax.plot(self.benchmark_pnl.index,self.benchmark_pnl.values,label="benchmark")
            ax.plot(self.strategy_pnl.index,self.strategy_pnl.values,label="timing")
            ax.plot(self.ex_pnl.index,self.ex_pnl.values,label="excess")

            ax.legend(fontsize=8)
            ax.grid(True,alpha=0.3)
            ax.tick_params(axis="x",labelrotation=30,labelsize=8)
            ax.tick_params(axis="y",labelsize=8)
            plt.show()

    def save(self):
        if self.factor_data is None:
            self.run(plot=False)

        subdir = "c" if self.signal_type=="c" else "b"
        save_path = os.path.join(self.basedir,subdir,"values",self.factor_name+".pkl")
        if os.path.exists(save_path):
            raise ValueError("factor already exists")
        self.factor_data.to_pickle(save_path)

    def calc_corr(self)->pd.Series|None:
        if self.factor_data is None:
            self.run(plot=False)

        subdir = "c" if self.signal_type=="c" else "b"
        values_dir = os.path.join(self.basedir,subdir,"values")

        factors = {}
        for path in os.listdir(values_dir):
            if not path.endswith(".pkl"):
                continue
            factor_name = path[:-4]
            file_path = os.path.join(values_dir,path)
            s = pd.read_pickle(file_path)
            s = self._to_series(s,factor_name)
            factors[factor_name] = self._to_weekly_last(s)

        df = pd.DataFrame(factors)
        target = self.factor_weekly

        if self.signal_type=="c":
            return df.corrwith(target)

        df = df.fillna(0).astype(float)
        target = target.fillna(0).astype(float)
        return df.corrwith(target)
            
