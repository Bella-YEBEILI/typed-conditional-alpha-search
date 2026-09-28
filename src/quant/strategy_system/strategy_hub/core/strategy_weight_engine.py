import pandas as pd
import yaml
from typing import Literal
from quant.quant_lib.analysis import *

from .config import STRATEGY_START_DATE,STRATEGY_END_DATE

class StrategyWeightEngine:
    def __init__(self,dm,base_dir:str):
        self.dm = dm
        self.base_dir = base_dir

    @staticmethod
    def buffer_select(df:pd.DataFrame,n1:int,n2:int)->pd.DataFrame: # n1:组合大小,n2:缓冲区大小
        dates = df.index
        codes = df.columns
        mask = pd.DataFrame(False,index=dates,columns=codes,dtype=bool)
        buffer_th = n2
        current = set()
        for i,d in enumerate(dates):
            score = df.loc[d]
            rank = score.rank(method="first",ascending=False)
            if i==0:
                current = set(rank.nsmallest(n1).index)
            else:
                new_current = set()
                # 保留仍在缓冲区内的
                for c in current:
                    if rank[c]<=buffer_th:
                        new_current.add(c)
                # 若不足n1则补入
                if len(new_current)<n1:
                    ranked = rank.sort_values()
                    for c in ranked.index:
                        if c not in new_current:
                            new_current.add(c)
                        if len(new_current)>=n1:
                            break
                current = new_current
            mask.loc[d,list(current)] = True
        return mask.astype(bool)
    
    @staticmethod
    def cap_weight_and_redistribute(w:np.ndarray,cap:float=0.05)->np.ndarray:
        """
        单票权重不超过 cap
        超出部分均分给当前权重严格小于 cap 的股票,迭代至无超限。
        若封顶后无「未满 cap」的接收方(例如正权重股数 < 20 且均被压到 0.05),会提前退出导致总权重 < 1
        因此最后做一次归一化，保证总权重 = 1(仅在正权重数很少时，归一化后可能有个别权重略超 cap)
        """
        w = np.asarray(w,dtype=float).copy()
        while True:
            over = w>cap
            if not over.any():
                break
            total_excess = (w[over]-cap).sum()
            w = np.minimum(w,cap)
            below_cap = (w>0)&(w<cap)
            n_recipient = int(below_cap.sum())
            if n_recipient==0:
                break
            w = np.where(below_cap,w+total_excess/n_recipient,w)
        s = w.sum()
        if s>1e-12 and s<1.0:
            w = w/s
        return w

    def inverse_vol_weight(self,df:pd.DataFrame,win:int,wmax:float)->pd.DataFrame:
        # df: bool 持仓矩阵,日期x股票
        ret = self.dm.get_data("ctc_returns")
        vol = ts_std(ret,win).reindex_like(df)
        vol = vol.replace(0,np.nan)

        mask = df.astype(bool)

        inv_vol = (1.0/vol).replace([np.inf,-np.inf],np.nan)
        inv_vol = inv_vol.where(mask,other=0.0).fillna(0.0)

        rowsum = inv_vol.sum(axis=1).replace(0,1.0)
        w = inv_vol.div(rowsum,axis=0)

        w_out = pd.DataFrame(0.0,index=w.index,columns=w.columns,dtype=np.float64)
        for dt in w.index:
            row_mask = mask.loc[dt]
            if not row_mask.any():
                continue
            # 只对持仓股票做 cap+再分配
            row = w.loc[dt,row_mask].to_numpy(dtype=float,copy=True)
            if row.sum()<=0:
                continue
            row = StrategyWeightEngine.cap_weight_and_redistribute(row,cap=wmax)
            # 填回对应股票
            w_out.loc[dt,row_mask] = row

        return w_out

    def generate_weight(self,
                        signal:pd.DataFrame,
                        universe:str,
                        method:Literal["custom","optimizer"],
                        weight_config_name:str,
                        start:str|None=None,
                        end:str|None=None)->pd.DataFrame:
        if start is None:
            start = STRATEGY_START_DATE
        if end is None:
            end = STRATEGY_END_DATE

        # 选股池
        universe = self.dm.get_data(universe)
        all_dates = universe.index

        # 打掩码
        signal,universe = signal.align(universe,join="inner",axis=0)
        signal = signal.where(universe)

        # 开始时间：start的前一天（若有）
        pos = signal.index.searchsorted(pd.Timestamp(start),side="left")
        if pos>0:
            signal = signal.iloc[pos-1:]

        yaml_path = os.path.join(self.base_dir,"weight_yamls",weight_config_name+".yaml")
        
        # 信号->掩码->权重
        if method=="custom":
            with open(yaml_path,"r",encoding="utf-8") as f:
                config = yaml.safe_load(f)

            number = config.get("number")
            if number is None:
                raise ValueError("yaml file missing key number")
            mask_method = config.get("mask_method")
            if mask_method is None:
                raise ValueError("yaml file missing key mask_method")
            weight_method = config.get("weight_method")
            if weight_method is None:
                raise ValueError("yaml file missing key mask_method")
            
            if mask_method=="simple":
                mask = (cs_rank(signal,ascending=False,pct=False)<=number)
            elif mask_method=="buffer":
                buffer_size = config.get("buffer_size")
                if buffer_size is None:
                    raise ValueError("yaml file missing key buffer_size")
                mask = StrategyWeightEngine.buffer_select(signal,number,buffer_size)
            
            if weight_method=="eqw":
                w = mask.astype(float)
            elif weight_method=="invvol":
                vol_window = config.get("vol_window")
                if vol_window is None:
                    raise ValueError("yaml file missing key vol_window")
                w = self.inverse_vol_weight(mask,win=vol_window,wmax=0.05)
            
        elif method=="optimizer":
            # TODO: 在这里加使用优化器的逻辑
            return
            
        # 调整权重 
        w = w.clip(lower=0.0)
        rowsum = w.sum(axis=1).replace(0,1.0)
        w = w.div(rowsum,axis=0)

        # 边界调整: 调成第二天开盘生效的持仓 
        if w.index[-1]==all_dates[-1]:
            w = w.shift(1).iloc[1:]
        else:
            last_dt = w.index[-1]
            loc = all_dates.get_loc(last_dt)
            next_dt = all_dates[loc+1]
            w = pd.concat([w.shift(1).iloc[1:],pd.DataFrame([w.iloc[-1].to_numpy()],index=[next_dt],columns=w.columns)],axis=0)

        return w.loc[:end]