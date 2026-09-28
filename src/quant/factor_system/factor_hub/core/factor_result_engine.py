import numpy as np
import pandas as pd
from typing import Any

from .factor_result_profiles import PROFILES,DEFAULT_PROFILE_ID
from quant.quant_lib.analysis import div
from quant.quant_lib.numbafunc import cs_rank_2d,cs_corr_2d,cs_mean_2d,cs_mean_1d

class FactorResultEngine:
    """
    根据因子值+profile_id计算因子结果
    """
    def __init__(self,dm):
        self.dm = dm
        self.data:dict[str,pd.DataFrame] = {}

    def _get_data(self,fld):
        data = self.data.get(fld)
        if data is None:
            data = self.dm.get_data(fld)
            self.data[fld] = data
        return data
    
    def _get_adj_opens(self):
        adj_opens = self.data.get("adj_opens")
        if adj_opens is None:
            opens = self._get_data("opens")
            adj_factors = self._get_data("adj_factors")
            adj_opens = opens*adj_factors
            self.data["adj_opens"] = adj_opens
        return adj_opens
    
    def _get_adj_closes(self):
        adj_closes = self.data.get("adj_closes")
        if adj_closes is None:
            closes = self._get_data("closes")
            adj_factors = self._get_data("adj_factors")
            adj_closes = closes*adj_factors
            self.data["adj_closes"] = adj_closes
        return adj_closes
    
    def _get_o2c(self):
        o2c = self.data.get("o2c")
        if o2c is None:
            adj_opens = self._get_adj_opens()
            adj_closes = self._get_adj_closes()
            o2c = div(adj_closes,adj_opens)-1
            self.data["o2c"] = o2c
        return o2c
    
    def _get_o2o(self):
        o2o = self.data.get("o2o")
        if o2o is None:
            adj_opens = self._get_adj_opens()
            o2o = div(adj_opens,adj_opens.shift(1))-1
            self.data["o2o"] = o2o
        return o2o
    
    def _get_c2o(self):
        c2o = self.data.get("c2o")
        if c2o is None:
            adj_opens = self._get_adj_opens()
            adj_closes = self._get_adj_closes()
            c2o = div(adj_opens,adj_closes.shift(1))-1
            self.data["c2o"] = c2o
        return c2o
    
    def _get_c2c(self):
        c2c = self.data.get("c2c")
        if c2c is None:
            adj_closes = self._get_adj_closes()
            c2c = div(adj_closes,adj_closes.shift(1))-1
            self.data["c2c"] = c2c
        return c2c

    def calc_result(self,factor_value:pd.DataFrame,profile_id:str=DEFAULT_PROFILE_ID)->dict[str,Any]:
        if not isinstance(factor_value,pd.DataFrame):
            raise ValueError("factor_value must be DataFrame")
        
        profile = PROFILES.get(profile_id)
        if profile is None:
            raise ValueError("profile id not exists")
        
        # params&mode
        buypoint = profile["buypoint"]
        buylag = profile["buylag"]
        sellpoint = profile["sellpoint"]
        selllag = profile["selllag"]
        if buylag>selllag:
            raise ValueError("sell lag cannot be less than buy lag")
        if buylag==selllag:
            if buypoint==sellpoint or (buypoint=="closes" and sellpoint=="opens"):
                raise ValueError("buy point must before sell point")
        universe_name = profile["universe"]
        qt = profile["qt"]
        # TODO: 扩展基于buy/sell point/lag的灵活回测逻辑
        if (buypoint=="opens") and (sellpoint=="opens") and (buylag==1) and (selllag==2):
            mode = 1
        elif (buypoint=="closes") and (sellpoint=="closes") and (buylag==1) and (selllag==2):
            mode = 2
        else:
            raise ValueError("unsupported profile")
        
        # factor
        universe = self._get_data(universe_name).astype(bool)
        factor_value,universe = factor_value.align(universe,join="inner",axis=0)
        factor_value = factor_value.where(universe)
        factor_df = factor_value.shift(1).iloc[1:]

        # data
        dates = factor_df.index
        stocks = factor_df.columns # all dfs have shared the same stocks
        nd = len(dates)
        ns = len(stocks)
        tradables = self._get_data("tradables").reindex(index=dates).to_numpy(dtype=bool,copy=False)
        fv1 = factor_df.to_numpy(dtype=float,copy=False) # factor array for precise ret
        fv2 = factor_df.shift(1).to_numpy(dtype=float,copy=False) # factor array for group ret and rank ic
        fr1 = cs_rank_2d(fv1,pct=True,ascending=True)
        fr2 = cs_rank_2d(fv2,pct=True,ascending=True)
        if mode==1:
            c2o = self._get_c2o().reindex(index=dates).to_numpy(dtype=float,copy=False)
            o2c = self._get_o2c().reindex(index=dates).to_numpy(dtype=float,copy=False)
            o2o = self._get_o2o().reindex(index=dates).to_numpy(dtype=float,copy=False)
            rv = o2o
            buylimit = self._get_data("limit_up_cto").reindex(index=dates).to_numpy(dtype=bool,copy=False)
            selllimit = self._get_data("limit_down_cto").reindex(index=dates).to_numpy(dtype=bool,copy=False)
        elif mode==2:
            c2c = self._get_c2c().reindex(index=dates).to_numpy(dtype=float,copy=False)
            rv = c2c
            buylimit = self._get_data("limit_up_ctc").reindex(index=dates).to_numpy(dtype=bool,copy=False)
            selllimit = self._get_data("limit_down_ctc").reindex(index=dates).to_numpy(dtype=bool,copy=False)
        
        result_dict = {}

        # long_rets/long_turnovers/long_nums
        """
        对于mode==1: 
        t日的target和real指t日开盘后理论持有/实际持有的持仓 
        t日的rets指t-1日收盘->t日收盘这段的收益 
        t日的turnovers指t日开盘买卖总量/t日开盘前的持仓 
        t日的nums指t日开盘后持有的量 
        对于mode==2: 
        t日的target和real指t日收盘后理论持有/实际持有的持仓 
        t日的rets指t-1日收盘->t日收盘这段的收益 
        t日的turnovers指t日收盘买卖总量/t日收盘前的持仓 
        t日的nums指t日收盘后持有的量
        """
        target = fr1>=1-qt
        if mode==1:
            rets_c2o = np.zeros(nd,dtype=float)
            rets_o2c = np.zeros(nd,dtype=float)
            turnovers = np.zeros(nd,dtype=float)
            nums = np.zeros(nd,dtype=float)
            prev = np.zeros(ns,dtype=bool)
            for i in range(nd):
                tgt = target[i]
                buyables = tradables[i]&(~buylimit[i])
                sellables = tradables[i]&(~selllimit[i])
                to_buy = (~prev)&tgt&buyables
                to_sell = prev&(~tgt)&sellables
                real = (prev&(~to_sell))|to_buy 
                n_prev = prev.sum()
                n_real = real.sum()
                if n_prev>0:
                    rets_c2o[i] = cs_mean_1d(c2o[i,prev])
                    turnovers[i] = (to_buy.sum()+to_sell.sum())/n_prev
                if n_real>0:
                    rets_o2c[i] = cs_mean_1d(o2c[i,real])
                    nums[i] = n_real
                prev = real
            rets = (1+rets_c2o)*(1+rets_o2c)-1
        elif mode==2:
            rets = np.zeros(nd,dtype=float)
            turnovers = np.zeros(nd,dtype=float)
            nums = np.zeros(nd,dtype=float)
            prev = np.zeros(ns,dtype=bool)
            for i in range(nd):
                tgt = target[i]
                buyables = tradables[i]&(~buylimit[i])
                sellables = tradables[i]&(~selllimit[i])
                to_buy = (~prev)&tgt&buyables
                to_sell = prev&(~tgt)&sellables
                real = (prev&(~to_sell))|to_buy 
                n_prev = prev.sum()
                n_real = real.sum()
                if n_prev>0:
                    rets[i] = cs_mean_1d(c2c[i,prev])
                    turnovers[i] = (to_buy.sum()+to_sell.sum())/n_prev
                if n_real>0:
                    nums[i] = n_real
                prev = real
        result_dict["long_rets"] = pd.Series(rets,index=dates,dtype=float)
        result_dict["long_turnovers"] = pd.Series(turnovers,index=dates,dtype=float)
        result_dict["long_nums"] = pd.Series(nums,index=dates,dtype=float)

        # short_rets/short_turnovers/short_nums
        """
        空头镜像:
        1. 收益取负
        2. 开空仓: 不能做空跌停股; 平空仓: 不能平仓涨停股
        """
        target = fr1<=qt
        if mode==1:
            rets_c2o = np.zeros(nd,dtype=float)
            rets_o2c = np.zeros(nd,dtype=float)
            turnovers = np.zeros(nd,dtype=float)
            nums = np.zeros(nd,dtype=float)
            prev = np.zeros(ns,dtype=bool)
            for i in range(nd):
                tgt = target[i]
                openables = tradables[i]&(~selllimit[i])
                closeables = tradables[i]&(~buylimit[i])
                to_open = (~prev)&tgt&openables
                to_close = prev&(~tgt)&closeables
                real = (prev&(~to_close))|to_open 
                n_prev = prev.sum()
                n_real = real.sum()
                if n_prev>0:
                    rets_c2o[i] = -cs_mean_1d(c2o[i,prev])
                    turnovers[i] = (to_open.sum()+to_close.sum())/n_prev
                if n_real>0:
                    rets_o2c[i] = -cs_mean_1d(o2c[i,real])
                    nums[i] = n_real
                prev = real
            rets = (1+rets_c2o)*(1+rets_o2c)-1
        elif mode==2:
            rets = np.zeros(nd,dtype=float)
            turnovers = np.zeros(nd,dtype=float)
            nums = np.zeros(nd,dtype=float)
            prev = np.zeros(ns,dtype=bool)
            for i in range(nd):
                tgt = target[i]
                openables = tradables[i]&(~selllimit[i])
                closeables = tradables[i]&(~buylimit[i])
                to_open = (~prev)&tgt&openables
                to_close = prev&(~tgt)&closeables
                real = (prev&(~to_close))|to_open 
                n_prev = prev.sum()
                n_real = real.sum()
                if n_prev>0:
                    rets[i] = -cs_mean_1d(c2c[i,prev])
                    turnovers[i] = (to_open.sum()+to_close.sum())/n_prev
                if n_real>0:
                    nums[i] = n_real
                prev = real
        result_dict["short_rets"] = pd.Series(rets,index=dates,dtype=float)
        result_dict["short_turnovers"] = pd.Series(turnovers,index=dates,dtype=float)
        result_dict["short_nums"] = pd.Series(nums,index=dates,dtype=float)

        # group ret
        """
        group ret_t定义为
        从t-1日换仓点到t日换仓点的粗略分组收益
        """
        group_rets = {}
        for i in range(10):
            l = i/10
            u = (i+1)/10 if i != 9 else 1+1e-9
            group_mask = (fr2>=l)&(fr2<u)
            group_rv = np.where(group_mask,rv,np.nan)
            group_rets[f"group_{i}"] = pd.Series(cs_mean_2d(group_rv),index=dates,dtype=float)
        result_dict["group_rets"] = pd.DataFrame(group_rets)

        # ic
        mask = np.isfinite(fv2)&np.isfinite(rv)
        fv_for_ic = np.where(mask,fv2,np.nan)
        rv_for_ic = np.where(mask,rv,np.nan)
        ics = cs_corr_2d(fv_for_ic,rv_for_ic)
        result_dict["ics"] = pd.Series(ics,index=dates,dtype=float)

        # rankic
        fr = cs_rank_2d(fv_for_ic,pct=True,ascending=True)
        rr = cs_rank_2d(rv_for_ic,pct=True,ascending=True)
        rankics = cs_corr_2d(fr,rr) 
        result_dict["rankics"] = pd.Series(rankics,index=dates,dtype=float)

        # precision
        fr_top = fr>=1-qt
        rr_top = rr>=1-qt
        hit = (fr_top&rr_top).sum(axis=1)
        total = fr_top.sum(axis=1)
        with np.errstate(invalid="ignore"):
            precisions = np.where(total>0,hit/total,np.nan)
        result_dict["precisions"] = pd.Series(precisions,index=dates,dtype=float)

        # coverage
        result_dict["coverages"] = factor_value.notna().sum(axis=1)/universe.sum(axis=1)

        return result_dict
        



                    


