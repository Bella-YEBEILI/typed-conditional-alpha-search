import pandas as pd
from quant.quant_lib.analysis import *
from .config import STRATEGY_START_DATE,STRATEGY_END_DATE

SUPPORTED_POINT = ["opens",
                   "open_vwaps_1m",
                   "open_vwaps_5m",
                   "open_vwaps_10m",
                   "open_vwaps_15m",
                   "open_vwaps_30m",
                   "vwaps"]

class StrategyResultEngine:
    def __init__(self,dm):
        self.dm = dm

    def calc_result(self,
                    weight:pd.DataFrame,
                    trade_point:str="open_vwaps_30m",
                    buy_fr:float=0.0001,
                    sell_fr:float=0.0006,
                    start:str|None=None,
                    end:str|None=None):
        if start is None:
            start = STRATEGY_START_DATE
        if end is None:
            end = STRATEGY_END_DATE

        if trade_point not in SUPPORTED_POINT:
            raise ValueError(f"unsupported point: {trade_point}")

        # 加载价格数据
        adj = self.dm.get_data("adj_factors")
        p1 = self.dm.get_data(trade_point)*adj     
        p2 = self.dm.get_data("closes")*adj  

        # 加载可交易性掩码
        limup,limdown = self.dm.get_data("limit_up_cto"),self.dm.get_data("limit_down_cto")
        tradables = self.dm.get_data("tradables").astype(bool)

        # 对齐索引
        dates = weight.index
        if start is not None:
            dates = dates[dates>=pd.Timestamp(start)]
        if end is not None:
            dates = dates[dates<=pd.Timestamp(end)]

        weight = weight.reindex(index=dates)         # 当日换仓后理论持有的权重
        p1 = p1.reindex(index=dates)                 # 换仓价格
        p2 = p2.reindex(index=dates)                 # 净值结算价格
        limup = limup.reindex(index=dates)          
        limdown = limdown.reindex(index=dates)
        tradables = tradables.reindex(index=dates)
        b = tradables&(~limup)                       # 可买掩码
        s = tradables&(~limdown)                     # 可卖掩码

        dates = weight.index
        n = len(weight.columns)

        # 转numpy并预处理权重
        w = weight.to_numpy(dtype=np.float64)

        p1 = p1.to_numpy(dtype=np.float64)
        p2 = p2.to_numpy(dtype=np.float64)
        b = b.to_numpy(dtype=bool)
        s = s.to_numpy(dtype=bool)

        # 初始化维护变量
        amounts = np.zeros(n)
        cash = 1.0

        # 初始化容器
        pnl = np.empty(len(dates))
        pre_amounts = np.empty(len(dates))
        buy_amounts = np.empty(len(dates))
        sell_amounts = np.empty(len(dates))
        effs = np.empty(len(dates))

        # 逐日回测 
        for i in range(len(dates)):
            # 取切片
            target = np.nan_to_num(w[i,:],nan=0.0)
            sum_weight = np.sum(target)
            if sum_weight>0:
                target = target/sum_weight
            buyable = b[i,:]
            sellable = s[i,:]

            if i==0:
                # 交易点
                buy = np.where(buyable,target,0.0)
                buy_amount = np.sum(buy)
                if buy_amount*(1+buy_fr)>1:
                    buy = buy/(buy_amount*(1+buy_fr))
                    buy_amount = 1/(1+buy_fr)
                amounts = amounts+buy
                buy_cost = buy_amount*buy_fr
                cash = cash-buy_amount-buy_cost

                # 结算点
                tp = p1[i,:]
                cp = p2[i,:]
                ratio = np.ones_like(cp,dtype=np.float64)
                np.divide(cp,tp,out=ratio,where=(np.isfinite(cp))&(np.isfinite(tp))&(tp>0))
                amounts = amounts*ratio
                aum = np.sum(amounts)+cash

                # 结算
                pnl[i] = aum
                pre_amounts[i] = 0.0
                buy_amounts[i] = buy_amount
                sell_amounts[i] = 0.0
                effs[i] = np.sum(amounts>0)
            
            else:
                pre_cp = p2[i-1,:]
                tp = p1[i,:]
                cp = p2[i,:]
                # 交易点
                ratio = np.ones_like(tp,dtype=np.float64)
                ratio = np.divide(tp,pre_cp,out=ratio,where=(np.isfinite(tp))&(np.isfinite(pre_cp))&(pre_cp>0))
                amounts = amounts*ratio
                pre_amount = np.sum(amounts)
                aum = pre_amount+cash
                target_amounts = aum*target
                # 先卖
                to_sell = np.clip(amounts-target_amounts,0.0,None)
                sell = np.where(sellable,to_sell,0.0)
                amounts = amounts-sell
                sell_amount = np.sum(sell)
                sell_cost = sell_amount*sell_fr
                cash = cash+sell_amount-sell_cost
                # 再买
                to_buy = np.clip(target_amounts-amounts,0.0,None)
                buy = np.where(buyable,to_buy,0.0)
                buy_amount = np.sum(buy)
                if buy_amount*(1+buy_fr)>cash:
                    buy = buy*cash/(buy_amount*(1+buy_fr))
                    buy_amount = cash/(1+buy_fr)
                amounts = amounts+buy
                buy_cost = buy_amount*buy_fr
                cash = cash-buy_amount-buy_cost

                # 结算点
                ratio = np.ones_like(cp,dtype=np.float64)
                np.divide(cp,tp,out=ratio,where=(np.isfinite(cp))&(np.isfinite(tp))&(tp>0))
                amounts = amounts*ratio
                aum = np.sum(amounts)+cash

                # 结算
                pnl[i] = aum
                pre_amounts[i] = pre_amount
                buy_amounts[i] = buy_amount
                sell_amounts[i] = sell_amount
                effs[i] = np.sum(amounts>0)
        
        # 换手率
        turnovers = np.zeros_like(pre_amounts)
        np.divide(buy_amounts+sell_amounts,pre_amounts,out=turnovers,where=np.isfinite(pre_amounts)&(pre_amounts!=0))

        # 按同样逻辑重算费前pnl
        amounts = np.zeros(n,dtype=np.float64)
        cash = 1.0
        gross_pnl = np.empty(len(dates),dtype=np.float64)
        for i in range(len(dates)):
            target = np.nan_to_num(w[i,:],nan=0.0)
            sum_weight = np.sum(target)
            if sum_weight>0:
                target = target/sum_weight
            buyable = b[i,:]
            sellable = s[i,:]
            if i==0:
                buy = np.where(buyable,target,0.0)
                amounts = amounts+buy
                buy_amount = np.sum(buy)
                cash = cash-buy_amount
                tp = p1[i,:]
                cp = p2[i,:]
                ratio = np.ones_like(cp,dtype=np.float64)
                np.divide(cp,tp,out=ratio,where=(np.isfinite(cp))&(np.isfinite(tp))&(tp>0))
                amounts = amounts*ratio
                aum = np.sum(amounts)+cash
                gross_pnl[i] = aum
            else:
                pre_cp = p2[i-1,:]
                tp = p1[i,:]
                cp = p2[i,:]
                ratio = np.ones_like(tp,dtype=np.float64)
                ratio = np.divide(tp,pre_cp,out=ratio,where=(np.isfinite(tp))&(np.isfinite(pre_cp))&(pre_cp>0))
                amounts = amounts*ratio
                pre_amount = np.sum(amounts)
                aum = pre_amount+cash
                target_amounts = aum*target
                to_sell = np.clip(amounts-target_amounts,0.0,None)
                sell = np.where(sellable,to_sell,0.0)
                amounts = amounts-sell
                sell_amount = np.sum(sell)
                cash = cash+sell_amount
                to_buy = np.clip(target_amounts-amounts,0.0,None)
                buy = np.where(buyable,to_buy,0.0)
                buy_amount = np.sum(buy)
                if buy_amount>cash:
                    buy = buy*cash/buy_amount
                    buy_amount = cash
                amounts = amounts+buy
                cash = cash-buy_amount
                ratio = np.ones_like(cp,dtype=np.float64)
                np.divide(cp,tp,out=ratio,where=(np.isfinite(cp))&(np.isfinite(tp))&(tp>0))
                amounts = amounts*ratio
                aum = np.sum(amounts)+cash
                gross_pnl[i] = aum

        return {
            "gross_pnl":pd.Series(gross_pnl,index=dates),
            "pnl":pd.Series(pnl,index=dates),
            "turnovers":pd.Series(turnovers,index=dates),
            "effs":pd.Series(effs,index=dates)
        }
