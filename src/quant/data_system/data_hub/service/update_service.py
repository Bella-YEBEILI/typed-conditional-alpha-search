import os
import pandas as pd
import numpy as np
import h5py
from quant.quant_lib.analysis import *
from ..core.fundamental_updater import FundamentalUpdater
from ..core.fundamental_neutralizer import FundamentalNeutralizer
from ..core.safe_pickle_writer import SafePickleWriter

# TODO: 回来改下代码

class UpdateService:
    def __init__(self,
                 source_dir:str,
                 base_dir:str,
                 fu:FundamentalUpdater,
                 fn:FundamentalNeutralizer):
        self.source_dir = source_dir
        self.base_dir = base_dir
        self.fu = fu
        self.fn = fn

    def move_and_fix_datas(self):
        src = self.source_dir
        dst = self.base_dir
        groups = {
            "pv":["adj_factors","opens","highs","lows","closes",
                "ctc_returns","cto_returns","oto_returns",
                "volumes","amounts","vwaps","turnovers",
                "float_market_caps","total_market_caps"],
            "limit":["limit_up_ctc","limit_up_cto","limit_down_ctc","limit_down_cto"],
            "universe":["liquidity_amounts_60d_top75pct_min10M","stables","universe_liq_div3y",
                        "sz50s","hs300s","zz500s","zz800s","zz1000s","zzhls"],
            "status":["tradables","st_stocks","days_since_ipos"],
            "styles/raw":["size","nlsize","mom","rev1","rev3","rev5",
                        "vol20","vol60","vol240","liq20","liq60","liq240","beta"],
            "industry":["industrys"],
            "index_data":["index_data"]
        }
        cutoff = pd.Timestamp("20150101")
        for sub,names in groups.items():
            out_dir = os.path.join(dst,sub)
            os.makedirs(out_dir,exist_ok=True)
            for n in names:
                fp = os.path.join(src,f"{n}.pkl")
                if not os.path.isfile(fp):
                    print(f"SKIP: {fp} not found")
                    continue
                df = pd.read_pickle(fp)
                df.index = pd.to_datetime(df.index.astype(str))
                df = df.loc[cutoff:]
                if sub=="limit" or sub=="universe":
                    df = df.fillna(0).astype(bool)
                
                out_fp = os.path.join(out_dir,f"{n}.pkl")
                SafePickleWriter.safe_to_pickle(df,out_fp)
                print(f"OK: {out_fp}  shape={df.shape}")

    def build_hfq_datas(self):
        dst = os.path.join(self.base_dir,"pv")
        adj = pd.read_pickle(os.path.join(dst,"adj_factors.pkl"))
        for name in ["opens","highs","lows","closes"]:
            df = pd.read_pickle(os.path.join(dst,f"{name}.pkl"))
            hfq = df*adj.reindex(index=df.index,columns=df.columns)
            out_fp = os.path.join(dst,f"hfq_{name}.pkl")
            SafePickleWriter.safe_to_pickle(hfq,out_fp)
            print(f"OK: {out_fp}  shape={hfq.shape}")

    def build_standards(self):
        dst = os.path.join(self.base_dir,"universe")
        stables = pd.read_pickle(os.path.join(dst,"stables.pkl"))
        liq = pd.read_pickle(os.path.join(dst,"liquidity_amounts_60d_top75pct_min10M.pkl"))
        standards = stables&liq
        SafePickleWriter.safe_to_pickle(standards,os.path.join(dst,"standards.pkl"))
        print(f"OK: standards.pkl  shape={standards.shape}")

    def build_universe_A1800_zzhls(self):
        dst = os.path.join(self.base_dir,"universe")
        zz1000s = pd.read_pickle(os.path.join(dst,"zz1000s.pkl"))
        zz500s = pd.read_pickle(os.path.join(dst,"zz500s.pkl"))
        hs300s = pd.read_pickle(os.path.join(dst,"hs300s.pkl"))
        zzhls = pd.read_pickle(os.path.join(dst,"zzhls.pkl"))
        universe_A1800_zzhls = (zz1000s|zz500s|hs300s|zzhls).astype(bool)
        SafePickleWriter.safe_to_pickle(universe_A1800_zzhls,os.path.join(dst,"universe_A1800_zzhls.pkl"))
        print(f"OK: universe_A1800_zzhls.pkl  shape={universe_A1800_zzhls.shape}")

    def build_top1800s(self):
        dst = os.path.join(self.base_dir,"universe")
        zz1000s = pd.read_pickle(os.path.join(dst,"zz1000s.pkl"))
        zz500s = pd.read_pickle(os.path.join(dst,"zz500s.pkl"))
        hs300s = pd.read_pickle(os.path.join(dst,"hs300s.pkl"))
        top1800s = (zz1000s|zz500s|hs300s).astype(bool)
        SafePickleWriter.safe_to_pickle(top1800s,os.path.join(dst,"top1800s.pkl"))
        print(f"OK: universe_A1800_zzhls.pkl  shape={top1800s.shape}")

    def build_universe_liq_div3y_A1800(self):
        dst = os.path.join(self.base_dir,"universe")
        standards = pd.read_pickle(os.path.join(dst,"universe_liq_div3y.pkl"))
        zzhls = pd.read_pickle(os.path.join(dst,"zzhls.pkl"))
        zz1000s = pd.read_pickle(os.path.join(dst,"zz1000s.pkl"))
        zz500s = pd.read_pickle(os.path.join(dst,"zz500s.pkl"))
        hs300s = pd.read_pickle(os.path.join(dst,"hs300s.pkl"))
        universe_liq_div3y_A1800 = (standards|zz1000s|zz500s|hs300s|zzhls).astype(bool)
        SafePickleWriter.safe_to_pickle(universe_liq_div3y_A1800,os.path.join(dst,"universe_liq_div3y_A1800.pkl"))
        print(f"OK: universe_liq_div3y_A1800.pkl  shape={universe_liq_div3y_A1800.shape}")

    def build_smalls(self):
        dst = os.path.join(self.base_dir,"universe")
        standards = pd.read_pickle(os.path.join(dst,"standards.pkl"))
        hs300s = pd.read_pickle(os.path.join(dst,"hs300s.pkl"))
        zz500s = pd.read_pickle(os.path.join(dst,"zz500s.pkl"))
        smalls = standards&(~hs300s)&(~zz500s)
        SafePickleWriter.safe_to_pickle(smalls,os.path.join(dst,"smalls.pkl"))
        print(f"OK: smalls.pkl  shape={smalls.shape}")

    @staticmethod
    def _calc_beta(ret:pd.DataFrame,cap:pd.DataFrame,window:int,min_periods:int)->pd.DataFrame:
        mkt_ret = (ret*cap).sum(axis=1)/cap.sum(axis=1)
        ret_mean = ret.rolling(window=window,min_periods=min_periods).mean()
        mkt_mean = mkt_ret.rolling(window=window,min_periods=min_periods).mean()
        mkt_var = mkt_ret.rolling(window=window,min_periods=min_periods).var(ddof=0)
        cov = ret.mul(mkt_ret,axis=0).rolling(window=window,min_periods=min_periods).mean()-ret_mean.mul(mkt_mean,axis=0)
        return cov.div(mkt_var,axis=0).replace([np.inf,-np.inf],np.nan)

    def fix_long_window_factors(self):

        src = self.source_dir
        dst = self.base_dir

        standards = pd.read_pickle(os.path.join(dst,"universe/standards.pkl"))
        ret = pd.read_pickle(os.path.join(src,"ctc_returns.pkl"))
        cap = pd.read_pickle(os.path.join(src,"float_market_caps.pkl"))
        to = pd.read_pickle(os.path.join(src,"turnovers.pkl"))
        for df in [ret,cap,to]:
            if not isinstance(df.index,pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index.astype(str))

        raw_dir = os.path.join(dst,"styles/raw")
        targets = {
            "vol240": lambda w,m: ret.rolling(window=w,min_periods=m).std(),
            "liq240": lambda w,m: np.log(to.rolling(window=w,min_periods=m).mean().pipe(lambda x:x.where(x>0))),
            "beta": lambda w,m: UpdateService._calc_beta(ret,cap,w,m),
        }
        for name,calc_fn in targets.items():
            original = pd.read_pickle(os.path.join(raw_dir,f"{name}.pkl")).reindex(index=standards.index)
            ver_a = calc_fn(120,60).reindex(index=standards.index)
            ver_b = calc_fn(120,20).reindex(index=standards.index)
            # step 1: fill with ver_a where original is NaN and in standards
            mask_a = original.isna()&standards&ver_a.notna()
            filled = original.copy()
            filled = filled.where(~mask_a,ver_a)
            cnt_a = mask_a.sum().sum()
            # step 2: fill remaining NaN with ver_b
            mask_b = filled.isna()&standards&ver_b.notna()
            filled = filled.where(~mask_b,ver_b)
            cnt_b = mask_b.sum().sum()
            out_fp = os.path.join(raw_dir,f"{name}.pkl")
            SafePickleWriter.safe_to_pickle(filled,out_fp)
            print(f"FIX: {name} filled {cnt_a} cells (A:120/60) + {cnt_b} cells (B:120/20) -> {out_fp}")


    def build_derived_styles(self):
        raw_dir = os.path.join(self.base_dir,"styles","raw")
        out_dir = os.path.join(self.base_dir,"styles","derived")
        os.makedirs(out_dir,exist_ok=True)
        cap = pd.read_pickle(os.path.join(self.base_dir,"pv","float_market_caps.pkl"))
        standards = pd.read_pickle(os.path.join(self.base_dir,"universe","standards.pkl")).astype(bool)
        a1800 = pd.read_pickle(os.path.join(self.base_dir,"universe","universe_liq_div3y_A1800.pkl")).astype(bool)
        names = ["size","nlsize","mom","rev1","rev3","rev5",
                "vol20","vol60","vol240","liq20","liq60","liq240","beta"]
        processed = {}
        for n in names:
            df = pd.read_pickle(os.path.join(raw_dir,f"{n}.pkl"))
            s_mask = standards.reindex(index=df.index,columns=df.columns).fillna(False)
            a_mask = a1800.reindex(index=df.index,columns=df.columns).fillna(False)
            mask = s_mask | a_mask
            df = df.where(mask)
            df = df.clip(lower=df.quantile(0.05,axis=1),upper=df.quantile(0.95,axis=1),axis=0)
            df = cs_weighted_zscore(df,cap.reindex(index=df.index,columns=df.columns))
            processed[n] = df
            print(f"processed: {n}  shape={df.shape}")
        factors = {
            "Size":processed["size"],
            "Nlsize":processed["nlsize"],
            "Mom":processed["mom"],
            "Rev":(processed["rev1"]+processed["rev3"]+processed["rev5"])/3,  
            "Vol":(processed["vol20"]+processed["vol60"]+processed["vol240"])/3,
            "Liq":(processed["liq20"]+processed["liq60"]+processed["liq240"])/3,
            "Beta":processed["beta"],
        }
        for fname,df in factors.items():
            fp = os.path.join(out_dir,f"{fname}.pkl")
            SafePickleWriter.safe_to_pickle(df,fp)
            print(f"OK: {fp}  shape={df.shape}")
    
    @staticmethod
    @njit(cache=True,parallel=True)
    def _vwap_one_day(amounts,volumes):
        """
        amounts,volumes: 2D array, shape=(n_minutes,n_stocks)
        返回: 1D vwap, shape=(n_stocks,)
        """
        nm,ns = amounts.shape
        out = np.full(ns,np.nan,dtype=np.float32)
        for j in prange(ns):
            s_amt = 0.0
            s_vol = 0.0
            for i in range(nm):
                a = amounts[i,j]
                v = volumes[i,j]
                if np.isfinite(a) and np.isfinite(v) and v>0:
                    s_amt += a
                    s_vol += v
            if s_vol>0:
                out[j] = s_amt/s_vol
        return out
    
    @staticmethod
    def _generate_vwap(f,startminute:str,endminute:str)->pd.DataFrame:
        """
        从衍生 hdf5 中计算 [startminute,endminute] 区间的 VWAP:
        vwap(date,stock) = sum(amounts)/sum(volumes)
        行轴: axis/dates, 列轴: axis/stocks
        """
        # 读取轴
        h5_dates = f["axis/dates"][:].astype(str)
        h5_minutes = f["axis/minutes"][:].astype(str)
        h5_stocks = f["axis/stocks"][:].astype(str)
        nd = len(h5_dates)
        ns = len(h5_stocks)
        # 获取分钟区间下标
        start = np.where(h5_minutes==startminute)[0]
        end = np.where(h5_minutes==endminute)[0]
        mi0 = int(start[0])
        mi1 = int(end[0])
        # 结果 (ndates x nstocks)
        out = np.full((nd,ns),np.nan,dtype=np.float32)
        # 逐日循环计算
        for di in range(nd):
            amts = f["data/amounts"][di,mi0:mi1+1,:]
            vols = f["data/volumes"][di,mi0:mi1+1,:]
            out[di,:] = UpdateService._vwap_one_day(amts,vols)
        # 返回DataFrame
        idx_dates = pd.to_datetime(h5_dates)
        idx_stocks = pd.Index(h5_stocks.astype(str))
        df = pd.DataFrame(out,index=idx_dates,columns=idx_stocks,dtype=np.float32)
        return df

    def build_vwaps(self):
        hdf5_path = "/home/workspace/common/hdf5/all_minute_data.h5"
        dst = os.path.join(self.base_dir,"minute_vwaps")
        os.makedirs(dst,exist_ok=True)

        all_dates = pd.read_pickle(os.path.join(self.source_dir,"all_dates.pkl"))
        all_stocks = pd.read_pickle(os.path.join(self.source_dir,"all_stocks.pkl"))

        cutoff = pd.Timestamp("20150101")

        with h5py.File(hdf5_path,"r") as f:
            print("generating VWAP ...")
            open_vwaps_1m = UpdateService._generate_vwap(f,"0931","0931").reindex(index=all_dates,columns=all_stocks).loc[cutoff:]
            open_vwaps_5m = UpdateService._generate_vwap(f,"0931","0935").reindex(index=all_dates,columns=all_stocks).loc[cutoff:]
            open_vwaps_10m = UpdateService._generate_vwap(f,"0931","0940").reindex(index=all_dates,columns=all_stocks).loc[cutoff:]
            open_vwaps_15m = UpdateService._generate_vwap(f,"0931","0945").reindex(index=all_dates,columns=all_stocks).loc[cutoff:]
            open_vwaps_30m = UpdateService._generate_vwap(f,"0931","1000").reindex(index=all_dates,columns=all_stocks).loc[cutoff:]

        for m,data in zip(["1m","5m","10m","15m","30m"],[open_vwaps_1m,open_vwaps_5m,open_vwaps_10m,open_vwaps_15m,open_vwaps_30m,]):
            print(f"open vwaps {m} shape: {data.shape}")
            print(f"open vwaps {m} index: {data.index.min()} -> {data.index.max()}")

            out_fp = os.path.join(dst,f"open_vwaps_{m}.pkl")
            print(f"saved: {out_fp}")
            SafePickleWriter.safe_to_pickle(data,out_fp)

    def build_derived_pvs(self):
        # 衍生短中期量价特征构造
        dst = os.path.join(self.base_dir,"pv_derived")
        os.makedirs(dst,exist_ok=True)

        c = pd.read_pickle(os.path.join(self.base_dir,"pv","hfq_closes.pkl"))
        v = pd.read_pickle(os.path.join(self.base_dir,"pv","volumes.pkl"))
        a = pd.read_pickle(os.path.join(self.base_dir,"pv","amounts.pkl"))
        r = pd.read_pickle(os.path.join(self.base_dir,"pv","ctc_returns.pkl"))

        # 价格
        ema5 = ema1(c,5)
        ema20 = ema1(c,20)

        # 成交量
        adv5 = ts_mean(v,5)
        adv20 = ts_mean(v,20)

        # 波动率
        vol5 = ts_std(r,5)
        vol20 = ts_std(r,20)

        # 量价相关
        pvc5 = ts_corr(c,v,5)
        pvc20 = ts_corr(c,v,20)

        # 量价相关2
        rvc5 = ts_corr(r,diff(log(v)),5)
        rvc20 = ts_corr(r,diff(log(v)),20)

        # 成交效率
        te =  div(abs(r),a)

        for n,data in zip(
            ["pvc5","pvc20","ema5","ema20","adv5","adv20","vol5","vol20","rvc5","rvc20","te"],
            [pvc5,pvc20,ema5,ema20,adv5,adv20,vol5,vol20,rvc5,rvc20,te],
        ):
            print(f"{n} shape: {data.shape}")
            print(f"{n} index: {data.index.min()} -> {data.index.max()}")
            
            out_fp = os.path.join(dst,f"{n}.pkl")
            print(f"saved: {out_fp}")
            SafePickleWriter.safe_to_pickle(data,out_fp)

    @staticmethod
    def _rolling_zscore_past(x:np.ndarray,window:int=20,min_periods:int=5)->np.ndarray:
        x = np.asarray(x,dtype=np.float32)
        valid = np.isfinite(x)
        x0 = np.where(valid,x,0.0).astype(np.float32,copy=False)
        x2 = (x0*x0).astype(np.float32,copy=False)
        zero_f = np.zeros((1,x.shape[1]),dtype=np.float32)
        zero_i = np.zeros((1,x.shape[1]),dtype=np.int32)
        cs = np.vstack((zero_f,np.cumsum(x0,axis=0,dtype=np.float32)))
        cs2 = np.vstack((zero_f,np.cumsum(x2,axis=0,dtype=np.float32)))
        cc = np.vstack((zero_i,np.cumsum(valid.astype(np.int32),axis=0,dtype=np.int32)))
        end = np.arange(x.shape[0],dtype=np.int32)
        start = np.maximum(end-window,0)
        cnt = cc[end]-cc[start]
        s1 = cs[end]-cs[start]
        s2 = cs2[end]-cs2[start]
        mean = np.divide(s1,cnt,out=np.zeros_like(s1),where=cnt>0)
        var = np.divide(s2,cnt,out=np.zeros_like(s2),where=cnt>0)-mean*mean
        z = (x-mean)/np.sqrt(np.maximum(var,np.float32(1e-8)))
        z[(cnt<min_periods)|(~valid)] = 0.0
        return np.nan_to_num(z,copy=False,nan=0.0,posinf=0.0,neginf=0.0).astype(np.float32,copy=False)

    @staticmethod
    def _h5_slice(ds,day_start:int,day_end:int,stock_idx=None,minute_idx=None)->np.ndarray:
        if minute_idx is None:
            data = ds[day_start:day_end]
        else:
            data = ds[day_start:day_end,minute_idx,:]
        if stock_idx is not None:
            data = data[...,stock_idx]
        return data.astype(np.float32)

    def build_minute_derived_fields(self):
        hdf5_path = "/home/workspace/common/hdf5/all_minute_data.h5"
        dst = os.path.join(self.base_dir,"minute_derived")
        os.makedirs(dst,exist_ok=True)

        all_dates = pd.read_pickle(os.path.join(self.source_dir,"all_dates.pkl"))
        all_stocks = pd.read_pickle(os.path.join(self.source_dir,"all_stocks.pkl"))
        cutoff = pd.Timestamp("20150101")
        eps = np.float32(1e-8)
        chunk_days = 8

        with h5py.File(hdf5_path,"r") as f:
            h5_dates = pd.to_datetime(f["axis/dates"][:].astype(str))
            h5_stocks = pd.Index(f["axis/stocks"][:].astype(str))
            nd = len(h5_dates)
            ns = len(h5_stocks)
            daily = {
                "path_high_time":np.empty((nd,ns),dtype=np.float32),
                "path_low_time":np.empty((nd,ns),dtype=np.float32),
                "path_high_to_close":np.empty((nd,ns),dtype=np.float32),
                "path_low_rebound":np.empty((nd,ns),dtype=np.float32),
                "path_efficiency":np.empty((nd,ns),dtype=np.float32),
                "gate_liq_high_amount":np.empty((nd,ns),dtype=np.float32),
                "gate_rev_high_rv":np.empty((nd,ns),dtype=np.float32),
                "gate_rev_pm_disagree":np.empty((nd,ns),dtype=np.float32),
                "gate_rev_high_oimb":np.empty((nd,ns),dtype=np.float32),
            }
            anomaly_base = {
                "amihud":np.empty((nd,ns),dtype=np.float32),
                "tail_vol_share":np.empty((nd,ns),dtype=np.float32),
                "high_to_close":np.empty((nd,ns),dtype=np.float32),
                "order_imbalance":np.empty((nd,ns),dtype=np.float32),
                "path_efficiency":np.empty((nd,ns),dtype=np.float32),
            }

            for s in range(0,nd,chunk_days):
                e = min(s+chunk_days,nd)
                opens0 = UpdateService._h5_slice(f["data/opens"],s,e,minute_idx=0)
                highs = UpdateService._h5_slice(f["data/highs"],s,e)
                lows = UpdateService._h5_slice(f["data/lows"],s,e)
                closes = UpdateService._h5_slice(f["data/closes"],s,e)
                returns = UpdateService._h5_slice(f["data/returns"],s,e)
                volumes = UpdateService._h5_slice(f["data/volumes"],s,e)
                amounts = UpdateService._h5_slice(f["data/amounts"],s,e)

                m = closes.shape[1]
                close_last = closes[:,-1,:]
                mid = closes[:,m//2,:]
                high_fill = np.where(np.isfinite(highs),highs,-np.inf)
                low_fill = np.where(np.isfinite(lows),lows,np.inf)
                valid_high = np.any(np.isfinite(highs),axis=1)
                valid_low = np.any(np.isfinite(lows),axis=1)
                high_time = np.argmax(high_fill,axis=1).astype(np.float32)/np.float32(max(m-1,1))
                low_time = np.argmin(low_fill,axis=1).astype(np.float32)/np.float32(max(m-1,1))
                high_time[~valid_high] = np.nan
                low_time[~valid_low] = np.nan

                high_max = np.nanmax(highs,axis=1)
                low_min = np.nanmin(lows,axis=1)
                path_len = np.nansum(np.abs(np.diff(closes,axis=1)),axis=1)
                intraday_ret = close_last/(opens0+eps)-1.0
                am_pm_ret_diff = mid/(opens0+eps)-1.0-(close_last/(mid+eps)-1.0)
                realized_vol = np.sqrt(np.nansum(returns*returns,axis=1))
                amount_sum = np.nansum(amounts,axis=1)
                volume_sum = np.nansum(volumes,axis=1)
                daily_amihud = np.nanmean(np.abs(returns)/(amounts+eps),axis=1)
                daily_oimb = np.nansum(np.sign(returns)*volumes,axis=1)/(volume_sum+eps)
                high_to_close = (high_max-close_last)/(np.abs(high_max-opens0)+eps)
                path_efficiency = np.abs(close_last-opens0)/(path_len+eps)

                high_amount_gate = amount_sum>=np.nanpercentile(amount_sum,70,axis=1)[:,None]
                high_rv_gate = realized_vol>=np.nanpercentile(realized_vol,70,axis=1)[:,None]
                pm_disagree_gate = (am_pm_ret_diff*intraday_ret)<0.0
                high_oimb_gate = np.abs(daily_oimb)>=np.nanpercentile(np.abs(daily_oimb),70,axis=1)[:,None]

                daily["path_high_time"][s:e] = high_time
                daily["path_low_time"][s:e] = low_time
                daily["path_high_to_close"][s:e] = high_to_close
                daily["path_low_rebound"][s:e] = (close_last-low_min)/(np.abs(opens0-low_min)+eps)
                daily["path_efficiency"][s:e] = path_efficiency
                daily["gate_liq_high_amount"][s:e] = np.where(high_amount_gate,-daily_amihud,0.0)
                daily["gate_rev_high_rv"][s:e] = np.where(high_rv_gate,-intraday_ret,0.0)
                daily["gate_rev_pm_disagree"][s:e] = np.where(pm_disagree_gate,-intraday_ret,0.0)
                daily["gate_rev_high_oimb"][s:e] = np.where(high_oimb_gate,-intraday_ret,0.0)

                anomaly_base["amihud"][s:e] = daily_amihud
                anomaly_base["tail_vol_share"][s:e] = np.nansum(volumes[:,-30:,:],axis=1)/(volume_sum+eps)
                anomaly_base["high_to_close"][s:e] = high_to_close
                anomaly_base["order_imbalance"][s:e] = daily_oimb
                anomaly_base["path_efficiency"][s:e] = path_efficiency

            daily["anom_amihud_z20"] = UpdateService._rolling_zscore_past(anomaly_base["amihud"],20)
            daily["anom_tail_vol_share_z20"] = UpdateService._rolling_zscore_past(anomaly_base["tail_vol_share"],20)
            daily["anom_high_to_close_z20"] = UpdateService._rolling_zscore_past(anomaly_base["high_to_close"],20)
            daily["anom_order_imbalance_z20"] = UpdateService._rolling_zscore_past(anomaly_base["order_imbalance"],20)
            daily["anom_path_efficiency_z20"] = UpdateService._rolling_zscore_past(anomaly_base["path_efficiency"],20)

        # with h5py.File(hdf5_path,"a") as fh:
        #     grp = fh.require_group("daily")
        #     for n,arr in daily.items():
        #         if n in grp:
        #             del grp[n]
        #         grp.create_dataset(
        #             n,
        #             data=arr.astype(np.float32,copy=False),
        #             chunks=(min(256,arr.shape[0]),arr.shape[1]),
        #             compression="gzip",
        #             compression_opts=4,
        #         )
        #     print(f"saved h5 daily/*: {len(daily)} fields -> {hdf5_path}")

        for n,data in daily.items():
            df = pd.DataFrame(data,index=h5_dates,columns=h5_stocks,dtype=np.float32)
            df = df.reindex(index=all_dates,columns=all_stocks).loc[cutoff:]
            out_fp = os.path.join(dst,f"{n}.pkl")
            SafePickleWriter.safe_to_pickle(df,out_fp)
            print(f"saved: {out_fp}  shape={df.shape}")

    def build_derived_index(self):
        # 衍生指数
        dst = os.path.join(self.base_dir,"index_data")
        index_data = pd.read_pickle(os.path.join(dst,"index_data.pkl"))
        r1 = index_data["zz1000s_returns"]
        r2 = index_data["hs300s_returns"]
        r = r2-r1
        out_fp = os.path.join(dst,"size_rets.pkl")
        print(f"saved: {out_fp}")
        SafePickleWriter.safe_to_pickle(r,out_fp)

    def update(self):
        self.move_and_fix_datas()
        self.build_hfq_datas()
        self.build_standards()
        self.build_universe_A1800_zzhls()
        self.build_universe_liq_div3y_A1800()
        self.build_smalls()
        self.fix_long_window_factors()
        self.build_derived_styles()
        self.build_vwaps()
        self.build_derived_pvs()
        self.build_minute_derived_fields()
        self.build_derived_index()
        self.fu.update()
        self.fn.update()

if __name__=="__main__":
    up = UpdateService(source_dir="/home/workspace/common/product_pkl/data",
                       base_dir="/home/workspace/common/quant_data",
                       fu=FundamentalUpdater("/home/workspace/common/product_pkl/data","/home/workspace/common/product_pkl/fundamentals","/home/workspace/common/quant_data"),
                       fn=FundamentalNeutralizer("/home/workspace/common/quant_data"))
    up.build_derived_pvs()