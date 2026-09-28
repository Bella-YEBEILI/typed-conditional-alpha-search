import os
import sys

import numpy as np
import pandas as pd
from .safe_pickle_writer import SafePickleWriter
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from .fundamental_updater import FACTOR_NAMES

MIN_STOCKS = 50
MAD_MULT = 5.0


class FundamentalNeutralizer:
    def __init__(self,
                 base_dir:str):
        self.base_dir = base_dir
        self.src = os.path.join(base_dir,"fundamental_factors","raw")
        self.dst = os.path.join(base_dir,"fundamental_factors","neutral")

    def _load_pkl(self,path:str)->pd.DataFrame:
        if not os.path.isfile(path):
            print(f"ERROR: file not found: {path}",file=sys.stderr)
            sys.exit(1)
        df = pd.read_pickle(path)
        if not isinstance(df.index,pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index.astype(str))
        return df.sort_index()

    @staticmethod
    def _mad_winsorize(s:pd.Series,mult:float=MAD_MULT)->pd.Series:
        med = s.median()
        mad = (s-med).abs().median()
        if mad < 1e-12:
            return s
        lower = med-mult*1.4826*mad
        upper = med+mult*1.4826*mad
        return s.clip(lower,upper)

    @staticmethod
    def _zscore(s:pd.Series)->pd.Series:
        std = s.std()
        if std < 1e-12:
            return s*0.0
        return (s-s.mean())/std

    def _neutralize_one_day(self,
                            reg:LinearRegression,
                            y_raw:pd.Series,
                            ln_cap:pd.Series,
                            ind:pd.Series):
        # 对齐三者都有值的股票 + 去掉空行业
        valid = y_raw.index.intersection(ln_cap.dropna().index).intersection(ind.dropna().index)
        ind_valid = ind[valid]
        valid = ind_valid[ind_valid.str.len() > 0].index
        if len(valid) < MIN_STOCKS:
            return None,valid

        # MAD 去极值 + Z-score
        y = self._zscore(self._mad_winsorize(y_raw[valid]))

        # 自变量: ln_mktcap + 行业哑变量 (不含截距)
        ln_cap_valid = ln_cap[valid]
        industry_dummies = pd.get_dummies(ind[valid],dtype=np.float64)
        X = pd.concat([ln_cap_valid.rename("ln_mktcap"),industry_dummies],axis=1)

        reg.fit(X.values,y.values)
        pred = reg.predict(X.values)
        resid = y.values-pred
        resid_std = resid.std()
        if resid_std > 1e-12:
            resid = (resid-resid.mean())/resid_std
        return resid.astype(np.float32),valid

    def _neutralize_factor(self,
                           factor_df:pd.DataFrame,
                           ln_mktcap_df:pd.DataFrame,
                           industry_df:pd.DataFrame,
                           dates:pd.DatetimeIndex|None=None)->pd.DataFrame:
        if dates is None:
            dates = factor_df.index
        residual_df = pd.DataFrame(np.nan,index=dates,columns=factor_df.columns,dtype=np.float32)
        reg = LinearRegression(fit_intercept=False)

        for date in dates:
            if date not in factor_df.index or date not in ln_mktcap_df.index or date not in industry_df.index:
                continue
            y_raw = factor_df.loc[date].dropna()
            ln_cap = ln_mktcap_df.loc[date]
            ind = industry_df.loc[date]
            resid,valid = self._neutralize_one_day(reg,y_raw,ln_cap,ind)
            if resid is not None:
                residual_df.loc[date,valid] = resid
        return residual_df

    def _get_output_path(self,name:str)->str:
        return os.path.join(self.dst,f"{name}_neutral.pkl")

    def _load_existing_output(self,name:str)->pd.DataFrame|None:
        fp = self._get_output_path(name)
        return pd.read_pickle(fp) if os.path.isfile(fp) else None

    def _can_incremental(self)->bool:
        return all(os.path.isfile(self._get_output_path(n)) for n in FACTOR_NAMES)

    def _get_saved_end_date(self)->pd.Timestamp|None:
        # 取所有因子 neutral pkl 末日的最小值, 保证全部追齐
        ends = []
        for n in FACTOR_NAMES:
            df = self._load_existing_output(n)
            if df is None or len(df.index)==0:
                return None
            ends.append(pd.Timestamp(pd.to_datetime(df.index).max()))
        return min(ends)

    def _merge_saved_output(self,
                            old:pd.DataFrame|None,
                            new:pd.DataFrame,
                            merge_start:pd.Timestamp)->pd.DataFrame:
        if old is None or len(old)==0:
            return new.sort_index()
        merged = pd.concat([
            old.loc[old.index < merge_start],
            new.loc[new.index >= merge_start],
        ]).sort_index()
        merged = merged.loc[~merged.index.duplicated(keep="last")]
        return merged.reindex(columns=new.columns)

    def update(self):
        os.makedirs(self.dst,exist_ok=True)

        mcap = self._load_pkl(os.path.join(self.base_dir,"pv","total_market_caps.pkl"))
        ln_mktcap = np.log(mcap.replace(0,np.nan))
        industry = self._load_pkl(os.path.join(self.base_dir,"industry","industrys.pkl"))
        industry = industry.where(industry.notna() & (industry != ""))
        print(f"ln_mktcap shape: {ln_mktcap.shape}")
        print(f"industry  shape: {industry.shape}")

        use_incremental = self._can_incremental()
        saved_end = self._get_saved_end_date() if use_incremental else None
        if saved_end is not None:
            print(f"Incremental neutralize from {saved_end:%Y-%m-%d}")
        else:
            print("Full neutralize")

        for name in tqdm(FACTOR_NAMES,desc="neutralize_fundamental"):
            raw_fp = os.path.join(self.src,f"{name}.pkl")
            if not os.path.isfile(raw_fp):
                print(f"SKIP {name}: {raw_fp} not found")
                continue
            factor_df = self._load_pkl(raw_fp)

            if use_incremental and saved_end is not None:
                new_dates = factor_df.index[factor_df.index > saved_end]
                if len(new_dates)==0:
                    print(f"{name}: already up to date at {saved_end:%Y-%m-%d}")
                    continue
                merge_start = new_dates[0]
                tail = self._neutralize_factor(factor_df,ln_mktcap,industry,dates=new_dates)
                old = self._load_existing_output(name)
                if old is not None:
                    old = old.reindex(columns=factor_df.columns)
                out_df = self._merge_saved_output(old,tail,merge_start).astype(np.float32)
            else:
                out_df = self._neutralize_factor(factor_df,ln_mktcap,industry).astype(np.float32)

            out_fp = self._get_output_path(name)
            SafePickleWriter.safe_to_pickle(out_df,out_fp)
            print(f"Save {name}: {out_fp}  shape={out_df.shape}")

        print("Done.")
