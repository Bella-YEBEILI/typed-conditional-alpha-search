from typing import Any

import pandas as pd
import numpy as np

from quant.quant_lib.numbafunc import precise_corr_2d

CORR_METRICS = ["PEARSON","JACCARD"]

"""
相关性度量：
PEARSON: 两个数值型Series的线性相关系数
JACCARD: 两个布尔型Series为True的重叠度
"""

class FactorComparator:
    def __init__(self):
        pass

    def check_prod_corr(self,
                        lib_data:dict[str,pd.Series],
                        new_data:pd.Series,
                        corr_metric:str="PEARSON"
                        )->pd.Series:
        if corr_metric not in CORR_METRICS:
            raise ValueError("unsupported corr metric")
        if not isinstance(lib_data,dict):
            raise ValueError("lib_data must be dict")
        for v in lib_data.values():
            if not isinstance(v,pd.Series):
                    raise ValueError("lib_data value must be Series")
        if not isinstance(new_data,pd.Series):
            raise ValueError("new data must be Series")
        
        if corr_metric=="PEARSON":
            df = pd.DataFrame(lib_data)
            aligned_df,aligned_series = df.align(new_data,join="inner",axis=0)
            corr = aligned_df.corrwith(aligned_series,axis=0)

            return corr.sort_values(ascending=False)
        
        elif corr_metric=="JACCARD":            
            df = pd.DataFrame(lib_data)
            aligned_df,aligned_series = df.align(new_data,join="inner",axis=0)

            aligned_df = aligned_df.astype("boolean")
            aligned_series = aligned_series.astype("boolean")

            out = {}
            for col in aligned_df.columns:
                a = aligned_df[col]
                b = aligned_series
                valid = a.notna()&b.notna()
                if valid.sum()==0:
                    out[col] = np.nan
                    continue

                a = a[valid].to_numpy(dtype=bool)
                b = b[valid].to_numpy(dtype=bool)

                union = np.sum(a|b)
                if union==0:
                    out[col] = np.nan
                else:
                    out[col] = np.sum(a&b)/union

            return pd.Series(out).sort_values(ascending=False)

    def check_self_corr(self,
                        data:dict[str,Any],
                        corr_metric:str="PEARSON")->tuple[pd.DataFrame,pd.DataFrame]:
        if corr_metric not in CORR_METRICS:
            raise ValueError("unsupported corr metric")
        if not isinstance(data,dict):
            raise ValueError("data must be dict")
        for v in data.values():
            if not isinstance(v,pd.Series):
                raise ValueError("lib_data value must be Series")
        
        if corr_metric=="PEARSON":        
            df = pd.DataFrame(data)

            if df.shape[1]==0:
                empty_df = pd.DataFrame()
                stats_df = pd.DataFrame(columns=["max_self_corr","max_self_corr_factor","avg_self_corr"])
                stats_df.index.name = "id"
                return empty_df,stats_df

            values = precise_corr_2d(df.to_numpy(dtype=float,copy=False))
            corr_df = pd.DataFrame(values,index=df.columns,columns=df.columns)

            rows = []
            for factor_name in corr_df.index:
                corr_s = corr_df.loc[factor_name].drop(labels=factor_name,errors="ignore").dropna().sort_values(ascending=False)
                if len(corr_s)>0:
                    rows.append({
                        "id":factor_name,
                        "max_self_corr":corr_s.iloc[0],
                        "max_self_corr_factor":corr_s.index[0],
                        "avg_self_corr":corr_s.mean(),
                    })
                else:
                    rows.append({
                        "id":factor_name,
                        "max_self_corr":pd.NA,
                        "max_self_corr_factor":pd.NA,
                        "avg_self_corr":pd.NA
                    })
            stats_df = pd.DataFrame(rows).set_index("id")
            return corr_df,stats_df
        
        elif corr_metric=="JACCARD":
            df = pd.DataFrame(data)

            if df.shape[1]==0:
                empty_df = pd.DataFrame()
                stats_df = pd.DataFrame(columns=["max_self_corr","max_self_corr_factor","avg_self_corr"])
                stats_df.index.name = "id"
                return empty_df,stats_df

            state = df.astype("boolean")
            valid = state.notna().to_numpy(dtype=np.int64)
            x = state.fillna(False).to_numpy(dtype=np.int64)

            joint = x.T@x
            true_cnt = x.T@valid
            union = true_cnt+true_cnt.T-joint
            valid_cnt = valid.T@valid

            values = np.full(union.shape,np.nan,dtype=float)
            mask = (union>0)&(valid_cnt>0)
            values[mask] = joint[mask]/union[mask]
            np.fill_diagonal(values,1.0)

            corr_df = pd.DataFrame(values,index=df.columns,columns=df.columns)

            rows = []
            for factor_name in corr_df.index:
                corr_s = corr_df.loc[factor_name].drop(labels=factor_name,errors="ignore").dropna().sort_values(ascending=False)
                if len(corr_s)>0:
                    rows.append({
                        "id":factor_name,
                        "max_self_corr":corr_s.iloc[0],
                        "max_self_corr_factor":corr_s.index[0],
                        "avg_self_corr":corr_s.mean(),
                    })
                else:
                    rows.append({
                        "id":factor_name,
                        "max_self_corr":pd.NA,
                        "max_self_corr_factor":pd.NA,
                        "avg_self_corr":pd.NA
                    })

            stats_df = pd.DataFrame(rows).set_index("id")
            return corr_df,stats_df
