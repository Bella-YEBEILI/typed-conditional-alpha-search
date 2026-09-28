import pandas as pd
import os

from quant.quant_lib.analysis import ts_decay_linear,group_neutralize,cs_multireg

from .factor_loader import load_factor_module

from .config import FACTOR_START_DATE,FACTOR_END_DATE

class FactorValueEngine:
    def __init__(self,dm,base_dir:str,registry):
        self.dm = dm
        self.registry = registry
        self.value_dir = os.path.join(base_dir,"factor_values")
        os.makedirs(self.value_dir,exist_ok=True)
        self.data:dict[str,pd.DataFrame] = {}

    def _get_data(self,fld:str)->pd.DataFrame:
        data = self.data.get(fld)
        if data is None:
            data = self.dm.get_data(fld)
            self.data[fld] = data
        return data

    def _get_exists_factor_value(self,factor_name:str)->pd.DataFrame:
        if not self.registry.has_factor(factor_name):
            raise ValueError(f"factor not found in registry: {factor_name}")
        path = os.path.join(self.value_dir,f"{factor_name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"factor value not found: {path}")
        data = pd.read_pickle(path)
        if not isinstance(data,pd.DataFrame):
            raise ValueError("stored factor value must be DataFrame")
        return data

    def calc_value_by_name(self,
                           factor_name:str,
                           start:str|None=None,
                           end:str|None=None)->pd.DataFrame:
        item = self.registry.get_factor(factor_name)
        if item is None:
            raise ValueError(f"factor not found: {factor_name}")
        return self.calc_value_by_path(item["file_path"],start,end)

    def calc_value_by_path(self,
                           file_path:str,
                           start:str|None=None,
                           end:str|None=None)->pd.DataFrame:
        if start is None:
            start = FACTOR_START_DATE
        if end is None:
            end = FACTOR_END_DATE

        m = load_factor_module(file_path)

        factor_type = m.TYPE
        level = m.META["level"]
        data_needed = m.SETTING["data_needed"]
        universe_name = m.SETTING["universe"]
        pasteurization = m.SETTING["pasteurization"]
        decay = m.SETTING.get("decay",0)
        neutralize = m.SETTING.get("neutralize",None)

        univ = self._get_data(universe_name).astype(bool)
        cindex = univ.index

        data_ctx = {}
        minute_ctx = {}
        factor_ctx = {}

        for fld in data_needed:
            data = self._get_data(fld)
            cindex = cindex.intersection(data.index)
            data_ctx[fld] = data

        if level=="minutes":
            import quant.quant_lib.minute_ops
            minute_ctx = m.prepare_minute_datas()
            if not isinstance(minute_ctx,dict):
                raise ValueError("prepare_minute_datas must return dict")
            for df in minute_ctx.values():
                cindex = cindex.intersection(df.index)

        if factor_type=="super":
            factor_ctx = {}
            factor_needed = m.SETTING["factor_needed"]
            for factor_name in factor_needed:
                factor = self._get_exists_factor_value(factor_name)
                cindex = cindex.intersection(factor.index)
                factor_ctx[factor_name] = factor
        
        univ = univ.reindex(index=cindex)
        if pasteurization:
            for k,v in data_ctx.items():
                data_ctx[k] = v.reindex(index=cindex).where(univ)
            for k,v in minute_ctx.items():
                minute_ctx[k] = v.reindex(index=cindex).where(univ)
            for k,v in factor_ctx.items():
                factor_ctx[k] = v.reindex(index=cindex).where(univ)
        else:
            for k,v in data_ctx.items():
                data_ctx[k] = v.reindex(index=cindex)
            for k,v in minute_ctx.items():
                minute_ctx[k] = v.reindex(index=cindex)
            for k,v in factor_ctx.items():
                factor_ctx[k] = v.reindex(index=cindex)   

        if factor_type=="regular":
            if level=="minutes":
                out = m.calc_factor(data_ctx=data_ctx,minute_ctx=minute_ctx)
            else:
                out = m.calc_factor(data_ctx=data_ctx)
        
        elif factor_type=="super":
            if level=="minutes":
                out = m.calc_factor(data_ctx=data_ctx,factor_ctx=factor_ctx,minute_ctx=minute_ctx)
            else:
                out = m.calc_factor(data_ctx=data_ctx,factor_ctx=factor_ctx)

        if not isinstance(out,pd.DataFrame):
            raise ValueError("calc_factor result must be DataFrame")
    
        # universe
        out = out.where(univ)

        # decay
        if decay>0:
            out = ts_decay_linear(out,decay)

        # neutralize
        if neutralize=="industry":
            industrys = self._get_data("industrys").reindex(index=out.index)
            out = group_neutralize(out,industrys)
        elif neutralize=="size":
            size = self._get_data("Size").reindex(index=out.index)
            nlsize = self._get_data("Nlsize").reindex(index=out.index)
            out = cs_multireg(X_list=[size,nlsize],y_df=out)
        elif neutralize=="ram":
            rev = self._get_data("Rev").reindex(index=out.index)
            mom = self._get_data("Mom").reindex(index=out.index)
            out = cs_multireg(X_list=[rev,mom],y_df=out)
        elif neutralize=="styles":
            x_list = [self._get_data(fld).reindex(index=out.index) for fld in ["Beta","Liq","Mom","Nlsize","Rev","Size","Vol"]]
            out = cs_multireg(x_list,out)
        elif neutralize=="complete":
            industrys = self._get_data("industrys").reindex(index=out.index)
            out = group_neutralize(out,industrys)
            x_list = []
            for fld in ["Beta","Liq","Mom","Nlsize","Rev","Size","Vol"]:
                x = self._get_data(fld).reindex(index=out.index)
                x = group_neutralize(x,industrys)
                x_list.append(x)
            out = cs_multireg(x_list,out)

        out = out.loc[start:end]
        return out
