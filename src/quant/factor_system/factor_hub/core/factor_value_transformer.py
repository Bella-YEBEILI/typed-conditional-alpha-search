import pandas as pd
from typing import Literal

from quant.quant_lib.analysis import group_neutralize,cs_multireg

STYLE_FIELDS = ["Beta","Liq","Mom","Nlsize","Rev","Size","Vol"]
SIZE_FIELDS = ["Size","Nlsize"]
RAM_FIELDS = ["Mom","Rev"]

class FactorValueTransformer:
    def __init__(self,dm):
        self.dm = dm
        self.data:dict[str,pd.DataFrame] = {}

    def _get_data(self,fld):
        data = self.data.get(fld)
        if data is None:
            data = self.dm.get_data(fld)
            self.data[fld] = data
        return data

    def _get_neutralized_style(self,fld):
        key = f"{fld}_ind_neutral"
        data = self.data.get(key)
        if data is None:
            x = self._get_data(fld)
            industrys = self._get_data("industrys")
            data = group_neutralize(x,industrys)
            self.data[key] = data
        return data

    def subuniverse_transform(self,factor_value:pd.DataFrame,
                              fld:Literal["hs300s","zz1000s"])->pd.DataFrame:
        idx = factor_value.index
        mask = self._get_data(fld).reindex(index=idx).fillna(0).astype(bool)
        return factor_value.where(mask)

    def neutralize_transform(self,factor_value:pd.DataFrame,
                             mode:Literal["industry","size","styles","ram","complete"])->pd.DataFrame:
        idx = factor_value.index
        if mode=="industry":
            industrys = self._get_data("industrys").reindex(index=idx)
            return group_neutralize(factor_value,industrys)

        if mode=="size":
            fields = SIZE_FIELDS
        elif mode=="ram":
            fields = RAM_FIELDS
        else:
            fields = STYLE_FIELDS

        if mode in ("size","ram","styles"):
            x_list = [self._get_data(fld).reindex(index=idx) for fld in fields]
            return cs_multireg(x_list,factor_value)

        # complete: FWL等价于行业哑变量+风格因子联合回归
        y = group_neutralize(factor_value,self._get_data("industrys").reindex(index=idx))
        x_list = [self._get_neutralized_style(fld).reindex(index=idx) for fld in fields]
        return cs_multireg(x_list,y)
