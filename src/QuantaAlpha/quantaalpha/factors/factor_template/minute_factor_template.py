import pandas as pd
import numpy as np
from vendors.quant_lib.analysis import *
from vendors.quant_lib.minute_tools import MinuteFactorEngine
import vendors.quant_lib.minute_ops
from quant_union.common.quantEnum import DomainType, CategoryType

# Minute factor template.
#
# minute factor:
# - uses raw data from DataProvider (daily level)
# - uses minute-level data via MinuteFactorEngine
# - prepare_minute_datas returns minute_ctx dict
# - calc_factor receives data_ctx and minute_ctx
TYPE = "regular"

META = {
    # Unique factor name in the library.
    "factor_name":"your_minute_factor_name",
    # Required author name.
    "author":"your_name",
    # Required factor level. Must be "minutes" for minute factors.
    "level":"minutes",
    # Required factor domain. Allowed values: "pv", "fundamental", "hybrid".
    "domain":DomainType.pv,
    # Optional tag string.
    "tag":"",
    # Factor category. Default "unknown".
    "category":"unknown",
}

SETTING = {
    # Raw daily data fields loaded from DataProvider.
    "data_needed":[],

    # Universe field name loaded from DataProvider.
    "universe":"standards",

    # If True, inputs will be masked by universe before calc_factor runs.
    "pasteurization":False,
    # Post-processing: ts_decay_linear window. 0 means no decay.
    "decay":0,
    # Post-processing: neutralize mode. None or one of "industry","size","ram","styles","complete".
    "neutralize":None,
}


def prepare_minute_datas()->dict[str,pd.DataFrame]:
    """
    Prepare minute-level data using MinuteFactorEngine.

    Returns a dict of DataFrames (index=trading dates, columns=stock codes).
    Each entry will be passed to calc_factor as minute_ctx.
    """
    mfe = MinuteFactorEngine()
    # example:
    # result = mfe.run(inputs=["closes","volumes"], operator="your_operator", endminute="1457")
    # for joint pv+minutes factors, keep raw minute processing here, then combine the
    # reduced daily 2D frames with daily pv fields such as data_ctx["hfq_closes"] in calc_factor().
    # note the field routing:
    # - actual quant_data fields use plural `vwaps` and `turnovers`
    # - in joint runs, shared fields route by context: minute subtrees use minute data,
    #   while declared top-level daily anchors stay on data_ctx / SETTING["data_needed"]
    # valid example:
    # mean_vwap_30m = mfe.run(inputs="vwaps", operator="mean", startminute="1428", endminute="1457")
    # mean_mid_30m = mfe.run(inputs=["highs", "lows"], operator="add", aggregator="mean", startminute="1428", endminute="1457") / 2.0
    # return {"mean_vwap_30m": mean_vwap_30m, "mean_mid_30m": mean_mid_30m}
    # return {"result": result}
    return {}


def calc_factor(data_ctx:dict,minute_ctx:dict)->pd.DataFrame:
    """
    Build and return a minute factor value DataFrame.

    Contract:
    - input:
      data_ctx is dict[str,pd.DataFrame] (daily data)
      minute_ctx is dict[str,pd.DataFrame] (from prepare_minute_datas)
    - output: pd.DataFrame
    - index: trading dates
    - columns: stock codes
    """
    pass
