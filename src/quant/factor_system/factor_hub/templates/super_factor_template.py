import pandas as pd
import numpy as np
from quant.quant_lib.analysis import *
from quant.quant_lib.quantEnum import DomainType,CategoryType

# Super factor template.

# super factor:
# - can use raw data from DataProvider
# - can use factor values from factor_base/factor_values
# - calc_factor receives two ctx dicts:
#   1. data_ctx for raw data
#   2. factor_ctx for dependent factor values

TYPE = "super"

META = {
    # Unique factor name in the library.
    "factor_name":"your_super_factor_name",
    # Required author name.
    "author":"your_name",
    # Required factor level. Allowed values: "days", "minutes".
    "level":"days",
    # Required factor domain. Allowed values: "pv", "fundamental"
    "domain":DomainType.pv,
    # Optional tag string.
    "tag":"",
    # Factor category. Default "unknown".
    "category":"unknown",
}

SETTING = {
    # Raw data fields loaded from DataProvider.
    # Each field will be injected into data_ctx with the same key.
    "data_needed":[
        "closes",
    ],

    # Factor names already stored in the factor library.
    # Each factor value will be injected into factor_ctx with the same key.
    "factor_needed":[
        "alpha_001",
        "alpha_002",
    ],

    # Universe field name loaded from DataProvider.
    "universe":"stables",

    # If True, both data_ctx and factor_ctx inputs will be masked by universe
    # before calc_factor runs.
    "pasteurization":False,
    # Post-processing: ts_decay_linear window. 0 means no decay.
    "decay":0,
    # Post-processing: neutralize mode. None or one of "industry","size","ram","styles","complete".
    "neutralize":None,
}


def calc_factor(data_ctx:dict,factor_ctx:dict)->pd.DataFrame:
    """
    Build and return a super factor value DataFrame.

    Contract:
    - input:
      data_ctx is dict[str,pd.DataFrame]
      factor_ctx is dict[str,pd.DataFrame]
    - output: pd.DataFrame
    - index: trading dates
    - columns: stock codes

    Access pattern:
    - raw data: data_ctx["closes"], data_ctx["volumes"]
    - factor value: factor_ctx["alpha_001"], factor_ctx["alpha_002"]
    """

    closes = data_ctx["closes"]
    alpha_001 = factor_ctx["alpha_001"]
    alpha_002 = factor_ctx["alpha_002"]
    return closes.rank(axis=1,pct=True)+(alpha_001+alpha_002)/2
