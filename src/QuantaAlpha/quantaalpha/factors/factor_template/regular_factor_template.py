import pandas as pd
import numpy as np
from vendors.quant_lib.analysis import *
from quant_union.common.quantEnum import DomainType, CategoryType

# Regular factor template.
#
# regular factor:
# - uses only raw data from DataProvider
# - calc_factor receives one ctx named data_ctx
TYPE = "regular"

META = {
    # Unique factor name in the library.
    "factor_name":"your_factor_name",
    # Required author name.
    "author":"your_name",
    # Required factor level. Allowed values: "days", "minutes".
    "level":"days",
    # Required factor domain. Allowed values: "pv", "fundamental", "hybrid".
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
        "hfq_closes",
    ],

    # Universe field name loaded from DataProvider.
    "universe":"standards",

    # If True, inputs in data_ctx will be masked by universe before calc_factor runs.
    "pasteurization":False,
    # Post-processing: ts_decay_linear window. 0 means no decay.
    "decay":0,
    # Post-processing: neutralize mode. None or one of "industry","size","ram","styles","complete".
    "neutralize":None,
}


def calc_factor(data_ctx:dict)->pd.DataFrame:
    """
    Build and return a regular factor value DataFrame.

    Contract:
    - input: data_ctx is a dict[str,pd.DataFrame]
    - output: pd.DataFrame
    - index: trading dates
    - columns: stock codes

    Access pattern:
    - raw data: data_ctx["hfq_closes"], data_ctx["volumes"]
    """

    hfq_closes = data_ctx["hfq_closes"]
    return -hfq_closes
