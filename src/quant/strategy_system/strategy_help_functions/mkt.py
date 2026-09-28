import pandas as pd
from quant.data_system.data_hub.main import DataManager
from quant.quant_lib.analysis import *


def mkt_adx(high:pd.Series,low:pd.Series,close:pd.Series,n:int=14)->pd.Series:
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up>dn)&(up>0),0.0)
    minus_dm = dn.where((dn>up)&(dn>0),0.0)
    tr = pd.concat([
        high-low,
        (high-close.shift(1)).abs(),
        (low-close.shift(1)).abs()
    ],axis=1).max(axis=1)
    alpha = 1.0/n
    atr = tr.ewm(alpha=alpha,adjust=False).mean()
    plus_di = 100.0*(plus_dm.ewm(alpha=alpha,adjust=False).mean()/atr)
    minus_di = 100.0*(minus_dm.ewm(alpha=alpha,adjust=False).mean()/atr)
    dx = 100.0*((plus_di-minus_di).abs()/(plus_di+minus_di))
    adx = dx.ewm(alpha=alpha,adjust=False).mean()
    return adx

def load_concentration()->pd.Series:
    amt = DataManager().get_data("amounts")
    return (amt.where(cs_rank(amt)>0.95)).sum(axis=1)/amt.sum(axis=1)

def load_mkt():
    o = DataManager().get_data("hfq_opens")
    h = DataManager().get_data("highs")
    l = DataManager().get_data("lows")
    c = DataManager().get_data("closes")
    limup = DataManager().get_data("limit_up_ctc").astype(int)
    limdown = DataManager().get_data("limit_down_ctc").astype(int)
    tradables = DataManager().get_data("tradables").notna().astype(int)
    index_ret = DataManager().get_data("index_data")
    hs300s = index_ret["hs300s_returns"]
    zz500s = index_ret["zz500s_returns"]
    zz1000s = index_ret["zz1000s_returns"]
    cnt = tradables.sum(axis=1)
    amt = DataManager().get_data("amounts")
    caps = DataManager().get_data("float_market_caps")
    shares = caps.shift(1)/c.shift(1)
    shares.iloc[0] = caps.iloc[0]/c.iloc[0]
    vo = (o*shares).sum(axis=1)
    vh = (h*shares).sum(axis=1)
    vl = (l*shares).sum(axis=1)
    vc = (c*shares).sum(axis=1)
    vo = vo/vc.iloc[0]
    vh = vh/vc.iloc[0]
    vl = vl/vc.iloc[0]
    vc = vc/vc.iloc[0]
    up_pct = (c>c.shift(1)).sum(axis=1)/cnt
    limup_pct = limup.sum(axis=1)/cnt
    limdown_pct = limdown.sum(axis=1)/cnt
    ret1 = vc/vc.shift(1)-1
    ret5 = vc/vc.shift(5)-1
    ret20 = vc/vc.shift(20)-1
    d1 = hs300s-zz500s
    d2 = zz500s-zz1000s
    vol20 = ret1.rolling(20).std()
    risk20 = (ret1<-0.01).rolling(20).mean()
    liq = amt.where(tradables.astype(bool)).sum(axis=1)/caps.where(tradables.astype(bool)).sum(axis=1)
    adx = mkt_adx(vh,vl,vc)
    mkt_dict:dict[str,pd.Series] = {
        "up_pct":up_pct,
        "limup_pct":limup_pct,
        "limdown_pct":limdown_pct,
        "ret1":ret1,
        "ret5":ret5,
        "ret20":ret20,
        "d1":d1,
        "d1_5":d1.rolling(5).mean(),
        "d1_20":d1.rolling(20).mean(),
        "d2":d2,
        "d2_5":d2.rolling(5).mean(),
        "d2_20":d2.rolling(20).mean(),
        "vol20":vol20,
        "risk20":risk20,
        "liq":liq,
        "adx":adx
    }
    return mkt_dict
