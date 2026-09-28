from enum import StrEnum

class UniverseType(StrEnum):
    general = "liquidity_amounts_60d_top75pct_min10M"
    stable = "stables"
    standard = "standards"
    dividend = "universe_liq_div3y"
    small = "smalls"

class IndexType(StrEnum):
    sz50 = "sz50s_returns"
    hs300 = "hs300s_returns"
    zz500 = "zz500s_returns"
    zz800 = "zz800s_returns"
    zz1000 = "zz1000s_returns"
    msci = "mscias_returns"

class TradepointType(StrEnum):
    close = "close"
    open = "open"
    vwap = "vwap"

class DataType(StrEnum):
    days = "days"
    minutes = "minutes"
    ticks = "ticks"

class CategoryType(StrEnum):
    mom = "mom"     # 动量
    rev = "rev"     # 反转
    liq = "liq"     # 流动
    vol = "vol"     # 波动
    pvc = "pvc"     # 量价关系
    rs = "rs"       # 相对强弱
    path = "path"   # 路径形态
    dist = "dist"   # 分布
    time = "time"   # 时段结构
    micro = "micro" # 微观结构
    unknown = "unknown"

class DomainType(StrEnum):
    pv = "pv"
    fundamental = "fundamental"
    hybrid = "hybrid"
