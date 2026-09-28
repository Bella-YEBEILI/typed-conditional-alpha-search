import numpy as np

FACTOR_START_DATE = "20160101"
FACTOR_END_DATE = "20260101"

# 硬性标准

DEFAULT_CRITERIA = {
    "raw":{
        "long_ret":{"min":0.08},
        "long_netret":{"min":0.0},
        "ls_ir":{"min":1.5},
        "ls_netret":{"min":0.0},
        "rankic":{"min":0.01},
        "rankicir":{"min":3.0},
        "long_turnover":{"min":0.05,"max":1.0},
        "coverage":{"min":0.7},
    },
    "zz1000s":{
        "ls_ir":{"min":1.0}
    },
    "complete":{
        "ls_ir":{"min":0.0},
        "rankicir":{"min":0.0}
    }
}

CRITERIA = {
    ("standards","pv"):DEFAULT_CRITERIA,
    ("standards","fundamental"):DEFAULT_CRITERIA,
    ("standards","hybrid"):DEFAULT_CRITERIA
}

# 连续评分规则
def _sigmod(x:float)->float:
    return 1/(1+np.exp(-x))

def _default_scoring(perf:dict)->float:
    """
    z分数: (x-med)/mad
    med为库内因子指标中位数
    mad为库内因子指标的中位数绝对偏差
    得到z分数后做sigmod变换映射到01区间
    """
    # 进攻能力
    long_ret = _sigmod((perf.get("long_ret",0)-0.12)/0.02)
    long_netret = _sigmod((perf.get("long_netret",0)-0.07)/0.02)
    
    score1 = (long_ret+long_netret)/2

    # 平衡能力
    ls_ir = _sigmod((perf.get("ls_ir",0)-2.48)/0.43)
    ls_netir = _sigmod((perf.get("ls_netir",0)-1.33)/0.43)
    rankic = _sigmod((perf.get("rankic",0)-0.03)/0.0085)
    rankicir = _sigmod((perf.get("rankicir",0)-5.58)/1.06)
    mono = _sigmod((perf.get("mono",0)-0.09)/0.016)
    monoir = _sigmod((perf.get("monoir",0)-0.16)/0.36)

    score2 = ((ls_ir+ls_netir)/2+(rankic+rankicir)/2+(mono+monoir)/2)/3

    return score1*0.5+score2*0.5

SCORING = {
    ("standards","pv"):_default_scoring,
    ("standards","fundamental"):_default_scoring,
    ("standards","hybrid"):_default_scoring,
}
