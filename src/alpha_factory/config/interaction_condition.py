CONS = [
    "ret1_ts_5","ret5_ts_20","vol5_ts_20","to1_ts_5","to5_ts_20","pv5_ts_20","illiq1_ts_5","illiq5_ts_20","eff20_ts_60",
    "ret1_cs_mkt","ret5_cs_mkt","ret20_cs_mkt","vol5_cs_mkt","vol20_cs_mkt","to1_cs_mkt","to20_cs_mkt","intraday_cs_mkt","gap_cs_mkt","amplitude_cs_mkt","body_cs_mkt","upper_cs_mkt","lower_cs_mkt","pos_b_cs_mkt","pv5_cs_mkt","pv20_cs_mkt","illiq5_cs_mkt","illiq20_cs_mkt","eff20_cs_mkt","eff60_cs_mkt",
    "value_ts_240","asset_return_ts_240","margin_ts_240","turnover_rate_ts_240","longdebt_ratio_ts_240","leverage_rate_ts_240",
    "value_cs_sector","asset_return_cs_sector","margin_cs_sector","turnover_rate_cs_sector","cashflow_quality_cs_sector","longdebt_ratio_cs_sector","leverage_rate_cs_sector","operate_growth_cs_sector",
]

NOTCONS = ["con_not("+con+")" for con in CONS]

CONFIG = {
    "seeds":["neg(ret1)","to1"],
    "threads":72,

    "selector":{
        "CANDIDATE":{
            "min_rankic":0.03,
            "min_rankicir":0.3,
            "min_ic":0.015,
            "min_turnover":0.01,
            "max_turnover":1.5,
            "min_coverage":0.7,
        },
        "SEED":{
            "min_rankic":0.02,
            "min_coverage":0.7,
        },
        "PERF_SCORE":{
            "rankic":10.0,
            "rankicir":1.0,
        },
    },

    "adjuster":{
        "min_abs_rankic":0.01,
        "turnover_decay":[
            (0.5,1.0,5),
            (1.0,1.5,10),
            (1.5,2.0,20),
        ],
    },

    "marker":{
        "structure":{
            "fundamental_bonus":0.5,
            "mixed_bonus":0.0,
        },
        "fitness":{
            "perf_weight":1.0,
            "structure_weight":1.0,
            "batch_corr_penalty":0.5,
            "prod_corr_penalty":0.5,
        },
        "candidate":{
            "corr_threshold":0.6,
            "score_improve":1.1,
        },
        "seed":{
            "corr_threshold":0.7,
        },
    },

    "os_checker":{
        "CANDIDATE":{
            "min_rankic":0.03,
        },
    },

    "source":{
        "layers":[1,2,3],
        "paths":None,
        "factories":None,
        "seed_prune_passed":True,
        "low_rankic":None,
        "reprune":None,
        "limit":None,
    },

    "strategy":{
        "module":"default_condition",
        "params":{
            "limit":10000,
            "random_seed":42,
            "ops":{
                "adjust_by":[CONS+NOTCONS,[0.5]],
                "reverse_by":[CONS],
                "reverse_rank_by":[CONS+NOTCONS],
                "trade_when":[CONS+NOTCONS,[0.2]],
            },
        },
    },
}
