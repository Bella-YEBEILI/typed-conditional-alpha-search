CONS = [
    "ret1_ts_5","ret5_ts_20","vol5_ts_20","to1_ts_5","to5_ts_20","pv5_ts_20","illiq1_ts_5","illiq5_ts_20","eff20_ts_60",
    "ret1_cs_mkt","ret5_cs_mkt","ret20_cs_mkt","vol5_cs_mkt","vol20_cs_mkt","to1_cs_mkt","to20_cs_mkt","intraday_cs_mkt","gap_cs_mkt","amplitude_cs_mkt","body_cs_mkt","upper_cs_mkt","lower_cs_mkt","pos_b_cs_mkt","pv5_cs_mkt","pv20_cs_mkt","illiq5_cs_mkt","illiq20_cs_mkt","eff20_cs_mkt","eff60_cs_mkt",
    "value_ts_240","asset_return_ts_240","margin_ts_240","turnover_rate_ts_240","longdebt_ratio_ts_240","leverage_rate_ts_240",
    "value_cs_sector","asset_return_cs_sector","margin_cs_sector","turnover_rate_cs_sector","cashflow_quality_cs_sector","longdebt_ratio_cs_sector","leverage_rate_cs_sector","operate_growth_cs_sector",
]

NOTCONS = ["con_not("+con+")" for con in CONS]

CONFIG = {
    "threads":72,

    "selector":{
        "CANDIDATE":{
            "min_rankic":0.045,
            "min_rankicir":0.4,
            "min_ic":0.025,
            "min_turnover":0.01,
            "max_turnover":1.0,
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
            (0.5,0.7,5),
            (0.7,1.0,10),
            (1.0,2.0,20),
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
            "corr_threshold":0.5,
            "score_improve":1.0,
        },
        "seed":{
            "corr_threshold":0.7,
        },
    },

    "os_checker":{
        "CANDIDATE":{
            "min_rankic":0.04,
        },
    },

    "source":{
        "atoms":["neg(ret1)"],
        "layers":[1,2,3],
        "paths":None,
        "factories":None,
        "seed_prune_passed":True,
        "low_rankic":None,
        "reprune":None,
        "limit":None,
    },

    "strategy":{
        "module":"default_cross_single",
        "params":{
            "random_seed":42,
            "same_layer_same_factory":{
                "limit":4000,
                "ops":{
                    "subr":[],
                    "div":[],
                    "ts_corr":[[20,60,240],[0]],
                    "ts_reg":[[20,60,240],[0]],
                    "cs_reg":[[0]], 
                },
            },
            "same_factory_diff_layer":{
                "limit":4000,
                "ops":{
                    "ts_reg":[[20,60],[0]],
                    "cs_reg":[[0]], 
                }
            },
            "diff_path":{
                "limit":12000,
                "ops":{
                    "addr":[],
                    "subr":[],
                    "mulr":[],
                    "mulz":[],
                    "div":[],
                    "if_else":[CONS,],
                }
            }
        },
    },
}
