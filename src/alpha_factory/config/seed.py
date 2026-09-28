LAYERS = {
    1:("math","ts","group"),
    2:("math","ts","group"),
    3:("math","ts","group"),
}

WINDOWS = [5,20,60,120,240]

CUT_FLDS = ["ret1","resret","to1","vol5","amplitude","pos_a","illiq1"]

GROUPS = ["sectors","industrys","subindustrys",
          "ret5_g5","vol5_g5","Size_g10","Mom_g10","Vol_g10","Liq_g10","Beta_g10",
          "Bp_g3","OQuality_g3","CQuality_g3","Growth_g3","Improve_g3","Leverage_g3"]


CONFIG = {
    "atoms":["neg(ret1)"],
    "threads":72,
    "layers":LAYERS,

    "seed_search":{
        "seed_prune_whitelist_ops":("math_self","cs_rank","cs_zscore","log","sqrt"),
        "last_layer_blacklist_ops":("math_self","cs_rank","cs_zscore"),
    },

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

    "factories":{
        "math":{
            "cs_rank":[],
            "cs_zscore":[],
            "abs":[],
            "power":[[2]],
            "sqrt":[],
            "log":[],
            "diff":[],
            "diff_q":[],
            "tanh":[],
            "midmap":[],
            "rightmap":[],
            "leftmap":[],
            "extreme_leftmap":[],
            "extreme_rightmap":[],
        },

        "ts":{
            "ts_delta":[WINDOWS],
            "ts_pct":[WINDOWS],
            "ts_rank":[WINDOWS],
            "ts_decay_linear":[WINDOWS],
            "ts_mean":[WINDOWS],
            "ts_std":[WINDOWS],
            "ts_skew":[WINDOWS],
            "ts_kur":[WINDOWS],
            "ts_zscore":[WINDOWS],
            "ts_range":[WINDOWS],
            "ts_median":[WINDOWS],
            "ts_argmax":[WINDOWS],
            "ts_argmin":[WINDOWS],
            "ts_min":[WINDOWS],
            "ts_ir":[WINDOWS],
            "ts_cut_diff":[CUT_FLDS,WINDOWS],
            "ts_cut_normdiff":[CUT_FLDS,WINDOWS],
        },

        "group":{
            "group_rank":[GROUPS],
            "group_midmap":[GROUPS],
            "group_leftmap":[GROUPS],
            "group_rightmap":[GROUPS],
            "group_neutralize":[GROUPS],
            "group_zscore":[GROUPS],
            "group_scale":[GROUPS],
        },
    },

    "os_checker":{
        "CANDIDATE":{
            "min_rankic":0.04,
        },
    },


}
