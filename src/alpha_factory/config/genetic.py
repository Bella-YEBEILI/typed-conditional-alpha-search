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

    "run":{
        "random_seed":42,
        "max_generations":20,
        "children_per_generation":1000,
        "stop_when_product_hit":True,
    },

    "population":{
        "size":200,
        "top_fitness_ratio":0.7,
        "low_prod_corr_ratio":0.2,
        "random_ratio":0.1,
    },

    "source":{
        "atoms":["ret1","to1","amplitude","ep","bp","roe"],
    },

    "genetic":{
        "mutation":{
            "wrap_op":0.7,
            "replace_leaf":0.3,
        },
        "field_filter":{
            "ban_fields":(),
        },
        "condition_pool":None,
        "group_pool":None,
        "banned_ops":("math_self","ts_self","group_self"),
        "group_merge_prob":0.2,
    },
}
