from itertools import product
import json
import random

from ..core.factor import make_factor
from ..core.id_manager import IdManager
from .job_key import make_job_key

CONS = [
    "ret1_ts_5","ret5_ts_20","vol5_ts_20","to1_ts_5","to5_ts_20","pv5_ts_20","illiq1_ts_5","illiq5_ts_20","eff20_ts_60",
    "ret1_cs_mkt","ret5_cs_mkt","ret20_cs_mkt","vol5_cs_mkt","vol20_cs_mkt","to1_cs_mkt","to20_cs_mkt","intraday_cs_mkt","gap_cs_mkt","amplitude_cs_mkt","body_cs_mkt","upper_cs_mkt","lower_cs_mkt","pos_b_cs_mkt","pv5_cs_mkt","pv20_cs_mkt","illiq5_cs_mkt","illiq20_cs_mkt","eff20_cs_mkt","eff60_cs_mkt",
    "value_ts_240","asset_return_ts_240","margin_ts_240","turnover_rate_ts_240","longdebt_ratio_ts_240","leverage_rate_ts_240",
    "value_cs_sector","asset_return_cs_sector","margin_cs_sector","turnover_rate_cs_sector","cashflow_quality_cs_sector","longdebt_ratio_cs_sector","leverage_rate_cs_sector","operate_growth_cs_sector",
]

NOTCONS = ["con_not("+con+")" for con in CONS]

DEFAULT_PARAMS = {
            "limit":10000,
            "random_seed":42,
            "ops":{
                "adjust_by":[CONS+NOTCONS,[0.5]],
                "reverse_by":[CONS],
                "reverse_rank_by":[CONS+NOTCONS],
                "trade_when":[CONS+NOTCONS,[0.2]],
            },
        }

def generate(factors,done_jobs,params):
    """
    Condition expansion:
    expand every op parameter axis into full choices, then split the sampling budget across input seeds.
    For each seed, shuffle ops, take one choice per op first, then shuffle the remaining choices.
    Before sampling, build a generic job_key from job_type+left+right+op+params,
    so completed jobs are skipped and replaced by later choices when the pool still has capacity.
    """
    rng = random.Random(params.get("random_seed",42))
    ops_config = params["ops"]
    limit = params.get("limit")

    choices_by_op = {}
    for op,axes in ops_config.items():
        choices_by_op[op] = list(product(*axes))

    n = len(factors)
    choices_per_factor = sum(len(v) for v in choices_by_op.values())
    total_capacity = n*choices_per_factor
    if n==0 or total_capacity==0:
        return [],[]

    budget = total_capacity if limit is None else min(int(limit),total_capacity)
    base = budget//n
    extra = budget%n
    done = set(done_jobs)
    generated = []
    records = []

    for i,factor in enumerate(factors):
        target = base+(1 if i<extra else 0)
        if target<=0:
            continue

        count = 0
        for op,op_params in _condition_choices(choices_by_op,rng):
            left_formula_id = _formula_id(factor.spec.formula,factor.spec.formula_id)
            job_key = make_job_key(
                job_type="condition",
                left_formula_id=left_formula_id,
                right_formula_id=None,
                op=op,
                params=op_params,
            )
            if job_key in done:
                continue

            formula = _formula(op,factor.spec.formula,op_params)
            done.add(job_key)
            child = make_factor(
                formula,
                atom=factor.meta.atom,
                layer=factor.meta.layer,
                path=factor.meta.path,
                factory="condition",
            )
            child.interaction_path = _record(
                factor=factor,
                kind="condition",
                op=op,
                params=op_params,
                seed1=left_formula_id,
            )
            generated.append(child)
            records.append(_record(
                factor=factor,
                kind="condition",
                op=op,
                params=op_params,
                seed1=left_formula_id,
            ))
            count += 1
            if count>=target:
                break

    return generated,records


def _condition_choices(choices_by_op,rng):
    ops = list(choices_by_op.keys())
    rng.shuffle(ops)
    first = []
    rest = []

    for op in ops:
        params_list = choices_by_op[op].copy()
        rng.shuffle(params_list)
        if len(params_list)>0:
            first.append((op,params_list[0]))
        for op_params in params_list[1:]:
            rest.append((op,op_params))

    rng.shuffle(rest)
    return first+rest


def _formula(op,base_formula,op_params):
    args = ",".join([str(p) for p in op_params])
    suffix = ","+args if args else ""
    return f"{op}({base_formula}{suffix})"


def _record(factor,kind,op,params,seed1):
    if len(factor.meta.atom)!=1:
        raise ValueError(f"condition seed must have one atom: {factor.meta.atom}")
    return {
        "kind":kind,
        "op":op,
        "params":json.dumps(list(params),sort_keys=True,ensure_ascii=False),
        "seed1":seed1,
        "seed1_atom":factor.meta.atom[0],
        "seed1_layer":factor.meta.layer,
        "seed1_path":factor.meta.path,
        "seed1_factory":factor.meta.factory,
        "seed2":None,
        "seed2_layer":None,
        "seed2_path":None,
        "seed2_factory":None,
    }


def _formula_id(formula,formula_id):
    if formula_id is not None:
        return formula_id
    return IdManager.get_formula_id(formula)
