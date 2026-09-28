from itertools import product
import json
import random

import numpy as np

from ..core.factor import make_factor
from ..core.id_manager import IdManager
from ..factory_ops.numpy_funcs import p2p_corr
from .job_key import make_job_key


ORDERLESS_OPS = {"addr","subr","mulr","mulz","if_else","ts_corr"}
DIRECTIONAL_OPS = {"ts_reg","cs_reg","div"}


def generate(factors,done_jobs,params):
    """
    Cross multi-atom expansion:
    this strategy only handles pairs from two different atoms; same-atom pairs belong to default_cross_single.
    It builds all cross-atom seed pairs, computes abs(rankics correlation), splits pairs by corr_quantile,
    and gives high_corr / low_corr their own budgets and op configs.
    Orderless ops use one stable left/right order. Directional ops such as div/ts_reg/cs_reg are sampled both ways.
    Completed jobs are filtered by job_key and replaced by later choices while capacity remains.
    """
    rng = random.Random(params.get("random_seed",42))
    atom_groups = _split_by_atom(factors)
    atoms = sorted(atom_groups.keys())
    if len(atoms)!=2:
        return [],[]

    pairs = _make_pairs(atom_groups[atoms[0]],atom_groups[atoms[1]])
    if len(pairs)==0:
        return [],[]

    high_pairs,low_pairs = _split_by_corr(pairs,params.get("corr_quantile",0.5))
    done = set(done_jobs)
    generated = []
    records = []

    _generate_bucket(
        pairs=high_pairs,
        config=params.get("high_corr"),
        rng=rng,
        done=done,
        generated=generated,
        records=records,
    )
    _generate_bucket(
        pairs=low_pairs,
        config=params.get("low_corr"),
        rng=rng,
        done=done,
        generated=generated,
        records=records,
    )

    return generated,records


def _generate_bucket(pairs,config,rng,done,generated,records):
    if config is None or len(pairs)==0:
        return

    rng.shuffle(pairs)
    ops_config = config["ops"]
    choices_by_pair = [_cross_choices(pair,ops_config,rng) for pair in pairs]
    capacity = sum(len(choices) for choices in choices_by_pair)
    if capacity==0:
        return

    limit = config.get("limit")
    budget = capacity if limit is None else min(int(limit),capacity)
    cursors = [0]*len(pairs)

    while budget>0:
        added = False
        for i,choices in enumerate(choices_by_pair):
            while cursors[i]<len(choices):
                op,op_params,left,right = choices[cursors[i]]
                cursors[i] += 1
                item = _make_item(op,op_params,left,right)
                if item["job_key"] in done:
                    continue

                done.add(item["job_key"])
                generated.append(item["factor"])
                records.append(item["record"])
                budget -= 1
                added = True
                break

            if budget<=0:
                break

        if not added:
            break


def _cross_choices(pair,ops_config,rng):
    left,right = pair
    ops = list(ops_config.keys())
    rng.shuffle(ops)
    first = []
    rest = []

    for op in ops:
        params_list = list(product(*ops_config[op]))
        rng.shuffle(params_list)
        choices = []
        for op_params in params_list:
            for x,y in _oriented_pairs(op,left,right):
                choices.append((op,op_params,x,y))
        rng.shuffle(choices)
        if len(choices)>0:
            first.append(choices[0])
        for choice in choices[1:]:
            rest.append(choice)

    rng.shuffle(rest)
    return first+rest


def _oriented_pairs(op,left,right):
    if op in DIRECTIONAL_OPS:
        return [(left,right),(right,left)]
    return [_stable_pair(left,right)]


def _make_pairs(left_factors,right_factors):
    pairs = []
    for left in sorted(left_factors,key=_factor_sort_key):
        for right in sorted(right_factors,key=_factor_sort_key):
            if left.meta.atom==right.meta.atom:
                continue
            pairs.append(_stable_pair(left,right))
    return pairs


def _split_by_atom(factors):
    groups = {}
    for factor in factors:
        groups.setdefault(_json_tuple(factor.meta.atom),[]).append(factor)
    return groups


def _split_by_corr(pairs,corr_quantile):
    values = []
    for left,right in pairs:
        corr = p2p_corr(left.result.rankics,right.result.rankics)
        if not np.isfinite(corr):
            corr = 0.0
        values.append(abs(corr))

    threshold = float(np.quantile(np.array(values,dtype=np.float64),corr_quantile))
    high = []
    low = []
    for pair,value in zip(pairs,values):
        if value>=threshold:
            high.append(pair)
        else:
            low.append(pair)
    return high,low


def _make_item(op,op_params,left,right):
    left_formula_id = _formula_id(left)
    right_formula_id = _formula_id(right)
    job_key = make_job_key(
        job_type="cross",
        left_formula_id=left_formula_id,
        right_formula_id=right_formula_id,
        op=op,
        params=op_params,
    )
    formula = _formula(op,left.spec.formula,right.spec.formula,op_params)
    factor = make_factor(
        formula,
        atom=left.meta.atom+right.meta.atom,
        layer=None,
        path=None,
        factory="cross",
    )
    record = {
        "kind":"cross",
        "op":op,
        "params":json.dumps(list(op_params),sort_keys=True,ensure_ascii=False),
        "seed1":left_formula_id,
        "seed1_atom":_atom_text(left.meta.atom),
        "seed1_layer":left.meta.layer,
        "seed1_path":left.meta.path,
        "seed1_factory":left.meta.factory,
        "seed2":right_formula_id,
        "seed2_layer":right.meta.layer,
        "seed2_path":right.meta.path,
        "seed2_factory":right.meta.factory,
    }
    factor.interaction_path = record

    return {
        "job_key":job_key,
        "factor":factor,
        "record":record,
    }


def _formula(op,left_formula,right_formula,op_params):
    args = ",".join([str(p) for p in op_params])
    suffix = ","+args if args else ""
    return f"{op}({left_formula},{right_formula}{suffix})"


def _stable_pair(left,right):
    if _factor_sort_key(left)<=_factor_sort_key(right):
        return left,right
    return right,left


def _factor_sort_key(factor):
    return (
        _json_tuple(factor.meta.atom),
        -1 if factor.meta.layer is None else factor.meta.layer,
        "" if factor.meta.path is None else factor.meta.path,
        "" if factor.meta.factory is None else factor.meta.factory,
        _formula_id(factor),
    )


def _formula_id(factor):
    if factor.spec.formula_id is not None:
        return factor.spec.formula_id
    return IdManager.get_formula_id(factor.spec.formula)


def _json_tuple(value):
    return json.dumps(list(value),ensure_ascii=False)


def _atom_text(value):
    if len(value)==1:
        return value[0]
    return json.dumps(list(value),ensure_ascii=False)
