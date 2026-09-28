from itertools import product
import json
import random

from ..core.factor import make_factor
from ..core.id_manager import IdManager
from .job_key import make_job_key


ORDERLESS_OPS = {"addr","subr","mulr","mulz","if_else","ts_corr"}
DIRECTIONAL_OPS = {"ts_reg","cs_reg","div"}
PAIR_TYPES = ("same_layer_same_factory","same_factory_diff_layer","diff_path")


def generate(factors,done_jobs,params):
    """
    Cross single-atom expansion:
    split pairs into same_layer_same_factory, same_factory_diff_layer, and diff_path buckets.
    Each bucket owns its own budget and op config; jobs are sampled round-robin across pairs.
    Orderless ops use one stable left/right order. Directional ops use both orders for same-layer
    and diff-path pairs, while diff-layer pairs use smaller layer as x and larger layer as y.
    In same_layer_same_factory, regression-style ops such as ts_reg/cs_reg are sampled both ways.
    In same_factory_diff_layer, regression-style ops use the smaller-layer factor to explain the larger-layer factor.
    Completed jobs are filtered by job_key and replaced by later choices while capacity remains.
    """
    rng = random.Random(params.get("random_seed",42))
    done = set(done_jobs)
    generated = []
    records = []

    for pair_type in PAIR_TYPES:
        config = params.get(pair_type)
        if config is None:
            continue

        pairs = _make_pairs(factors,pair_type)
        rng.shuffle(pairs)
        _generate_bucket(
            pair_type=pair_type,
            pairs=pairs,
            config=config,
            rng=rng,
            done=done,
            generated=generated,
            records=records,
        )

    return generated,records


def _generate_bucket(pair_type,pairs,config,rng,done,generated,records):
    if len(pairs)==0:
        return

    ops_config = config["ops"]
    choices_by_pair = [_cross_choices(pair_type,pair,ops_config,rng) for pair in pairs]
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


def _make_pairs(factors,pair_type):
    items = sorted(factors,key=_factor_sort_key)
    pairs = []

    for i,left in enumerate(items):
        for right in items[i+1:]:
            if left.meta.atom!=right.meta.atom:
                continue
            if pair_type=="same_layer_same_factory":
                if left.meta.layer==right.meta.layer and left.meta.factory==right.meta.factory:
                    pairs.append(_stable_pair(left,right))
            elif pair_type=="same_factory_diff_layer":
                if left.meta.factory==right.meta.factory and left.meta.layer!=right.meta.layer:
                    pairs.append(_layer_pair(left,right))
            elif pair_type=="diff_path":
                if left.meta.path!=right.meta.path:
                    pairs.append(_stable_pair(left,right))

    return pairs


def _cross_choices(pair_type,pair,ops_config,rng):
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
            for x,y in _oriented_pairs(pair_type,op,left,right):
                choices.append((op,op_params,x,y))
        rng.shuffle(choices)
        if len(choices)>0:
            first.append(choices[0])
        for choice in choices[1:]:
            rest.append(choice)

    rng.shuffle(rest)
    return first+rest


def _oriented_pairs(pair_type,op,left,right):
    if op in DIRECTIONAL_OPS:
        if pair_type=="same_factory_diff_layer":
            return [(left,right)]
        return [(left,right),(right,left)]
    return [(left,right)]


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
        atom=left.meta.atom,
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


def _layer_pair(left,right):
    left_layer = -1 if left.meta.layer is None else left.meta.layer
    right_layer = -1 if right.meta.layer is None else right.meta.layer
    if left_layer<right_layer:
        return left,right
    if right_layer<left_layer:
        return right,left
    return _stable_pair(left,right)


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
