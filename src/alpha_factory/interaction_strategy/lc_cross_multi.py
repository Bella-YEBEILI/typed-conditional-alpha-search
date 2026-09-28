import heapq
import math
import random

import numpy as np

from .default_cross_multi import (
    _split_by_atom,
    _make_pairs,
    _split_by_corr,
    _cross_choices,
    _make_item,
)


def generate(factors,done_jobs,params):
    """
    Low-correlation weighted cross multi:
    first split cross-atom pairs into high_corr / low_corr buckets exactly like default_cross_multi.
    Within each bucket, pair scheduling is weighted by each seed's latest product correlation:
    seed_weight=max(floor,(1-prod_corr)**alpha), pair_weight=sqrt(left_weight*right_weight).
    Negative product-correlation seeds receive more cross budget. Completed path jobs are skipped,
    and each pair continues to later op/params choices until exhausted.
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
        alpha=params.get("prod_corr_alpha",2.0),
        floor=params.get("prod_corr_floor",0.05),
    )
    _generate_bucket(
        pairs=low_pairs,
        config=params.get("low_corr"),
        rng=rng,
        done=done,
        generated=generated,
        records=records,
        alpha=params.get("prod_corr_alpha",2.0),
        floor=params.get("prod_corr_floor",0.05),
    )

    return generated,records


def _generate_bucket(pairs,config,rng,done,generated,records,alpha:float,floor:float):
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
    heap = []

    for i,choices in enumerate(choices_by_pair):
        if len(choices)==0:
            continue
        weight = _pair_weight(pairs[i],alpha,floor)
        heapq.heappush(heap,(rng.random()/weight,i,weight))

    while budget>0 and len(heap)>0:
        priority,i,weight = heapq.heappop(heap)
        choices = choices_by_pair[i]
        item = None

        while cursors[i]<len(choices):
            op,op_params,left,right = choices[cursors[i]]
            cursors[i] += 1
            item = _make_item(op,op_params,left,right)
            if item["job_key"] in done:
                item = None
                continue
            break

        if item is None:
            continue

        done.add(item["job_key"])
        generated.append(item["factor"])
        records.append(item["record"])
        budget -= 1

        if cursors[i]<len(choices):
            heapq.heappush(heap,(priority+1.0/weight,i,weight))


def _pair_weight(pair,alpha:float,floor:float)->float:
    left,right = pair
    return math.sqrt(_seed_weight(left,alpha,floor)*_seed_weight(right,alpha,floor))


def _seed_weight(factor,alpha:float,floor:float)->float:
    corr = factor.result.prod_corr
    if corr is None or not np.isfinite(corr):
        corr = 0.0
    corr = min(1.0,max(-1.0,float(corr)))
    return max(float(floor),(1.0-corr)**float(alpha))
