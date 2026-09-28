import random

from .default_cross_multi import (
    _split_by_atom,
    _make_pairs,
    _generate_bucket,
)


def generate(factors,done_jobs,params):
    """
    Common cross multi-atom expansion:
    this strategy only handles pairs from two different atoms; same-atom pairs belong to default_cross_single.
    It does not split pairs by high/low correlation. All cross-atom seed pairs share one budget and one op config.
    Jobs are sampled round-robin across pairs, completed path jobs are skipped, and later choices refill
    the budget until capacity is exhausted.
    """
    rng = random.Random(params.get("random_seed",42))
    atom_groups = _split_by_atom(factors)
    atoms = sorted(atom_groups.keys())
    if len(atoms)!=2:
        return [],[]

    pairs = _make_pairs(atom_groups[atoms[0]],atom_groups[atoms[1]])
    if len(pairs)==0:
        return [],[]

    done = set(done_jobs)
    generated = []
    records = []
    config = params.get("cross",params)

    _generate_bucket(
        pairs=pairs,
        config=config,
        rng=rng,
        done=done,
        generated=generated,
        records=records,
    )

    return generated,records
