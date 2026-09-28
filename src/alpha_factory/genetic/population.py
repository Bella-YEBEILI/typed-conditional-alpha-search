import random

import numpy as np

from ..core.factor import Factor
from ..core.id_manager import IdManager


class PopulationSelector:
    def __init__(self,config:dict):
        self.size = config.get("size",200)
        self.top_fitness_ratio = config.get("top_fitness_ratio",0.7)
        self.low_prod_corr_ratio = config.get("low_prod_corr_ratio",0.2)
        self.random_ratio = config.get("random_ratio",0.1)

    def select(self,
               old_population:list[Factor],
               passed_seeds:list[Factor],
               evaluated:list[Factor],
               rng:random.Random)->list[Factor]:
        selected = []
        seen = set()

        n_top = int(self.size*self.top_fitness_ratio)
        n_low = int(self.size*self.low_prod_corr_ratio)
        n_random = self.size-n_top-n_low

        top_source = sorted(
            old_population+passed_seeds,
            key=self._fitness,
            reverse=True,
        )
        self._extend(selected,seen,top_source,n_top)

        low_source = sorted(
            self._valid_evaluated(evaluated),
            key=self._prod_corr,
        )
        self._extend(selected,seen,low_source,n_low)

        random_source = self._valid_evaluated(evaluated)
        rng.shuffle(random_source)
        self._extend(selected,seen,random_source,n_random)

        if len(selected)<self.size:
            fill = top_source+low_source+random_source
            self._extend(selected,seen,fill,self.size-len(selected))

        return selected[:self.size]

    def _extend(self,selected,seen,source,count):
        for factor in source:
            if len(selected)>=self.size or count<=0:
                return
            formula_id = IdManager.get_formula_id(factor.spec.formula)
            if formula_id in seen:
                continue
            seen.add(formula_id)
            factor.spec.decay = 0
            factor.spec.neutralize = None
            selected.append(factor)
            count -= 1

    @staticmethod
    def _valid_evaluated(factors:list[Factor])->list[Factor]:
        out = []
        for factor in factors:
            r = factor.result
            if r.coverage is not None and np.isfinite(r.coverage):
                out.append(factor)
        return out

    @staticmethod
    def _fitness(factor:Factor):
        fitness = factor.result.fitness
        if fitness is None or not np.isfinite(fitness):
            return -np.inf
        return float(fitness)

    @staticmethod
    def _prod_corr(factor:Factor):
        corr = factor.result.prod_corr
        if corr is None or not np.isfinite(corr):
            return 0.0
        return float(corr)
