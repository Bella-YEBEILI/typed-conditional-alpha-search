import numpy as np
from itertools import combinations
from typing import Union

from ..core.factor import Factor,FactorMeta,FactorSpec
from ..core.evaluator import Evaluator
from ..core.checker import OSChecker,FullChecker
from ..core.id_manager import IdManager
from ..core.result_marker import ResultMarker
from ..factory_ops.numpy_funcs import p2p_corr
from ..repository.product_repository import ProductRepository


class ProductManager:
    def __init__(self,
                 is_evaluator:Evaluator,
                 os_checker:OSChecker,
                 full_checker:FullChecker,
                 marker:ResultMarker,
                 product_repository:ProductRepository,
                 threads:int=8):
        self.is_evaluator = is_evaluator
        self.os_checker = os_checker
        self.full_checker = full_checker
        self.marker = marker
        self.product_repository = product_repository
        self.threads = threads

    def load_product_factors(self)->list[Factor]:
        factors = self.product_repository.load_factors()
        if self._ensure_structure_score(factors):
            self.product_repository.save_is_batch(factors)
        return factors

    def process_candidates(self,candidates:list[Factor])->list[Factor]:
        candidates = [factor for factor in candidates if factor.result.is_candidate==True]
        if len(candidates)==0:
            return []

        self._ensure_structure_score(candidates)
        self.product_repository.save_is_batch(candidates)
        os_all = self.os_checker.evaluate(candidates)
        self.product_repository.save_os_batch(os_all)
        os_factors = [factor for factor in os_all if factor.result.is_candidate==True]
        full_factors = self.full_checker.check(os_factors)
        self.product_repository.save_full_batch(full_factors)
        self.refresh_product_corrs()
        return full_factors

    def check_submission(self,target:Union[str,dict,Factor])->dict:
        is_factor,os_factor,full_factor = self._evaluate_submission(target)
        corr_stats = self._full_corr_stats(full_factor)
        return self._submission_result(is_factor,os_factor,full_factor,corr_stats)

    def submit(self,target:Union[str,dict,Factor])->dict:
        is_factor,os_factor,full_factor = self._evaluate_submission(target)
        self.product_repository.save_is_batch([is_factor])
        self.product_repository.save_os_batch([os_factor])
        self.product_repository.save_full_batch([full_factor])
        self.refresh_product_corrs()
        corr_stats = self._full_corr_stats(full_factor)
        out = self._submission_result(is_factor,os_factor,full_factor,corr_stats)
        out["submitted"] = True
        return out

    def delete(self,spec_id:str)->dict:
        self.product_repository.delete_factor(spec_id)
        self.refresh_product_corrs()
        return {"spec_id":spec_id,"deleted":True}

    def optimize_product_exact(self,corr_threshold:float=0.5,apply:bool=False)->dict:
        factors = self.product_repository.load_full_factors()
        n = len(factors)
        if n==0:
            return {"keep_ids":[],"delete_ids":[]}

        conflict = self._product_conflict_matrix(factors,corr_threshold)
        best = ()
        best_score = None

        for size in range(n,0,-1):
            for combo in combinations(range(n),size):
                if not self._is_independent(combo,conflict):
                    continue
                score = sum(float(factors[i].result.rankic) for i in combo)
                if best_score is None or score>best_score:
                    best = combo
                    best_score = score
            if len(best)>0:
                break

        keep = set(best)
        keep_ids = [factors[i].spec.spec_id for i in range(n) if i in keep]
        delete_ids = [factors[i].spec.spec_id for i in range(n) if i not in keep]

        if apply:
            for spec_id in delete_ids:
                self.product_repository.delete_factor(spec_id)
            self.refresh_product_corrs()

        return {
            "keep_ids":keep_ids,
            "delete_ids":delete_ids,
            "keep_count":len(keep_ids),
            "delete_count":len(delete_ids),
            "rankic_mean":best_score/len(keep_ids) if keep_ids else None,
            "applied":apply,
        }

    def refresh_product_corrs(self):
        full_factors = self.product_repository.load_full_factors()
        for factor in full_factors:
            stats = self._corr_stats(
                factor.result.rankics,
                full_factors,
                exclude_spec_id=factor.spec.spec_id,
            )
            self.product_repository.update_corr_fields(
                factor.spec.spec_id,
                stats["max_corr"],
                stats["max_corr_id"],
                stats["avg_corr"],
                tables=("product_full",),
            )

    def _make_factor(self,target:Union[str,dict,Factor])->Factor:
        if isinstance(target,Factor):
            factor = self._clone_factor(target)
            self._bind_spec_id(factor)
            return factor
        if isinstance(target,str):
            spec = FactorSpec(formula=target)
        else:
            spec = FactorSpec(
                formula=target["formula"],
                decay=target.get("decay",0),
                neutralize=target.get("neutralize"),
            )
        spec.formula_id = IdManager.get_formula_id(spec.formula)
        spec.spec_id = IdManager.get_spec_id(spec.formula,spec.decay,spec.neutralize)
        return Factor(meta=FactorMeta(atom=("submission",)),spec=spec)

    def _evaluate_submission(self,target:Union[str,dict,Factor])->tuple[Factor,Factor,Factor]:
        is_factor = self.is_evaluator.multi_evaluate([self._make_factor(target)],self.threads)[0]
        self._ensure_structure_score([is_factor])
        os_factor = self.os_checker.evaluate([is_factor])[0]
        full_factor = self.full_checker.check([is_factor])[0]
        return is_factor,os_factor,full_factor

    def _ensure_structure_score(self,factors:list[Factor])->bool:
        missing = [factor for factor in factors if factor.result.structure_score is None]
        if len(missing)==0:
            return False
        self.marker.mark_structure(missing)
        return True

    def _full_corr_stats(self,factor:Factor)->dict:
        product_factors = self.product_repository.load_full_factors()
        return self._corr_stats(
            factor.result.rankics,
            product_factors,
            exclude_spec_id=factor.spec.spec_id,
        )

    @staticmethod
    def _clone_factor(factor:Factor)->Factor:
        spec = FactorSpec(
            formula=factor.spec.formula,
            decay=factor.spec.decay,
            neutralize=factor.spec.neutralize,
            formula_id=factor.spec.formula_id,
            spec_id=factor.spec.spec_id,
        )
        meta = FactorMeta(
            atom=factor.meta.atom,
            layer=factor.meta.layer,
            path=factor.meta.path,
            factory=factor.meta.factory,
        )
        return Factor(meta=meta,spec=spec)

    def _corr_stats(self,
                    rankics:np.ndarray,
                    product_factors:list[Factor],
                    exclude_spec_id:Union[str,None]=None)->dict:
        corrs = []
        max_corr = None
        max_corr_id = None
        for factor in product_factors:
            if factor.spec.spec_id==exclude_spec_id:
                continue
            corr = p2p_corr(rankics,factor.result.rankics)
            if np.isfinite(corr):
                corr = abs(float(corr))
                corrs.append(corr)
                if max_corr is None or corr>max_corr:
                    max_corr = corr
                    max_corr_id = factor.spec.spec_id

        return {
            "avg_corr":sum(corrs)/len(corrs) if corrs else None,
            "max_corr":max_corr,
            "max_corr_id":max_corr_id,
            "product_count":len(corrs),
        }

    @staticmethod
    def _product_conflict_matrix(factors:list[Factor],corr_threshold:float)->list[list[bool]]:
        n = len(factors)
        conflict = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1,n):
                corr = p2p_corr(factors[i].result.rankics,factors[j].result.rankics)
                if np.isfinite(corr) and abs(float(corr))>corr_threshold:
                    conflict[i][j] = True
                    conflict[j][i] = True
        return conflict

    @staticmethod
    def _is_independent(combo,conflict:list[list[bool]])->bool:
        for i,left in enumerate(combo):
            for right in combo[i+1:]:
                if conflict[left][right]:
                    return False
        return True

    def _submission_result(self,is_factor:Factor,os_factor:Factor,full_factor:Factor,corr_stats:dict)->dict:
        spec = is_factor.spec
        return {
            "spec_id":spec.spec_id,
            "formula_id":spec.formula_id,
            "formula":spec.formula,
            "decay":spec.decay,
            "neutralize":spec.neutralize,
            "is":self._metrics(is_factor),
            "os":self._metrics(os_factor),
            "full":self._metrics(full_factor),
            "corr":corr_stats,
        }

    @staticmethod
    def _metrics(factor:Factor)->dict:
        result = factor.result
        return {
            "ic":float(result.ic),
            "rankic":float(result.rankic),
            "rankicir":float(result.rankicir),
            "longret":float(result.longret),
            "turnover":float(result.turnover),
            "coverage":float(result.coverage),
        }

    @staticmethod
    def _bind_spec_id(factor:Factor):
        spec = factor.spec
        if spec.formula_id is None:
            spec.formula_id = IdManager.get_formula_id(spec.formula)
        if spec.spec_id is None:
            spec.spec_id = IdManager.get_spec_id(spec.formula,spec.decay,spec.neutralize)
