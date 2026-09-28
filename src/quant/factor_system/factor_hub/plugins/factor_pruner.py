import os
import re
import shutil
import numpy as np
import pandas as pd

from ..core.factor_loader import load_factor_module
from ..core.factor_result_profiles import DEFAULT_PROFILE_ID
from ..core.factor_performance_engine import DEFAULT_PARAMS
from ..core.factor_criteria import FactorCriteria
from ..core.factor_scorer import FactorScorer
from ..services.research_service import ResearchService


class FactorPruner:
    def __init__(self,
                 research:ResearchService,
                 criteria:FactorCriteria,
                 scorer:FactorScorer):
        self.research = research
        self.criteria = criteria
        self.scorer = scorer

    def _resolve_universe_domain(self,target,universe:str|None,domain:str|None)->tuple[str,str]:
        """Resolve universe and domain from target or defaults."""
        if universe is not None and domain is not None:
            return universe,domain

        # Try to load from py file
        if isinstance(target,str) and os.path.isdir(target):
            py_files = [f for f in os.listdir(target) if f.endswith(".py")]
            if py_files:
                try:
                    m = load_factor_module(os.path.join(target,py_files[0]))
                    if universe is None:
                        universe = m.SETTING.get("universe","standards")
                    if domain is None:
                        domain = str(m.META.get("domain","pv"))
                    return universe,domain
                except Exception:
                    pass

        return universe or "standards",domain or "pv"

    def _extract_section_perfs(self,perf_df:pd.DataFrame,factor_id:str)->dict[str,dict]:
        """Extract per-section perf dicts from MultiIndex perf_df for one factor."""
        sections = {}
        for tk in perf_df.index.get_level_values("transform_key").unique():
            try:
                row = perf_df.loc[(factor_id,tk),:]
                key = "raw" if tk=="raw" else tk
                sections[key] = row.to_dict()
            except KeyError:
                pass
        return sections

    def _section_passed(self,check_result:dict,criteria:dict,section:str)->bool:
        if section not in criteria:
            return True
        details = check_result.get("details",{}).get(section,{})
        if not details:
            return False
        return all(item.get("passed",False) for item in details.values())

    def _get_raw_perf(self,perf_df:pd.DataFrame,factor_ids:list[str])->pd.DataFrame:
        if not factor_ids:
            return pd.DataFrame()
        return perf_df.loc[pd.IndexSlice[factor_ids,"raw"],:].droplevel("transform_key")

    def _score_factors(self,raw_perf:pd.DataFrame,factor_ids:list[str],universe:str,domain:str)->pd.Series:
        if not factor_ids:
            return pd.Series(dtype=float)
        scores = {}
        for fid in factor_ids:
            perf = raw_perf.loc[fid].to_dict()
            scores[fid] = self.scorer.score(perf,universe,domain)
        return pd.Series(scores).sort_values(ascending=False)

    def _is_too_corr(self,fid:str,selected_ids:list[str],corr_df:pd.DataFrame,max_corr:float)->bool:
        for selected_id in selected_ids:
            if selected_id in corr_df.columns and fid in corr_df.index:
                corr = corr_df.loc[fid,selected_id]
                if not np.isnan(corr) and abs(corr)>max_corr:
                    return True
        return False

    def _prune_ordered_ids(self,
                           ordered_ids:list[str],
                           score_series:pd.Series,
                           corr_df:pd.DataFrame,
                           max_corr:float,
                           base_ids:list[str]|None=None)->tuple[list[str],list[str]]:
        accepted = []
        redundant = []
        selected_ids = list(base_ids or [])
        for fid in ordered_ids:
            score = score_series[fid]
            if np.isnan(score):
                redundant.append(fid)
                continue
            if self._is_too_corr(fid,selected_ids,corr_df,max_corr):
                redundant.append(fid)
                continue
            accepted.append(fid)
            selected_ids.append(fid)
        return accepted,redundant

    def _build_perf_df(self,raw_perf:pd.DataFrame,score_series:pd.Series,factor_ids:list[str])->pd.DataFrame:
        if not factor_ids:
            return pd.DataFrame()
        perf_df = raw_perf.loc[raw_perf.index.isin(factor_ids)].copy()
        perf_df["score"] = score_series.loc[perf_df.index]
        return perf_df.sort_values("score",ascending=False)

    def _collect_factor_files(self,target)->dict[str,str]:
        file_paths = []
        if isinstance(target,str) and os.path.isdir(target):
            file_paths = [os.path.join(target,f) for f in os.listdir(target) if f.endswith(".py")]
        elif isinstance(target,str) and os.path.isfile(target) and target.endswith(".py"):
            file_paths = [target]
        factor_files = {}
        for file_path in sorted(file_paths):
            module = load_factor_module(file_path)
            factor_files[module.META["factor_name"]] = file_path
        return factor_files

    def _replace_tag(self,text:str,tag:str)->str:
        return re.sub(r'("tag"\s*:\s*)"[^"]*"',rf'\1"{tag}"',text,count=1)

    def _export_factor_files(self,target,save_dir:str,passed:list[str],style_sensitive:list[str]):
        os.makedirs(save_dir,exist_ok=True)
        factor_files = self._collect_factor_files(target)
        for fid in passed:
            src_path = factor_files[fid]
            dst_path = os.path.join(save_dir,os.path.basename(src_path))
            shutil.copy2(src_path,dst_path)
        for fid in style_sensitive:
            src_path = factor_files[fid]
            dst_path = os.path.join(save_dir,os.path.basename(src_path))
            with open(src_path,"r",encoding="utf-8") as f:
                text = f.read()
            text = self._replace_tag(text,"smart_styles")
            with open(dst_path,"w",encoding="utf-8",newline="\n") as f:
                f.write(text)

    def prune(self,
              target,
              universe:str|None=None,
              domain:str|None=None,
              max_corr:float=0.8,
              profile_id:str=DEFAULT_PROFILE_ID,
              params:dict=DEFAULT_PARAMS,
              save_dir:str|None=None)->dict:
        universe,domain = self._resolve_universe_domain(target,universe,domain)

        transform_spec = {"subuniverse":"zz1000s","neutralize":"complete"}

        result = self.research.evaluate_batch(
            target,
            profile_id=profile_id,
            params=params,
            self_corr=True,
            transform_spec=transform_spec,
        )
        perf_df,corr_df = result

        if perf_df.empty:
            if save_dir is not None:
                os.makedirs(save_dir,exist_ok=True)
            return {
                "rejected":[],
                "redundant":[],
                "passed":[],
                "style_sensitive":[],
                "all_perf_df":perf_df,
                "passed_perf_df":pd.DataFrame(),
                "style_sensitive_perf_df":pd.DataFrame(),
                "corr_df":corr_df,
            }

        factor_ids = perf_df.index.get_level_values("id").unique().tolist()
        criteria = self.criteria.get_criteria(universe,domain)

        passed_pool_ids = []
        style_sensitive_pool_ids = []
        rejected_ids = []
        for fid in factor_ids:
            section_perfs = self._extract_section_perfs(perf_df,fid)
            check = self.criteria.check(section_perfs,universe,domain)
            raw_passed = self._section_passed(check,criteria,"raw")
            zz1000s_passed = self._section_passed(check,criteria,"zz1000s")
            complete_passed = self._section_passed(check,criteria,"complete")
            if raw_passed and zz1000s_passed and complete_passed:
                passed_pool_ids.append(fid)
            elif raw_passed and zz1000s_passed and not complete_passed:
                style_sensitive_pool_ids.append(fid)
            else:
                rejected_ids.append(fid)

        candidate_ids = passed_pool_ids+style_sensitive_pool_ids
        raw_perf = self._get_raw_perf(perf_df,candidate_ids)
        score_series = self._score_factors(raw_perf,candidate_ids,universe,domain)

        passed_pool_set = set(passed_pool_ids)
        style_sensitive_pool_set = set(style_sensitive_pool_ids)
        passed_ordered_ids = [fid for fid in score_series.index if fid in passed_pool_set]
        style_sensitive_ordered_ids = [fid for fid in score_series.index if fid in style_sensitive_pool_set]

        passed,redundant = self._prune_ordered_ids(passed_ordered_ids,score_series,corr_df,max_corr)
        style_sensitive,style_sensitive_redundant = self._prune_ordered_ids(
            style_sensitive_ordered_ids,
            score_series,
            corr_df,
            max_corr,
            base_ids=passed,
        )
        redundant.extend(style_sensitive_redundant)

        final_ids = [fid for fid in passed+style_sensitive if fid in corr_df.index and fid in corr_df.columns]
        final_corr_df = corr_df.loc[final_ids,final_ids].copy() if final_ids else pd.DataFrame()
        passed_perf_df = self._build_perf_df(raw_perf,score_series,passed)
        style_sensitive_perf_df = self._build_perf_df(raw_perf,score_series,style_sensitive)

        if save_dir is not None:
            self._export_factor_files(target,save_dir,passed,style_sensitive)

        return {
            "rejected":rejected_ids,
            "redundant":redundant,
            "passed":passed,
            "style_sensitive":style_sensitive,
            "all_perf_df":perf_df,
            "passed_perf_df":passed_perf_df,
            "style_sensitive_perf_df":style_sensitive_perf_df,
            "corr_df":final_corr_df,
        }
