import numpy as np

from .config import CRITERIA,DEFAULT_CRITERIA


class FactorCriteria:
    def get_criteria(self,universe:str,domain:str)->dict:
        return CRITERIA.get((universe,domain),DEFAULT_CRITERIA)

    def _check_section(self,perf:dict,rules:dict)->dict:
        details = {}
        passed = True
        for field,rule in rules.items():
            value = perf.get(field,np.nan)
            if isinstance(value,float) and np.isnan(value):
                field_passed = False
            else:
                field_passed = True
                if "min" in rule and value<rule["min"]:
                    field_passed = False
                if "max" in rule and value>rule["max"]:
                    field_passed = False
            details[field] = {"value":value,**rule,"passed":field_passed}
            if not field_passed:
                passed = False
        return {"passed":passed,"details":details}

    def check(self,perfs:dict[str,dict],universe:str,domain:str)->dict:
        criteria = self.get_criteria(universe,domain)
        all_passed = True
        all_details = {}
        for section,rules in criteria.items():
            if section not in perfs:
                continue
            result = self._check_section(perfs[section],rules)
            all_details[section] = result["details"]
            if not result["passed"]:
                all_passed = False
        return {"passed":all_passed,"details":all_details}
