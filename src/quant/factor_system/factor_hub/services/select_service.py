import pandas as pd
import numpy as np

from ..core.factor_result_profiles import DEFAULT_PROFILE_ID
from ..core.factor_performance_engine import DEFAULT_PARAMS

from ..components.lib_interactor import LibInteractor
from ..components.factor_comparator import FactorComparator


class SelectService:
    def __init__(self,
                 interactor:LibInteractor,
                 comparator:FactorComparator):
        self.interactor = interactor
        self.comparator = comparator

    # ----------------------------------------------------------
    # operations
    # ----------------------------------------------------------
    def list_selected(self)->list[str]:
        return self.interactor.list_selected()

    def select_add(self,factor_name:str):
        self.interactor.select_add(factor_name)

    def select_remove(self,factor_name:str):
        self.interactor.select_remove(factor_name)

    def set(self,factor_names:list[str])->None:
        self.interactor.select_set(factor_names)

