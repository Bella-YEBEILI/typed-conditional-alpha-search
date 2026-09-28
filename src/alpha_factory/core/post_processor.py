from ..factory_ops.numpy_funcs import ts_decay_linear,mask
from ..factory_data.data_manager import DataManager


class PostProcessor:
    def __init__(self,dm:DataManager):
        self.dm = dm
        self.standards = None

    def process(self,value,decay=0,neutralize=None):
        value = self._apply_decay(value,decay)
        value = self._apply_neutralize(value,neutralize)
        value = self._apply_standards(value)
        return value

    def _apply_decay(self,value,decay):
        if decay is None or decay==0:
            return value
        return ts_decay_linear(value,decay)

    def _apply_neutralize(self,value,neutralize):
        if neutralize is None:
            return value
        raise NotImplementedError("neutralize is not implemented yet")

    def _apply_standards(self,value):
        standards = self._get_standards()
        return mask(value,standards)

    def _get_standards(self):
        if self.standards is None:
            self.standards = self.dm.get_data("standards")
        return self.standards

    def clear_cache(self):
        self.standards = None
