from ..factory_ops import numpy_funcs
from ..factory_data.data_manager import DataManager


class ValueEngine:
    def __init__(self,dm:DataManager,cache_flds=True):
        self.dm = dm
        self.cache_flds = cache_flds
        self.fld_cache = {}
        self.value_cache = {}

    def calc(self,node):
        memo = {}
        return self._calc(node,memo)

    def _calc(self,node,memo):
        key = self._node_key(node)
        if key in memo:
            return memo[key]
        if key in self.value_cache:
            value = self.value_cache[key]
            memo[key] = value
            return value

        if node.kind=="fld":
            value = self._get_fld(node.value)
        elif node.kind=="param":
            value = node.value
        elif node.kind=="op":
            args = [self._calc(arg,memo) for arg in node.args]
            value = getattr(numpy_funcs,node.value)(*args)
        else:
            raise ValueError(f"unknown node kind: {node.kind}")

        memo[key] = value
        return value

    def cache_value(self,node):
        value = self.calc(node)
        self.value_cache[self._node_key(node)] = value
        return value

    def clear_value_cache(self):
        self.value_cache.clear()

    def _get_fld(self,fld):
        if self.cache_flds and fld in self.fld_cache:
            return self.fld_cache[fld]

        value = self.dm.get_data(fld)
        if self.cache_flds:
            self.fld_cache[fld] = value
        return value

    def clear_fld_cache(self):
        self.fld_cache.clear()

    def _node_key(self,node):
        if node.kind=="fld":
            return ("fld",node.value)
        if node.kind=="param":
            return ("param",node.value)
        if node.kind=="op":
            return ("op",node.value,tuple(self._node_key(arg) for arg in node.args))
        raise ValueError(f"unknown node kind: {node.kind}")
