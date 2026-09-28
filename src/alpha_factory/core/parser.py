import ast
from ..factory_ops import numpy_funcs
from ..factory_data.data_manager import DataManager
from .factor import FactorNode
from .node import Node


class FormulaParser:
    def __init__(self,dm:DataManager):
        self.dm = dm

    def parse(self,formula:str)->FactorNode:
        try:
            tree = ast.parse(formula,mode="eval")
            root = self._parse_node(tree.body)
            flds = []
            ops = []
            self._collect_flds(root,flds)
            self._collect_ops(root,ops)
            return FactorNode(
                root=root,
                flds=tuple(dict.fromkeys(flds)),
                ops=tuple(dict.fromkeys(ops)),
            )
        except Exception as e:
            raise ValueError(f"invalid formula: {formula}, error: {e}")

    def validate(self,formula:str):
        try:
            self.parse(formula)
            return True,None
        except Exception as e:
            return False,str(e)

    def _parse_node(self,node):
        if isinstance(node,ast.Call):
            return self._parse_call(node)
        if isinstance(node,ast.Name):
            return self._parse_fld(node)
        if isinstance(node,ast.Constant):
            return Node("param",node.value)
        raise ValueError(f"only function-style formula is supported: {ast.dump(node)}")

    def _parse_call(self,node):
        if not isinstance(node.func,ast.Name):
            raise ValueError(f"unsupported function node: {ast.dump(node.func)}")

        op = node.func.id
        if not hasattr(numpy_funcs,op):
            raise ValueError(f"unknown operator: {op}")

        args = [self._parse_node(arg) for arg in node.args]
        return Node("op",op,args)

    def _parse_fld(self,node):
        fld = node.id
        if fld not in self.dm.list_datas():
            raise ValueError(f"unknown data field: {fld}")
        return Node("fld",fld)

    def _collect_flds(self,node,flds:list[str]):
        if node.kind=="fld":
            flds.append(node.value)
            return
        for arg in node.args:
            if isinstance(arg,Node):
                self._collect_flds(arg,flds)

    def _collect_ops(self,node,ops:list[str]):
        if node.kind=="op":
            ops.append(node.value)
        for arg in node.args:
            if isinstance(arg,Node):
                self._collect_ops(arg,ops)
