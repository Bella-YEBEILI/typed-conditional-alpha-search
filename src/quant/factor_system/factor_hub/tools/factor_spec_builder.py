"""
FactorSpecBuilder用于把研究端传入的因子描述转换为内部spec。

职责：
1. 校验dict输入协议是否完整且字段类型是否合法。
2. 解析formula中出现的名称和算子。
3. 基于DataProvider的数据字段列表识别data_needed。
4. 基于registry中已注册的因子名识别super因子的factor_needed。
5. 支持从原始spec dict构建标准spec，也支持从formula直接补全默认spec。

规则：
- regular和super都要求提供formula。
- regular的data_needed从formula自动解析。
- super的data_needed和factor_needed从formula自动解析。
- 从formula直接构建时，会自动判断类型，并补上默认author、level、tag、universe和pasteurization。
- 如果formula中出现未知名称，或使用了不支持的算子，会直接报错。

这个组件只负责spec构建，不负责渲染为.py文件，也不负责执行回测。
"""


import ast
import inspect

from quant.quant_lib import analysis

VALID_TYPES = {"regular","super"}
VALID_LEVELS = {"days","minutes"}

class FactorSpecBuilder:
    def __init__(self,dm,registry):
        self.dm = dm
        self.registry = registry
        self.data_fields = set(dm.list_datas())
        self.operator_names = self._get_operator_names()
        self.factor_names = set(registry.list_factors().keys())

    def _get_operator_names(self)->set[str]:
        names = set()
        for name in dir(analysis):
            if name.startswith("_"):
                continue
            value = getattr(analysis,name)
            if inspect.isbuiltin(value) or inspect.isfunction(value):
                names.add(name)
        return names

    def _validate_base(self,spec:dict)->None:
        if not isinstance(spec,dict):
            raise ValueError("spec must be dict")
        factor_type = spec.get("type")
        if factor_type not in VALID_TYPES:
            raise ValueError(f"type must be one of {sorted(VALID_TYPES)}")
        if not isinstance(spec["factor_name"],str) or spec["factor_name"]=="":
            raise ValueError("factor_name must be non-empty str")
        if not isinstance(spec["author"],str) or spec["author"]=="":
            raise ValueError("author must be non-empty str")
        if not isinstance(spec["level"],str) or spec["level"] not in VALID_LEVELS:
            raise ValueError(f"level must be one of {sorted(VALID_LEVELS)}")
        if not isinstance(spec["universe"],str) or spec["universe"]=="":
            raise ValueError("universe must be non-empty str")
        if not isinstance(spec["pasteurization"],bool):
            raise ValueError("pasteurization must be bool")
        if not isinstance(spec["formula"],str) or spec["formula"]=="":
            raise ValueError("formula must be non-empty str")
        if "tag" in spec and not isinstance(spec["tag"],str):
            raise ValueError("tag must be str")
        if "category" in spec and not isinstance(spec["category"],str):
            raise ValueError("category must be str")

    def _collect_names(self,expr:ast.AST)->set[str]:
        names = set()
        for node in ast.walk(expr):
            if isinstance(node,ast.Name):
                names.add(node.id)
        return names

    def _collect_call_names(self,expr:ast.AST)->set[str]:
        names = set()
        for node in ast.walk(expr):
            if not isinstance(node,ast.Call):
                continue
            if isinstance(node.func,ast.Name):
                names.add(node.func.id)
        return names

    def collect_formula_names(self,formula:str)->set[str]:
        try:
            expr = ast.parse(formula,mode="eval")
        except SyntaxError as e:
            raise ValueError(f"invalid formula: {e}") from e
        return self._collect_names(expr)

    def infer_type_from_formula(self,formula:str)->str:
        names = self.collect_formula_names(formula)
        for name in names:
            if name in self.factor_names:
                return "super"
        return "regular"

    def build_from_formula(self,formula:str)->dict:
        if not isinstance(formula,str) or formula=="":
            raise ValueError("formula must be non-empty str")
        spec = {
            "type":self.infer_type_from_formula(formula),
            "factor_name":formula,
            "author":"quant",
            "level":"days",
            "domain":"pv",
            "tag":"",
            "category":"unknown",
            "universe":"standards",
            "pasteurization":False,
            "formula":formula,
        }
        return self.build_from_dict(spec)

    def _check_regular_formula(self,all_names:set[str],operator_names:set[str])->list[str]:
        invalid = []
        for name in sorted(all_names):
            if name in self.data_fields:
                continue
            if name in operator_names:
                continue
            invalid.append(name)
        if len(invalid)>0:
            raise ValueError(f"regular formula contains unknown names: {invalid}")
        return sorted(name for name in all_names if name in self.data_fields)

    def _check_super_formula(self,all_names:set[str],operator_names:set[str])->tuple[list[str],list[str]]:
        factor_needed = []
        invalid = []
        for name in sorted(all_names):
            if name in self.data_fields:
                continue
            if name in self.factor_names:
                factor_needed.append(name)
                continue
            if name in operator_names:
                continue
            invalid.append(name)
        if len(invalid)>0:
            raise ValueError(f"super formula contains unknown names: {invalid}")
        data_needed = sorted(name for name in all_names if name in self.data_fields)
        return data_needed,factor_needed

    def build_from_dict(self,spec:dict)->dict:
        if not isinstance(spec,dict):
            raise ValueError("spec must be dict")
        if "formula" not in spec:
            raise ValueError("spec must contain 'formula'")
        formula = spec["formula"]
        spec.setdefault("type",self.infer_type_from_formula(formula))
        spec.setdefault("factor_name",formula)
        spec.setdefault("author","unknown")
        spec.setdefault("level","days")
        spec.setdefault("domain","pv")
        spec.setdefault("tag","")
        spec.setdefault("category","unknown")
        spec.setdefault("universe","standards")
        spec.setdefault("pasteurization",False)
        spec.setdefault("decay",0)
        spec.setdefault("neutralize",None)
        self._validate_base(spec)

        formula = spec["formula"]
        try:
            expr = ast.parse(formula,mode="eval")
        except SyntaxError as e:
            raise ValueError(f"invalid formula: {e}") from e
        all_names = self._collect_names(expr)
        operator_names = self._collect_call_names(expr)
        invalid_operators = sorted(name for name in operator_names if name not in self.operator_names)
        if len(invalid_operators)>0:
            raise ValueError(f"formula contains unsupported operators: {invalid_operators}")

        if spec["type"]=="regular":
            data_needed = self._check_regular_formula(all_names,operator_names)
            factor_needed = []
        else:
            data_needed,factor_needed = self._check_super_formula(all_names,operator_names)

        return {
            "type":spec["type"],
            "factor_name":spec["factor_name"],
            "author":spec["author"],
            "level":spec["level"],
            "domain":spec.get("domain","pv"),
            "tag":spec.get("tag",""),
            "category":spec.get("category","unknown"),
            "universe":spec["universe"],
            "pasteurization":spec["pasteurization"],
            "formula":formula,
            "data_needed":data_needed,
            "factor_needed":factor_needed,
            "operators_used":sorted(name for name in operator_names if name in self.operator_names),
            "decay":spec.get("decay",0),
            "neutralize":spec.get("neutralize",None),
        }
