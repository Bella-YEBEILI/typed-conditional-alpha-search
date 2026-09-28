import ast
import random

from ..config.fld_registry import FLD_REGISTRY
from ..config.ops_registry import OPS_REGISTRY,INPUT_DOMAINS,FLOAT_DIMS


FLOAT_OUTPUTS = ("score","rank","zscore")
FIELD_INPUTS = ("float","score","rank","zscore","condition","group")


class GeneticContext:
    def __init__(self,parser,config:dict):
        self.parser = parser
        self.config = config
        self.ban_fields = set(config.get("field_filter",{}).get("ban_fields",()))
        self.banned_ops = set(config.get("banned_ops",("math_self","ts_self","group_self")))
        self.group_merge_prob = config.get("group_merge_prob",0.2)
        self._build_field_pools()
        self.ops = {
            op:meta
            for op,meta in OPS_REGISTRY.items()
            if op not in self.banned_ops and self._is_float_output(meta)
        }

    def _build_field_pools(self):
        self.float_pool = []
        self.score_pool = []
        self.rank_pool = []
        self.zscore_pool = []
        condition_pool = []
        group_pool = []

        for fld,meta in FLD_REGISTRY.items():
            if fld in self.ban_fields:
                continue
            dtype = meta["dtype"]
            if dtype=="float":
                dim = meta.get("dim","score")
                self.float_pool.append(fld)
                if dim=="score":
                    self.score_pool.append(fld)
                elif dim=="rank":
                    self.rank_pool.append(fld)
                elif dim=="zscore":
                    self.zscore_pool.append(fld)
            elif dtype=="condition":
                condition_pool.append(fld)
            elif dtype=="group":
                group_pool.append(fld)

        self.condition_pool = self._config_pool("condition_pool",condition_pool)
        self.group_pool = self._config_pool("group_pool",group_pool)

    def _config_pool(self,name,default):
        pool = self.config.get(name)
        if pool is None:
            return tuple(default)
        return tuple(fld for fld in pool if fld not in self.ban_fields)

    def usable_wrap_choices(self,parent_formula:str,population):
        parent_dim = self.formula_dim(parent_formula)
        available_dims = self.population_dims(population)
        choices = []
        for op,meta in self.ops.items():
            for sig in self.signatures(meta):
                for i,slot in enumerate(sig):
                    if self.compatible(parent_dim,slot) and self.can_fill_signature(sig,i,available_dims):
                        choices.append((op,meta,sig,i))
        return choices

    def can_fill_signature(self,sig,parent_pos,available_dims:set):
        for i,slot in enumerate(sig):
            if i!=parent_pos and not self.can_sample_slot(slot,available_dims):
                return False
        return True

    def can_sample_slot(self,slot,available_dims:set):
        if slot in INPUT_DOMAINS:
            return len(INPUT_DOMAINS[slot])>0
        if slot=="group":
            return len(self.group_pool)>0
        if slot=="condition":
            return len(self.condition_pool)>0
        if slot in ("float","score","rank","zscore"):
            return self.dim_available(slot,available_dims) or len(self.field_pool(slot))>0
        return False

    def population_dims(self,population):
        dims = set()
        for factor in population:
            try:
                dims.add(self.formula_dim(factor.spec.formula))
            except Exception:
                pass
        return dims

    def dim_available(self,slot,available_dims:set):
        if slot=="float":
            return any(dim in FLOAT_DIMS for dim in available_dims)
        return slot in available_dims

    def signatures(self,meta:dict):
        inputs = meta["inputs"]
        if isinstance(inputs,list):
            return inputs
        return (inputs,)

    def formula_dim(self,formula:str):
        tree = ast.parse(formula,mode="eval")
        return self._node_dim(tree.body)

    def _node_dim(self,node):
        if isinstance(node,ast.Name):
            meta = FLD_REGISTRY[node.id]
            if meta["dtype"]=="float":
                return meta.get("dim","score")
            return meta["dtype"]
        if isinstance(node,ast.Call):
            op = node.func.id
            output = OPS_REGISTRY[op]["output"]
            if output=="same":
                return self._node_dim(node.args[0])
            return output
        return "param"

    def compatible(self,actual,slot):
        if slot=="float":
            return actual in FLOAT_DIMS
        return actual==slot

    def sample_slot(self,slot,parent_formula,rng:random.Random,population):
        if slot in INPUT_DOMAINS:
            return rng.choice(INPUT_DOMAINS[slot])
        if slot=="group":
            return self.sample_group(parent_formula,rng)
        if slot=="condition":
            return rng.choice(self.condition_pool)
        if slot in ("float","score","rank","zscore"):
            return self.sample_float(slot,rng,population)
        raise ValueError(f"unknown input slot: {slot}")

    def sample_float(self,slot,rng:random.Random,population):
        candidates = []
        for factor in population:
            try:
                dim = self.formula_dim(factor.spec.formula)
            except Exception:
                continue
            if self.compatible(dim,slot):
                candidates.append(factor.spec.formula)

        pool = self.field_pool(slot)
        if len(candidates)>0 and (len(pool)==0 or rng.random()<0.5):
            return rng.choice(candidates)
        if len(pool)>0:
            return rng.choice(pool)
        raise ValueError(f"empty pool for slot: {slot}")

    def field_pool(self,slot):
        if slot=="float":
            return self.float_pool
        if slot=="score":
            return self.score_pool
        if slot=="rank":
            return self.rank_pool
        if slot=="zscore":
            return self.zscore_pool
        return ()

    def sample_group(self,parent_formula,rng:random.Random):
        base = rng.choice(self.group_pool)
        old_groups = [
            fld for fld in self.formula_fields(parent_formula)
            if FLD_REGISTRY.get(fld,{}).get("dtype")=="group"
        ]
        if len(old_groups)>0 and rng.random()<self.group_merge_prob:
            old = rng.choice(old_groups)
            if old!=base:
                return f"group_merge({old},{base})"
        return base

    def replacement_for_field(self,fld,rng:random.Random):
        meta = FLD_REGISTRY[fld]
        dtype = meta["dtype"]
        if dtype=="condition":
            pool = [x for x in self.condition_pool if x!=fld]
            return rng.choice(pool) if pool else fld
        if dtype=="group":
            pool = [x for x in self.group_pool if x!=fld]
            if pool and rng.random()<self.group_merge_prob:
                return f"group_merge({fld},{rng.choice(pool)})"
            return rng.choice(pool) if pool else fld

        dim = meta.get("dim","score")
        if dim=="rank":
            pool = [x for x in self.rank_pool if x!=fld]
        elif dim=="zscore":
            pool = [x for x in self.zscore_pool if x!=fld]
        else:
            pool = [x for x in self.score_pool if x!=fld]
        return rng.choice(pool) if pool else fld

    def formula_fields(self,formula):
        tree = ast.parse(formula,mode="eval")
        return [
            node.id
            for node in ast.walk(tree)
            if isinstance(node,ast.Name) and node.id in FLD_REGISTRY
        ]

    @staticmethod
    def format_formula(op,args):
        return f"{op}("+",".join(GeneticContext.format_arg(arg) for arg in args)+")"

    @staticmethod
    def format_arg(arg):
        if isinstance(arg,str):
            if arg in FLD_REGISTRY or "(" in arg:
                return arg
            return repr(arg)
        return repr(arg)

    @staticmethod
    def normalize_formula(formula):
        return formula.replace(", ",",")

    @staticmethod
    def _is_float_output(meta):
        output = meta["output"]
        return output in FLOAT_OUTPUTS or output=="same"
