import ast
import random

from ..core.factor import Factor,make_factor
from ..core.id_manager import IdManager
from ..config.fld_registry import FLD_REGISTRY
from ..config.ops_registry import INPUT_DOMAINS
from .context import GeneticContext


class LocalMutator:
    def __init__(self,context:GeneticContext,config:dict):
        self.context = context
        mutation = config.get("mutation",{})
        self.replace_leaf_prob = mutation.get("replace_leaf",0.6)
        self.replace_window_prob = mutation.get("replace_window",0.25)
        self.replace_condition_prob = mutation.get("replace_condition",0.1)
        self.replace_group_prob = mutation.get("replace_group",0.05)
        self.same_level_b = config.get("field_match",{}).get("same_level_b",True)
        self.windows = tuple(INPUT_DOMAINS["window"])

    def mutate_batch(self,
                     population:list[Factor],
                     count:int,
                     rng:random.Random,
                     seen_formula_ids:set)->list[Factor]:
        children = []
        attempts = 0
        max_attempts = count*20
        while len(children)<count and attempts<max_attempts and len(population)>0:
            attempts += 1
            parent = rng.choice(population)
            child = self.mutate_one(parent,rng,seen_formula_ids)
            if child is not None:
                children.append(child)
        return children

    def mutate_one(self,parent:Factor,rng:random.Random,seen_formula_ids:set):
        op = self._sample_mutation(rng)
        if op=="replace_window":
            child = self.replace_window(parent,rng)
        elif op=="replace_condition":
            child = self.replace_field(parent,rng,dtype="condition")
        elif op=="replace_group":
            child = self.replace_field(parent,rng,dtype="group")
        else:
            child = self.replace_field(parent,rng,dtype="float")
        if child is None:
            return None

        formula_id = IdManager.get_formula_id(child.spec.formula)
        if formula_id in seen_formula_ids:
            return None
        ok,_ = self.context.parser.validate(child.spec.formula)
        if not ok:
            return None

        seen_formula_ids.add(formula_id)
        child.spec.formula_id = formula_id
        child.spec.spec_id = IdManager.get_spec_id(child.spec.formula,0,None)
        return child

    def replace_field(self,parent:Factor,rng:random.Random,dtype:str):
        fields = [
            fld for fld in self.context.formula_fields(parent.spec.formula)
            if FLD_REGISTRY[fld]["dtype"]==dtype
        ]
        if len(fields)==0:
            return None

        old = rng.choice(fields)
        candidates = self._field_candidates(old)
        if len(candidates)==0:
            return None

        new = rng.choice(candidates)
        tree = ast.parse(parent.spec.formula,mode="eval")
        tree = _ReplaceName(old,new).visit(tree)
        ast.fix_missing_locations(tree)
        formula = self.context.normalize_formula(ast.unparse(tree))
        return self._make_child(parent,formula,"replace_"+dtype)

    def replace_window(self,parent:Factor,rng:random.Random):
        tree = ast.parse(parent.spec.formula,mode="eval")
        values = [
            node.value for node in ast.walk(tree)
            if isinstance(node,ast.Constant) and type(node.value) is int and node.value in self.windows
        ]
        if len(values)==0:
            return None

        old = rng.choice(values)
        choices = [x for x in self.windows if x!=old]
        if len(choices)==0:
            return None
        new = rng.choice(choices)

        tree = _ReplaceWindow(old,new).visit(tree)
        ast.fix_missing_locations(tree)
        formula = self.context.normalize_formula(ast.unparse(tree))
        return self._make_child(parent,formula,"replace_window")

    def _field_candidates(self,old):
        old_meta = FLD_REGISTRY[old]
        out = []
        for fld,meta in FLD_REGISTRY.items():
            if fld==old or fld in self.context.ban_fields:
                continue
            if meta["dtype"]!=old_meta["dtype"]:
                continue
            if meta["dtype"]=="condition" and fld not in self.context.condition_pool:
                continue
            if meta["dtype"]=="group" and fld not in self.context.group_pool:
                continue
            if self.same_level_b and meta.get("level_b")!=old_meta.get("level_b"):
                continue
            if meta["dtype"]=="float" and meta.get("dim","score")!=old_meta.get("dim","score"):
                continue
            out.append(fld)
        return out

    def _sample_mutation(self,rng:random.Random):
        items = (
            ("replace_leaf",self.replace_leaf_prob),
            ("replace_window",self.replace_window_prob),
            ("replace_condition",self.replace_condition_prob),
            ("replace_group",self.replace_group_prob),
        )
        total = sum(w for _,w in items)
        x = rng.random()*total
        acc = 0.0
        for name,weight in items:
            acc += weight
            if x<=acc:
                return name
        return "replace_leaf"

    @staticmethod
    def _make_child(parent:Factor,formula:str,mutation_type:str):
        child = make_factor(
            formula,
            atom=("refine",),
            layer=None,
            path=mutation_type,
            factory=parent.meta.factory,
            decay=0,
        )
        child.refine_parent_spec_id = IdManager.get_spec_id(parent.spec.formula,0,None)
        child.refine_mutation_type = mutation_type
        return child


class _ReplaceName(ast.NodeTransformer):
    def __init__(self,old,new):
        self.old = old
        self.new = new

    def visit_Name(self,node):
        if node.id==self.old:
            return ast.copy_location(ast.Name(id=self.new,ctx=node.ctx),node)
        return node


class _ReplaceWindow(ast.NodeTransformer):
    def __init__(self,old,new):
        self.old = old
        self.new = new
        self.done = False

    def visit_Constant(self,node):
        if not self.done and type(node.value) is int and node.value==self.old:
            self.done = True
            return ast.copy_location(ast.Constant(value=self.new),node)
        return node
