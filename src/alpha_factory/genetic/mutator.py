import ast
import random

from ..core.factor import Factor,make_factor
from ..core.id_manager import IdManager
from ..config.fld_registry import FLD_REGISTRY
from .context import GeneticContext


class GeneticMutator:
    def __init__(self,context:GeneticContext,config:dict):
        self.context = context
        mutation = config.get("mutation",{})
        self.wrap_prob = mutation.get("wrap_op",0.7)
        self.replace_leaf_prob = mutation.get("replace_leaf",0.3)

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
            child = self.mutate_one(parent,population,rng,seen_formula_ids)
            if child is not None:
                children.append(child)
        return children

    def mutate_one(self,parent:Factor,population:list[Factor],rng:random.Random,seen_formula_ids:set):
        if rng.random()<self.wrap_prob/(self.wrap_prob+self.replace_leaf_prob):
            child = self.wrap_op(parent,population,rng)
        else:
            child = self.replace_leaf(parent,rng)
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

    def wrap_op(self,parent:Factor,population:list[Factor],rng:random.Random):
        choices = self.context.usable_wrap_choices(parent.spec.formula,population)
        if len(choices)==0:
            return None

        op,meta,sig,parent_pos = rng.choice(choices)
        args = []
        for i,slot in enumerate(sig):
            if i==parent_pos:
                args.append(parent.spec.formula)
            else:
                args.append(self.context.sample_slot(slot,parent.spec.formula,rng,population))

        formula = self.context.format_formula(op,args)
        return self._make_child(parent,formula,"wrap_op",op)

    def replace_leaf(self,parent:Factor,rng:random.Random):
        fields = [
            fld for fld in self.context.formula_fields(parent.spec.formula)
            if fld in FLD_REGISTRY
        ]
        if len(fields)==0:
            return None

        old = rng.choice(fields)
        new = self.context.replacement_for_field(old,rng)
        if new==old:
            return None

        tree = ast.parse(parent.spec.formula,mode="eval")
        tree = _ReplaceName(old,new).visit(tree)
        ast.fix_missing_locations(tree)
        formula = self.context.normalize_formula(ast.unparse(tree))
        return self._make_child(parent,formula,"replace_leaf",parent.meta.factory)

    @staticmethod
    def _make_child(parent:Factor,formula:str,mutation_type:str,factory):
        child = make_factor(
            formula,
            atom=("genetic",),
            layer=None,
            path=mutation_type,
            factory=factory,
            decay=0,
        )
        child.genetic_parent_spec_id = IdManager.get_spec_id(parent.spec.formula,0,None)
        child.genetic_mutation_type = mutation_type
        return child


class _ReplaceName(ast.NodeTransformer):
    def __init__(self,old,new):
        self.old = old
        self.new = new

    def visit_Name(self,node):
        if node.id!=self.old:
            return node
        if isinstance(self.new,str) and "(" in self.new:
            return ast.parse(self.new,mode="eval").body
        return ast.copy_location(ast.Name(id=self.new,ctx=node.ctx),node)
