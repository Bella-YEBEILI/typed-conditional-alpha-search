import numpy as np
from dataclasses import dataclass,field
from typing import Union,Literal


@dataclass
class FactorMeta:
    atom: tuple[str,...] = ()
    layer: Union[int,None] = None
    path: Union[str,None] = None
    factory: Union[Literal["math","ts","group","condition","cross"],None] = None


@dataclass
class FactorSpec:
    formula: str
    decay: int = 0
    neutralize: Union[str,None] = None
    formula_id: Union[str,None] = None
    spec_id: Union[str,None] = None


@dataclass
class FactorNode:
    root: Union[object,None] = None
    flds: tuple[str,...] = ()
    ops: tuple[str,...] = ()


@dataclass
class FactorResult:
    ic: Union[float,None] = None
    rankic: Union[float,None] = None
    rankicir: Union[float,None] = None
    longret: Union[float,None] = None
    turnover: Union[float,None] = None
    coverage: Union[float,None] = None
    rankics: np.ndarray = field(default_factory=lambda:np.array([],dtype=np.float64))
    is_candidate: Union[bool,None] = None
    is_seed: Union[bool,None] = None
    perf_score: Union[float,None] = None
    structure_score: Union[float,None] = None
    batch_corr: Union[float,None] = None
    prod_corr: Union[float,None] = None
    fitness: Union[float,None] = None
    candidate_prod_prune_passed: Union[bool,None] = None
    candidate_prod_max_corr: Union[float,None] = None
    candidate_prod_max_id: Union[str,None] = None
    candidate_self_prune_passed: Union[bool,None] = None
    candidate_self_max_corr: Union[float,None] = None
    candidate_self_max_id: Union[str,None] = None
    seed_prune_passed: Union[bool,None] = None
    seed_max_corr: Union[float,None] = None
    seed_max_id: Union[str,None] = None


@dataclass
class Factor:
    meta: FactorMeta
    spec: FactorSpec
    node: FactorNode = field(default_factory=FactorNode)
    result: FactorResult = field(default_factory=FactorResult)


def make_factor(formula:str,
                atom:tuple[str,...]=(),
                layer:Union[int,None]=None,
                path:Union[str,None]=None,
                factory:Union[str,None]=None,
                decay:int=0,
                neutralize:Union[str,None]=None)->Factor:
    return Factor(
        meta=FactorMeta(atom=atom,layer=layer,path=path,factory=factory),
        spec=FactorSpec(formula=formula,decay=decay,neutralize=neutralize),
    )


def make_factor_from_spec(spec:FactorSpec,
                          atom:tuple[str,...]=(),
                          layer:Union[int,None]=None,
                          path:Union[str,None]=None,
                          factory:Union[str,None]=None)->Factor:
    return Factor(
        meta=FactorMeta(atom=atom,layer=layer,path=path,factory=factory),
        spec=spec,
    )


def clone_factor(factor:Factor,
                 formula:str,
                 decay:int=0,
                 neutralize:Union[str,None]=None)->Factor:
    return make_factor(
        formula=formula,
        atom=factor.meta.atom,
        layer=factor.meta.layer,
        path=factor.meta.path,
        factory=factor.meta.factory,
        decay=decay,
        neutralize=neutralize,
    )
