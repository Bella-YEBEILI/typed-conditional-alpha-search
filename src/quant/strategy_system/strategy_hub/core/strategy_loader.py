import os
import importlib.util
import inspect
from types import ModuleType


VALID_STRATEGY_TYPES = {"regular","super"}
META_REQUIRED_KEYS = {"strategy_name"}
REGULAR_SETTING_REQUIRED_KEYS = {"universe","method","weight_config_name"}
SUPER_SETTING_REQUIRED_KEYS = {"strategy_needed","universe","method","weight_config_name"}

def load_module(file_path:str)->ModuleType:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"factor file not found: {file_path}")

    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name,file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def validate_module(module:ModuleType)->None:
    if not hasattr(module,"TYPE"):
        raise ValueError("strategy module missing TYPE")
    if not hasattr(module,"META"):
        raise ValueError("strategy module missing META")
    if not hasattr(module,"SETTING"):
        raise ValueError("strategy module missing SETTING")
    if not hasattr(module,"run"):
        raise ValueError("strategy module missing run")
    if not isinstance(module.TYPE,str):
        raise ValueError("TYPE must be str")
    if module.TYPE not in VALID_STRATEGY_TYPES:
        raise ValueError(f"TYPE must be one of {sorted(VALID_STRATEGY_TYPES)}")
    if not isinstance(module.META,dict):
        raise ValueError("META must be dict")
    if not isinstance(module.SETTING,dict):
        raise ValueError("SETTING must be dict")
    if not callable(module.run):
        raise ValueError("run is not callable")

    for key in META_REQUIRED_KEYS:
        if key not in module.META:
            raise ValueError(f"META missing required key: {key}")
    if not isinstance(module.META["strategy_name"],str) or module.META["strategy_name"]=="":
        raise ValueError("META.strategy_name must be non-empty str")

    if module.TYPE=="regular":
        required_setting_keys = REGULAR_SETTING_REQUIRED_KEYS
    else:
        required_setting_keys = SUPER_SETTING_REQUIRED_KEYS

    for key in required_setting_keys:
        if key not in module.SETTING:
            raise ValueError(f"SETTING missing required key: {key}")
    
    sig = inspect.signature(module.run)
    params = list(sig.parameters.values())
    if len(params)!=2:
        raise ValueError("strategy run must have exactly two parameters: start, end")
    if params[0].name!="start" or params[1].name!="end":
        raise ValueError('strategy run parameters must be named "start" and "end"')

def load_strategy_module(file_path:str)->ModuleType:
    module = load_module(file_path)
    validate_module(module)
    return module