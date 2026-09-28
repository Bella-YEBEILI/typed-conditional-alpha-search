import os
import sys
import importlib.util
import inspect
from types import ModuleType


VALID_FACTOR_TYPES = {"regular","super"}
VALID_FACTOR_LEVELS = {"days","minutes"}
VALID_NEUTRALIZE = {"industry","size","ram","styles","complete"}
VALID_DOMAINS = {"pv","fundamental","hybrid"}
META_REQUIRED_KEYS = {"factor_name","author","level","domain"}
REGULAR_SETTING_REQUIRED_KEYS = {"data_needed","universe","pasteurization"}
SUPER_SETTING_REQUIRED_KEYS = {"data_needed","factor_needed","universe","pasteurization"}


def _ensure_module_dir(module_dir:str)->None:
    if module_dir not in sys.path:
        sys.path.insert(0,module_dir)


def load_module(file_path:str)->ModuleType:
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"factor file not found: {abs_path}")

    module_dir = os.path.dirname(abs_path)
    module_name = os.path.splitext(os.path.basename(abs_path))[0]
    _ensure_module_dir(module_dir)

    spec = importlib.util.spec_from_file_location(module_name,abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec: {abs_path}")

    prev_module = sys.modules.get(module_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if prev_module is None:
            sys.modules.pop(module_name,None)
        else:
            sys.modules[module_name] = prev_module
        raise
    return module


def validate_module(module:ModuleType)->None:
    if not hasattr(module,"TYPE"):
        raise ValueError("factor module missing TYPE")
    if not hasattr(module,"META"):
        raise ValueError("factor module missing META")
    if not hasattr(module,"SETTING"):
        raise ValueError("factor module missing SETTING")
    if not hasattr(module,"calc_factor"):
        raise ValueError("factor module missing calc_factor")
    if not isinstance(module.TYPE,str):
        raise ValueError("TYPE must be str")
    if module.TYPE not in VALID_FACTOR_TYPES:
        raise ValueError(f"TYPE must be one of {sorted(VALID_FACTOR_TYPES)}")
    if not isinstance(module.META,dict):
        raise ValueError("META must be dict")
    if not isinstance(module.SETTING,dict):
        raise ValueError("SETTING must be dict")
    if not callable(module.calc_factor):
        raise ValueError("calc_factor is not callable")

    for key in META_REQUIRED_KEYS:
        if key not in module.META:
            raise ValueError(f"META missing required key: {key}")
    if not isinstance(module.META["factor_name"],str) or module.META["factor_name"]=="":
        raise ValueError("META.factor_name must be non-empty str")
    if not isinstance(module.META["author"],str) or module.META["author"]=="":
        raise ValueError("META.author must be non-empty str")
    if not isinstance(module.META["level"],str):
        raise ValueError("META.level must be str")
    if module.META["level"] not in VALID_FACTOR_LEVELS:
        raise ValueError(f"META.level must be one of {sorted(VALID_FACTOR_LEVELS)}")
    if str(module.META["domain"]) not in VALID_DOMAINS:
        raise ValueError(f"META.domain must be one of {sorted(VALID_DOMAINS)}")

    if module.TYPE=="regular":
        required_setting_keys = REGULAR_SETTING_REQUIRED_KEYS
    else:
        required_setting_keys = SUPER_SETTING_REQUIRED_KEYS

    for key in required_setting_keys:
        if key not in module.SETTING:
            raise ValueError(f"SETTING missing required key: {key}")

    if not isinstance(module.SETTING["data_needed"],list):
        raise ValueError("SETTING.data_needed must be list")
    if not isinstance(module.SETTING["universe"],str) or module.SETTING["universe"]=="":
        raise ValueError("SETTING.universe must be non-empty str")
    if not isinstance(module.SETTING["pasteurization"],bool):
        raise ValueError("SETTING.pasteurization must be bool")
    if module.TYPE=="super":
        if not isinstance(module.SETTING["factor_needed"],list):
            raise ValueError("SETTING.factor_needed must be list")
        if len(module.SETTING["factor_needed"])==0:
            raise ValueError("SETTING.factor_needed must be non-empty for super factor")

    # validate optional SETTING fields
    if "decay" in module.SETTING:
        if not isinstance(module.SETTING["decay"],int) or module.SETTING["decay"]<0:
            raise ValueError("SETTING.decay must be non-negative int")
    if "neutralize" in module.SETTING:
        if module.SETTING["neutralize"] is not None and module.SETTING["neutralize"] not in VALID_NEUTRALIZE:
            raise ValueError(f"SETTING.neutralize must be None or one of {sorted(VALID_NEUTRALIZE)}")

    level = module.META["level"]

    # validate calc_factor signature
    sig = inspect.signature(module.calc_factor)
    params = list(sig.parameters.values())

    if level=="minutes":
        # minutes level: must have prepare_minute_datas and calc_factor(data_ctx, minute_ctx)
        if not hasattr(module,"prepare_minute_datas"):
            raise ValueError("minutes level factor must define prepare_minute_datas()")
        if not callable(module.prepare_minute_datas):
            raise ValueError("prepare_minute_datas must be callable")
        if module.TYPE=="regular":
            if len(params)!=2:
                raise ValueError("minutes regular factor calc_factor must have exactly two parameters: data_ctx, minute_ctx")
            if params[0].name!="data_ctx" or params[1].name!="minute_ctx":
                raise ValueError('minutes regular factor calc_factor parameters must be named "data_ctx" and "minute_ctx"')
        else:
            if len(params)!=3:
                raise ValueError("minutes super factor calc_factor must have exactly three parameters: data_ctx, factor_ctx, minute_ctx")
            if params[0].name!="data_ctx" or params[1].name!="factor_ctx" or params[2].name!="minute_ctx":
                raise ValueError('minutes super factor calc_factor parameters must be named "data_ctx", "factor_ctx" and "minute_ctx"')
    else:
        # days level
        if module.TYPE=="regular":
            if len(params)!=1:
                raise ValueError("regular factor calc_factor must have exactly one parameter: data_ctx")
            if params[0].name!="data_ctx":
                raise ValueError('regular factor calc_factor parameter must be named "data_ctx"')
        else:
            if len(params)!=2:
                raise ValueError("super factor calc_factor must have exactly two parameters: data_ctx, factor_ctx")
            if params[0].name!="data_ctx" or params[1].name!="factor_ctx":
                raise ValueError('super factor calc_factor parameters must be named "data_ctx" and "factor_ctx"')


def load_factor_module(file_path:str)->ModuleType:
    module = load_module(file_path)
    validate_module(module)
    return module
