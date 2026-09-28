import json

from ..core.id_manager import IdManager


def make_job_key(job_type,left_formula_id,right_formula_id,op,params):
    data = {
        "job_type":job_type,
        "left_formula_id":left_formula_id,
        "right_formula_id":right_formula_id,
        "op":op,
        "params":list(params),
    }
    text = json.dumps(data,sort_keys=True,ensure_ascii=False)
    return IdManager.get_formula_id(text)
