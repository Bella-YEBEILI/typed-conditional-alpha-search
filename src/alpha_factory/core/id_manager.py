import hashlib
import json


class IdManager:
    @staticmethod
    def stable_hash(obj,n=16):
        text = json.dumps(obj,ensure_ascii=False,sort_keys=True,default=str)
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]

    @staticmethod
    def get_formula_id(formula:str):
        return IdManager.stable_hash({
            "formula":formula
        })

    @staticmethod
    def get_spec_id(formula:str,decay:int=0,neutralize=None):
        formula_id = IdManager.get_formula_id(formula)
        return IdManager.stable_hash({
            "formula_id":formula_id,
            "decay":int(decay),
            "neutralize":neutralize
        })
