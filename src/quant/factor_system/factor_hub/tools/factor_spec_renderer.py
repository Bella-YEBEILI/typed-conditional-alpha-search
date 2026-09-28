"""
FactorSpecRenderer用于把标准化后的spec渲染为可执行的因子.py文件。

职责：
1. 校验spec是否具备渲染所需的关键字段。
2. 按既定格式生成TYPE、META、SETTING和calc_factor代码。
3. 根据regular或super自动生成不同的函数签名和上下文取值代码。
4. 支持把渲染结果输出为源码字符串、指定文件或临时文件。

规则：
- 输入应当是已经过FactorSpecBuilder标准化的spec。
- regular使用calc_factor(data_ctx:dict)->pd.DataFrame。
- super使用calc_factor(data_ctx:dict,factor_ctx:dict)->pd.DataFrame。
- data_needed和factor_needed会展开成局部变量绑定，再直接return formula。

这个组件只负责spec到.py文件的渲染，不负责校验业务依赖，也不负责执行回测。
"""

import os
import tempfile


class FactorSpecRenderer:
    def _validate_spec(self,spec:dict)->None:
        if not isinstance(spec,dict):
            raise ValueError("spec must be dict")
        required_keys = {
            "type",
            "factor_name",
            "author",
            "level",
            "tag",
            "universe",
            "pasteurization",
            "formula",
            "data_needed",
            "factor_needed",
        }
        for key in required_keys:
            if key not in spec:
                raise ValueError(f"spec missing required key: {key}")
        if spec["type"] not in {"regular","super"}:
            raise ValueError("spec.type must be regular or super")
        if not isinstance(spec["data_needed"],list):
            raise ValueError("spec.data_needed must be list")
        if not isinstance(spec["factor_needed"],list):
            raise ValueError("spec.factor_needed must be list")

    def _render_scalar(self,value)->str:
        return repr(value)

    def _render_list(self,values:list)->list[str]:
        if len(values)==0:
            return ["[]"]
        lines = ["["]
        for value in values:
            lines.append(f"    {self._render_scalar(value)},")
        lines.append("]")
        return lines

    def _render_dict_block(self,name:str,data:dict,key_order:list[str])->list[str]:
        lines = [f"{name} = "+"{"]
        for key in key_order:
            value = data[key]
            if isinstance(value,list):
                list_lines = self._render_list(value)
                if len(list_lines)==1:
                    lines.append(f'    "{key}":{list_lines[0]},')
                else:
                    lines.append(f'    "{key}":{list_lines[0]}')
                    for line in list_lines[1:-1]:
                        lines.append(line)
                    lines.append(f"{list_lines[-1]},")
            else:
                lines.append(f'    "{key}":{self._render_scalar(value)},')
        lines.append("}")
        return lines

    def _render_calc_factor(self,spec:dict)->list[str]:
        factor_type = spec["type"]
        lines = []
        if factor_type=="regular":
            lines.append("def calc_factor(data_ctx:dict)->pd.DataFrame:")
        else:
            lines.append("def calc_factor(data_ctx:dict,factor_ctx:dict)->pd.DataFrame:")

        for field in spec["data_needed"]:
            lines.append(f'    {field} = data_ctx["{field}"]')
        if factor_type=="super":
            for factor_name in spec["factor_needed"]:
                lines.append(f'    {factor_name} = factor_ctx["{factor_name}"]')
        if len(lines)>1:
            lines.append("")
        lines.append(f'    return {spec["formula"]}')
        return lines

    def render_code(self,spec:dict)->str:
        self._validate_spec(spec)

        meta = {
            "factor_name":spec["factor_name"],
            "author":spec["author"],
            "level":spec["level"],
            "domain":spec.get("domain","pv"),
            "tag":spec["tag"],
            "category":spec.get("category","unknown"),
        }
        setting = {
            "data_needed":spec["data_needed"],
            "universe":spec["universe"],
            "pasteurization":spec["pasteurization"],
            "decay":spec.get("decay",0),
            "neutralize":spec.get("neutralize",None),
        }
        if spec["type"]=="super":
            setting["factor_needed"] = spec["factor_needed"]

        lines = [
            "import pandas as pd",
            "import numpy as np",
            "from quant.quant_lib.analysis import *",
            "",
            f'TYPE = "{spec["type"]}"',
            "",
        ]
        lines.extend(self._render_dict_block("META",meta,["factor_name","author","level","domain","tag","category"]))
        lines.append("")
        setting_keys = ["data_needed"]
        if spec["type"]=="super":
            setting_keys.append("factor_needed")
        setting_keys.extend(["universe","pasteurization","decay","neutralize"])
        lines.extend(self._render_dict_block("SETTING",setting,setting_keys))
        lines.append("")
        lines.extend(self._render_calc_factor(spec))
        return "\n".join(lines)+"\n"

    def render_to_file(self,spec:dict,file_path:str)->str:
        code = self.render_code(spec)
        folder = os.path.dirname(file_path)
        if folder:
            os.makedirs(folder,exist_ok=True)
        with open(file_path,"w",encoding="utf-8") as f:
            f.write(code)
        return file_path

    def render_temp_file(self,spec:dict,temp_dir:str|None=None)->str:
        fd,file_path = tempfile.mkstemp(prefix="factor_spec_",suffix=".py",dir=temp_dir,text=True)
        os.close(fd)
        return self.render_to_file(spec,file_path)
