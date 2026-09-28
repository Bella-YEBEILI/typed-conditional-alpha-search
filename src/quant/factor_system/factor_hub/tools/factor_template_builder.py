"""
FactorTemplateBuilder用于基于内置模板快速生成因子骨架文件。

当前职责：
1. 读取regular或super或minute模板文件。
2. 按传入参数覆盖META和SETTING中的字段。
3. 生成TYPE、META、SETTING和calc_factor代码。
4. 把结果写入目标目录中的regular.py或super.py或minute.py。

当前规则：
- 支持的模板类型只有regular、super。level=minutes时使用minute模板。
- 可覆盖的字段仅限模板协议中已有字段。
- 未显式传入时，tag默认补为空字符串。
- data_needed默认补为空列表，super的factor_needed也默认补为空列表。

这个组件只负责模板生成，不负责校验业务依赖，也不负责执行回测或提交入库。
"""

import ast
import os


class FactorTemplateBuilder:
    def __init__(self):
        self.template_dir = os.path.normpath(os.path.join(os.path.dirname(__file__),"..","templates"))

    def _get_template_path(self,factor_type:str,level:str="days")->str:
        if level=="minutes":
            return os.path.join(self.template_dir,"minute_factor_template.py")
        if factor_type=="regular":
            return os.path.join(self.template_dir,"regular_factor_template.py")
        if factor_type=="super":
            return os.path.join(self.template_dir,"super_factor_template.py")
        raise ValueError(f"unsupported factor type: {factor_type}")

    def _update_dict_node(self,node:ast.Dict,key:str,value):
        key_node = ast.Constant(value=key)
        value_node = ast.parse(repr(value),mode="eval").body
        for i,exist_key in enumerate(node.keys):
            if isinstance(exist_key,ast.Constant) and exist_key.value==key:
                node.values[i] = value_node
                return
        node.keys.append(key_node)
        node.values.append(value_node)

    def _get_dict_value_node(self,node:ast.Dict,key:str):
        for exist_key,value_node in zip(node.keys,node.values):
            if isinstance(exist_key,ast.Constant) and exist_key.value==key:
                return value_node
        return None

    def _line_starts(self,source:str)->list[int]:
        starts = [0]
        for i,char in enumerate(source):
            if char=="\n":
                starts.append(i+1)
        return starts

    def _node_span(self,line_starts:list[int],node:ast.AST)->tuple[int,int]:
        start = line_starts[node.lineno-1]+node.col_offset
        end = line_starts[node.end_lineno-1]+node.end_col_offset
        return start,end

    def _apply_replacements(self,source:str,replacements:list[tuple[int,int,str]])->str:
        for start,end,new_text in sorted(replacements,key=lambda item:item[0],reverse=True):
            source = source[:start]+new_text+source[end:]
        return source

    def _dict_from_node(self,node:ast.Dict)->dict:
        data = {}
        for key_node,value_node in zip(node.keys,node.values):
            if not isinstance(key_node,ast.Constant) or not isinstance(key_node.value,str):
                raise ValueError("template dict key must be str")
            data[key_node.value] = ast.literal_eval(value_node)
        return data

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

    def _render_calc_factor(self,factor_type:str,level:str="days")->list[str]:
        if level=="minutes":
            if factor_type=="super":
                return [
                    "def calc_factor(data_ctx:dict,factor_ctx:dict,minute_ctx:dict)->pd.DataFrame:",
                    "    pass",
                ]
            return [
                "def calc_factor(data_ctx:dict,minute_ctx:dict)->pd.DataFrame:",
                "    pass",
            ]
        if factor_type=="regular":
            return [
                "def calc_factor(data_ctx:dict)->pd.DataFrame:",
                "    pass",
            ]
        return [
            "def calc_factor(data_ctx:dict,factor_ctx:dict)->pd.DataFrame:",
            "    pass",
        ]

    def _render_prepare_minute_datas(self)->list[str]:
        return [
            "def prepare_minute_datas()->dict[str,pd.DataFrame]:",
            "    mfe = MinuteFactorEngine()",
            "    return {}",
        ]

    def _render_template_code(self,factor_type:str,meta:dict,setting:dict,import_nodes:list[ast.stmt],level:str="days")->str:
        lines = []
        for node in import_nodes:
            lines.append(ast.unparse(node))
        if lines:
            lines.append("")

        lines.append(f'TYPE = "{factor_type}"')
        lines.append("")
        lines.extend(self._render_dict_block("META",meta,["factor_name","author","level","domain","tag","category"]))
        lines.append("")
        setting_keys = ["data_needed"]
        if factor_type=="super":
            setting_keys.append("factor_needed")
        setting_keys.extend(["universe","pasteurization","decay","neutralize"])
        lines.extend(self._render_dict_block("SETTING",setting,setting_keys))
        lines.append("")
        if level=="minutes":
            lines.extend(self._render_prepare_minute_datas())
            lines.append("")
        lines.extend(self._render_calc_factor(factor_type,level))
        return "\n".join(lines)+"\n"

    def create_template(self,factor_type:str="regular",out_dir:str|None=None,**kwargs)->str:
        factor_type = kwargs.get("type",factor_type)
        level = kwargs.get("level","days")
        template_path = self._get_template_path(factor_type,level)
        with open(template_path,"r",encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source,filename=template_path)

        assign_map = {}
        for node in tree.body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                assign_map[node.targets[0].id] = node

        if "TYPE" not in assign_map or "META" not in assign_map or "SETTING" not in assign_map:
            raise ValueError(f"invalid template file: {template_path}")

        meta_node = assign_map["META"]
        setting_node = assign_map["SETTING"]
        if not isinstance(meta_node.value,ast.Dict) or not isinstance(setting_node.value,ast.Dict):
            raise ValueError(f"invalid template file: {template_path}")

        overrides = dict(kwargs)
        if "name" in overrides and "factor_name" not in overrides:
            overrides["factor_name"] = overrides.pop("name")

        overrides.pop("file_name",None)
        meta_keys = {"factor_name","author","level","domain","tag","category"}
        setting_keys = {"data_needed","factor_needed","universe","pasteurization","decay","neutralize"}
        line_starts = self._line_starts(source)
        replacements = [(*self._node_span(line_starts,assign_map["TYPE"].value),repr(factor_type))]

        for key,value in overrides.items():
            if key in meta_keys:
                value_node = self._get_dict_value_node(meta_node.value,key)
                if value_node is None:
                    raise ValueError(f"template missing META key: {key}")
                replacements.append((*self._node_span(line_starts,value_node),repr(value)))
            elif key in setting_keys:
                value_node = self._get_dict_value_node(setting_node.value,key)
                if value_node is None:
                    raise ValueError(f"template missing SETTING key: {key}")
                replacements.append((*self._node_span(line_starts,value_node),repr(value)))
            elif key not in ("type",):
                raise ValueError(f"unsupported template override: {key}")

        factor_name = overrides.get("factor_name")
        if factor_name is not None and (not isinstance(factor_name,str) or factor_name==""):
            raise ValueError("factor_name must be non-empty")

        out_dir = os.getcwd() if out_dir is None else out_dir
        os.makedirs(out_dir,exist_ok=True)
        if level=="minutes":
            out_filename = "minute.py"
        else:
            out_filename = f"{factor_type}.py"
        out_path = os.path.join(out_dir,out_filename)
        code = self._apply_replacements(source,replacements)
        if not code.endswith("\n"):
            code += "\n"
        with open(out_path,"w",encoding="utf-8") as f:
            f.write(code)
        return out_path
