from __future__ import annotations
import ast
import re
import subprocess
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from filelock import FileLock

from quantaalpha.coder.costeer.task import CoSTEERTask
from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
from quantaalpha.factors.combined_domain_contract import validate_joint_pv_minutes_contract
from quantaalpha.factors.data_domains import resolve_factor_domains
from quantaalpha.core.exception import CodeFormatError, CustomRuntimeError, NoOutputError
from quantaalpha.core.experiment import Experiment, FBWorkspace
from quantaalpha.core.utils import cache_with_pickle
from quantaalpha.llm.client import md5_hash


class FactorTask(CoSTEERTask):
    # TODO:  generalized the attributes into the Task
    # - factor_* -> *
    def __init__(
        self,
        factor_name,
        factor_description,
        factor_formulation,
        factor_expression = None,
        *args,
        variables: dict = {},
        resource: str = None,
        factor_implementation: bool = False,
        **kwargs,
    ) -> None:
        self.factor_name = (
            factor_name  # TODO: remove it in the later version. Keep it only for pickle version compatibility
        )
        self.factor_description = factor_description
        self.factor_formulation = factor_formulation
        self.factor_expression = factor_expression
        self.variables = variables
        self.factor_resources = resource
        self.factor_implementation = factor_implementation
        super().__init__(name=factor_name, *args, **kwargs)

    def get_task_information(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""
    

    def get_task_description(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}"""

    def get_task_information_and_implementation_result(self):
        result = {
            "factor_name": self.factor_name,
            "factor_description": self.factor_description,
            "factor_formulation": self.factor_formulation,
            "factor_expression": self.factor_expression,
            "variables": str(self.variables),
            "factor_implementation": str(self.factor_implementation),
        }
        for attr_name in (
            "alpha_evaluation_feedback",
            "runtime_analysis_context",
            "selection_diagnostics_context",
        ):
            value = getattr(self, attr_name, "")
            if value:
                result[attr_name] = value
        submission_check = getattr(self, "submission_check", None)
        if isinstance(submission_check, dict) and submission_check:
            result["submission_check"] = {
                "check_passed": bool(submission_check.get("check_passed")),
                "train_passed": bool(submission_check.get("train_passed")),
                "raw_passed": bool(submission_check.get("raw_passed")),
                "zz1000s_passed": bool(submission_check.get("zz1000s_passed")),
                "complete_passed": bool(submission_check.get("complete_passed")),
                "failed_metrics": [
                    str(item).strip() for item in (submission_check.get("failed_metrics") or []) if str(item).strip()
                ],
                "feedback_text": str(submission_check.get("feedback_text") or ""),
            }
        return result

    @staticmethod
    def from_dict(dict):
        return FactorTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.factor_name}]>"


class FactorFBWorkspace(FBWorkspace):
    """
    This class is used to implement a factor by writing the code to a file.
    Input data and output factor value are also written to files.
    """

    # TODO: (Xiao) think raising errors may get better information for processing
    FB_EXEC_SUCCESS = "Execution succeeded without error."
    FB_CODE_NOT_SET = "code is not set."
    FB_EXECUTION_SUCCEEDED = "Execution succeeded without error."
    FB_OUTPUT_FILE_NOT_FOUND = "\nExpected output file not found."
    FB_OUTPUT_FILE_FOUND = "\nExpected output file found."
    FB_BRIDGE_FALLBACK_SUCCESS = "\nRecovered factor value via deterministic TQ bridge fallback."
    FB_BRIDGE_FALLBACK_FAILED = "\nDeterministic TQ bridge fallback failed."
    EXECUTION_HASH_VERSION = "minute_bridge_fallback_v1"

    def __init__(
        self,
        *args,
        raise_exception: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.raise_exception = raise_exception

    def hash_func(self, data_type: str = "Debug") -> str:
        if "factor.py" not in self.code_dict or self.raise_exception:
            return None
        code_text = self.code_dict["factor.py"]
        if self._looks_like_minute_factor_module(code_text):
            code_text = self._normalize_minute_factor_code(code_text)
        return md5_hash(self.EXECUTION_HASH_VERSION + data_type + code_text)

    def _looks_like_minute_factor_module(self, code_text: str) -> bool:
        normalized = str(code_text or "").lower()
        if '"level"' in normalized and '"minutes"' in normalized:
            return True
        return "prepare_minute_datas" in normalized

    def _validate_joint_pv_minutes_module_contract(self, code_text: str) -> str | None:
        if "STRUCTURED_RENDER_FAILURE_MESSAGE" in str(code_text or ""):
            return None
        validation_errors = validate_joint_pv_minutes_contract(
            code_text,
            active_domains=resolve_factor_domains(),
        )
        if not validation_errors:
            return None
        return "Combined pv/minutes contract validation failed:\n- " + "\n- ".join(validation_errors)

    def _evaluate_factor_with_bridge(
        self,
        code_path: Path,
        output_path: Path,
    ) -> tuple[str, pd.DataFrame | None]:
        try:
            from quantaalpha.backtest.bridge import TQUpstreamBridge

            factor_value = TQUpstreamBridge().evaluate_factor_file_value(code_path, context="mining")
            if output_path.exists():
                output_path.unlink()
            factor_value.to_hdf(output_path, key="data")
            return self.FB_BRIDGE_FALLBACK_SUCCESS, factor_value
        except Exception as exc:
            return f"{self.FB_BRIDGE_FALLBACK_FAILED}: {exc}", None

    def _repair_common_module_syntax(self, code_text: str) -> str:
        text = str(code_text or "")
        if not text.strip():
            return text

        lines = text.splitlines(keepends=True)
        repaired_lines: list[str] = []
        inside_simple_dict = False

        for line in lines:
            if not inside_simple_dict and line.lstrip().startswith(("META = {", "SETTING = {")):
                inside_simple_dict = True
            elif inside_simple_dict and line.lstrip().startswith("}"):
                inside_simple_dict = False

            if inside_simple_dict and repaired_lines and line.lstrip().startswith(('"', "'")):
                previous = repaired_lines[-1]
                previous_body = previous.rstrip("\r\n")
                if previous_body.strip() and not previous_body.rstrip().endswith((",", "{")):
                    newline = previous[len(previous_body) :]
                    repaired_lines[-1] = previous_body + "," + newline

            repaired_lines.append(line)

        repaired = "".join(repaired_lines)
        try:
            ast.parse(repaired)
        except SyntaxError:
            return text
        return repaired

    def _normalize_minute_factor_code(self, code_text: str) -> str:
        normalized = str(code_text or "")

        expression = str(getattr(self.target_task, "factor_expression", "") or "").strip()
        minute_fields = {
            "opens",
            "highs",
            "lows",
            "closes",
            "volumes",
            "amounts",
            "ratios",
            "vwaps",
            "turnovers",
            "returns",
            "lreturns",
        }
        daily_pv_fields = {
            "hfq_opens",
            "hfq_closes",
            "hfq_highs",
            "hfq_lows",
            "volumes",
            "vwaps",
            "turnovers",
            "volume",
            "vwap",
            "turnover",
        }
        joint_match = re.fullmatch(
            r"safe_div\(\s*ts_return\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
            expression,
        )
        if joint_match:
            minute_field, lag_text, daily_field = joint_match.groups()
            if minute_field in minute_fields and daily_field in daily_pv_fields:
                lag = int(lag_text)
                factor_name = self.target_task.factor_name
                return f'''import h5py
import numpy as np
import pandas as pd
from vendors.quant_lib.minute_tools import MinuteFactorEngine, load_single_minute
from quant_union.common.quantEnum import DomainType, CategoryType

TYPE = "regular"
META = {{
    "factor_name": "{factor_name}",
    "author": "quantaalpha",
    "level": "minutes",
    "domain": DomainType.pv,
    "tag": "",
    "category": "unknown",
}}
SETTING = {{
    "universe": "standards",
    "data_needed": ["{daily_field}"],
    "pasteurization": False,
    "decay": 0,
    "neutralize": None,
}}

def prepare_minute_datas():
    mfe = MinuteFactorEngine()
    with h5py.File(mfe.h5_path, "r") as handle:
        minutes = pd.Index(handle["axis/minutes"][:].astype(str))
        if len(minutes) <= {lag}:
            raise ValueError("minute axis is shorter than required lag {lag}")
        current_minute = str(minutes[-1])
        lagged_minute = str(minutes[-1 - {lag}])
        current_value = load_single_minute(handle, "{minute_field}", current_minute)
        lagged_value = load_single_minute(handle, "{minute_field}", lagged_minute).replace(0.0, np.nan)
    minute_return = current_value.divide(lagged_value).subtract(1.0)
    return {{
        "{minute_field}_return_{lag}": minute_return,
    }}

def calc_factor(data_ctx, minute_ctx):
    daily_anchor = data_ctx["{daily_field}"].replace(0.0, np.nan)
    return minute_ctx["{minute_field}_return_{lag}"].divide(daily_anchor)
'''

        joint_agg_match = re.fullmatch(
            r"safe_div\(\s*(?:(mean|sum|ts_mean))\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
            expression,
        )
        if joint_agg_match:
            agg_name, minute_field, window_text, daily_field = joint_agg_match.groups()
            if minute_field in minute_fields and daily_field in daily_pv_fields:
                window = int(window_text)
                lookback = window - 1
                factor_name = self.target_task.factor_name
                groupby_op = "mean" if agg_name in {"mean", "ts_mean"} else "sum"
                return f'''import h5py
import numpy as np
import pandas as pd
from vendors.quant_lib.minute_tools import MinuteFactorEngine, load_single_minute
from quant_union.common.quantEnum import DomainType, CategoryType

TYPE = "regular"
META = {{
    "factor_name": "{factor_name}",
    "author": "quantaalpha",
    "level": "minutes",
    "domain": DomainType.pv,
    "tag": "",
    "category": "unknown",
}}
SETTING = {{
    "universe": "standards",
    "data_needed": ["{daily_field}"],
    "pasteurization": False,
    "decay": 0,
    "neutralize": None,
}}

def prepare_minute_datas():
    mfe = MinuteFactorEngine()
    with h5py.File(mfe.h5_path, "r") as handle:
        minutes = pd.Index(handle["axis/minutes"][:].astype(str))
        if len(minutes) < {window}:
            raise ValueError("minute axis is shorter than required window {window}")
        end_idx = len(minutes) - 1
        start_idx = end_idx - {lookback}
        agg_frames = [
            load_single_minute(handle, "{minute_field}", str(minute_text))
            for minute_text in minutes[start_idx : end_idx + 1]
        ]
    stacked = pd.concat(agg_frames, keys=range(len(agg_frames)), names=["offset"])
    aggregated = stacked.groupby(level=1).{groupby_op}()
    return {{
        "{minute_field}_{agg_name}_{window}": aggregated,
    }}

def calc_factor(data_ctx, minute_ctx):
    daily_anchor = data_ctx["{daily_field}"].replace(0.0, np.nan)
    return minute_ctx["{minute_field}_{agg_name}_{window}"].divide(daily_anchor)
'''

        expr_match = re.fullmatch(r"ts_return\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)", expression)
        if expr_match:
            minute_field, lag_text = expr_match.groups()
            if minute_field in minute_fields:
                lag = int(lag_text)
                factor_name = self.target_task.factor_name
                return f'''import h5py
import numpy as np
import pandas as pd
from vendors.quant_lib.minute_tools import MinuteFactorEngine, load_single_minute
from quant_union.common.quantEnum import DomainType, CategoryType

TYPE = "regular"
META = {{
    "factor_name": "{factor_name}",
    "author": "quantaalpha",
    "level": "minutes",
    "domain": DomainType.pv,
    "tag": "",
    "category": "unknown",
}}
SETTING = {{
    "universe": "standards",
    "data_needed": [],
    "pasteurization": False,
    "decay": 0,
    "neutralize": None,
}}

def prepare_minute_datas():
    mfe = MinuteFactorEngine()
    with h5py.File(mfe.h5_path, "r") as handle:
        minutes = pd.Index(handle["axis/minutes"][:].astype(str))
        if len(minutes) <= {lag}:
            raise ValueError("minute axis is shorter than required lag {lag}")
        current_minute = str(minutes[-1])
        lagged_minute = str(minutes[-1 - {lag}])
        current_value = load_single_minute(handle, "{minute_field}", current_minute)
        lagged_value = load_single_minute(handle, "{minute_field}", lagged_minute)
    return {{
        "{minute_field}_current": current_value,
        "{minute_field}_lag_{lag}": lagged_value,
    }}

def calc_factor(data_ctx, minute_ctx):
    current_value = minute_ctx["{minute_field}_current"]
    lagged_value = minute_ctx["{minute_field}_lag_{lag}"]
    return current_value.divide(lagged_value).subtract(1.0)
'''

        agg_match = re.fullmatch(r"(?:(mean|sum|ts_mean))\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)", expression)
        if agg_match:
            agg_name, minute_field, window_text = agg_match.groups()
            if minute_field in minute_fields:
                window = int(window_text)
                lookback = window - 1
                groupby_op = "mean" if agg_name in {"mean", "ts_mean"} else "sum"
                factor_name = self.target_task.factor_name
                return f'''import h5py
import numpy as np
import pandas as pd
from vendors.quant_lib.minute_tools import MinuteFactorEngine, load_single_minute
from quant_union.common.quantEnum import DomainType, CategoryType

TYPE = "regular"
META = {{
    "factor_name": "{factor_name}",
    "author": "quantaalpha",
    "level": "minutes",
    "domain": DomainType.pv,
    "tag": "",
    "category": "unknown",
}}
SETTING = {{
    "universe": "standards",
    "data_needed": [],
    "pasteurization": False,
    "decay": 0,
    "neutralize": None,
}}

def prepare_minute_datas():
    mfe = MinuteFactorEngine()
    with h5py.File(mfe.h5_path, "r") as handle:
        minutes = pd.Index(handle["axis/minutes"][:].astype(str))
        if len(minutes) < {window}:
            raise ValueError("minute axis is shorter than required window {window}")
        end_idx = len(minutes) - 1
        start_idx = end_idx - {lookback}
        agg_frames = [
            load_single_minute(handle, "{minute_field}", str(minute_text))
            for minute_text in minutes[start_idx : end_idx + 1]
        ]
    aggregated = pd.concat(agg_frames).groupby(level=0).{groupby_op}()
    return {{
        "{minute_field}_{groupby_op}_{window}": aggregated,
    }}

def calc_factor(data_ctx, minute_ctx):
    return minute_ctx["{minute_field}_{groupby_op}_{window}"]
'''

        pct_agg_match = re.fullmatch(r"(?:(mean|sum|ts_mean))\(\s*pct\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)\s*\)", expression)
        if pct_agg_match:
            agg_name, minute_field, lag_text = pct_agg_match.groups()
            if minute_field in minute_fields:
                lag = int(lag_text)
                groupby_op = "mean" if agg_name in {"mean", "ts_mean"} else "sum"
                factor_name = self.target_task.factor_name
                return f'''import h5py
import numpy as np
import pandas as pd
from vendors.quant_lib.minute_tools import MinuteFactorEngine, load_single_minute
from quant_union.common.quantEnum import DomainType, CategoryType

TYPE = "regular"
META = {{
    "factor_name": "{factor_name}",
    "author": "quantaalpha",
    "level": "minutes",
    "domain": DomainType.pv,
    "tag": "",
    "category": "unknown",
}}
SETTING = {{
    "universe": "standards",
    "data_needed": [],
    "pasteurization": False,
    "decay": 0,
    "neutralize": None,
}}

def prepare_minute_datas():
    mfe = MinuteFactorEngine()
    with h5py.File(mfe.h5_path, "r") as handle:
        minutes = pd.Index(handle["axis/minutes"][:].astype(str))
        if len(minutes) <= {lag}:
            raise ValueError("minute axis is shorter than required lag {lag}")
        pct_frames = []
        for current_idx in range({lag}, len(minutes)):
            current_value = load_single_minute(handle, "{minute_field}", str(minutes[current_idx]))
            lagged_value = load_single_minute(handle, "{minute_field}", str(minutes[current_idx - {lag}])).replace(0.0, np.nan)
            pct_frames.append(current_value.divide(lagged_value).subtract(1.0))
    aggregated = pd.concat(pct_frames).groupby(level=0).{groupby_op}()
    return {{
        "{minute_field}_pct_{groupby_op}_{lag}": aggregated,
    }}

def calc_factor(data_ctx, minute_ctx):
    return minute_ctx["{minute_field}_pct_{groupby_op}_{lag}"]
'''

        safe_div_match = re.fullmatch(
            r"safe_div\(\s*ts_return\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)\s*,\s*ts_mean\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*\)\s*\)",
            expression,
        )
        if safe_div_match:
            return_field, return_lag_text, mean_field, mean_window_text = safe_div_match.groups()
            if return_field in minute_fields and mean_field in minute_fields:
                return_lag = int(return_lag_text)
                mean_window = int(mean_window_text)
                lookback = max(return_lag, mean_window - 1)
                factor_name = self.target_task.factor_name
                return f'''import h5py
import numpy as np
import pandas as pd
from vendors.quant_lib.minute_tools import MinuteFactorEngine, load_single_minute
from quant_union.common.quantEnum import DomainType, CategoryType

TYPE = "regular"
META = {{
    "factor_name": "{factor_name}",
    "author": "quantaalpha",
    "level": "minutes",
    "domain": DomainType.pv,
    "tag": "",
    "category": "unknown",
}}
SETTING = {{
    "universe": "standards",
    "data_needed": [],
    "pasteurization": False,
    "decay": 0,
    "neutralize": None,
}}

def prepare_minute_datas():
    mfe = MinuteFactorEngine()
    with h5py.File(mfe.h5_path, "r") as handle:
        minutes = pd.Index(handle["axis/minutes"][:].astype(str))
        if len(minutes) <= {lookback}:
            raise ValueError("minute axis is shorter than required lookback {lookback}")
        current_idx = len(minutes) - 1
        current_minute = str(minutes[current_idx])
        lagged_minute = str(minutes[current_idx - {return_lag}])
        return_current = load_single_minute(handle, "{return_field}", current_minute)
        return_lagged = load_single_minute(handle, "{return_field}", lagged_minute)
        mean_frames = [
            load_single_minute(handle, "{mean_field}", str(minute_text))
            for minute_text in minutes[current_idx - {mean_window} + 1 : current_idx + 1]
        ]
    rolling_mean = pd.concat(mean_frames).groupby(level=0).mean()
    return {{
        "{return_field}_current": return_current,
        "{return_field}_lag_{return_lag}": return_lagged,
        "{mean_field}_mean_{mean_window}": rolling_mean,
    }}

def calc_factor(data_ctx, minute_ctx):
    numerator = minute_ctx["{return_field}_current"].divide(
        minute_ctx["{return_field}_lag_{return_lag}"]
    ).subtract(1.0)
    denominator = minute_ctx["{mean_field}_mean_{mean_window}"].replace(0.0, np.nan)
    return numerator.divide(denominator)
'''

        for pattern in (
            "from quantaalpha.common.data import TQUpstreamBridge",
            "from quantaalpha.utils.tq_bridge import TQUpstreamBridge",
            "from quantaalpha.data.tq_bridge import TQUpstreamBridge",
            "from quantaalpha.upstream import TQUpstreamBridge",
            "from quantaalpha.backtest.tq_bridge import TQUpstreamBridge",
            "from quantaalpha.backtest.minute_tools import TQUpstreamBridge",
            "from quantaalpha.backtest import TQUpstreamBridge",
        ):
            normalized = normalized.replace(
                pattern,
                "from quantaalpha.backtest.bridge import TQUpstreamBridge",
            )

        if '"factor_name"' not in normalized and "'factor_name'" not in normalized:
            normalized = re.sub(
                r"(META\s*=\s*\{)",
                rf'\1\n    "factor_name": "{self.target_task.factor_name}",',
                normalized,
                count=1,
            )

        normalized = re.sub(
            r'("universe"\s*:\s*)["\'][^"\']+["\']',
            r'\1"standards"',
            normalized,
            count=1,
        )
        if '"universe"' not in normalized and "'universe'" not in normalized and "SETTING = {" in normalized:
            normalized = normalized.replace("SETTING = {\n", 'SETTING = {\n    "universe": "standards",\n', 1)
        if '"pasteurization"' not in normalized and "'pasteurization'" not in normalized and "SETTING = {" in normalized:
            normalized = normalized.replace("SETTING = {\n", 'SETTING = {\n    "pasteurization": False,\n', 1)

        normalized = re.sub(
            r"\.pct_change\(\s*\)",
            ".pct_change(fill_method=None)",
            normalized,
        )
        normalized = re.sub(
            r"\.pct_change\(\s*(\d+)\s*\)",
            r".pct_change(periods=\1, fill_method=None)",
            normalized,
        )
        normalized = re.sub(
            r"\.diff\(\s*fill_method\s*=\s*None\s*\)",
            ".diff()",
            normalized,
        )
        normalized = re.sub(
            r"\.diff\(\s*periods\s*=\s*(\d+)\s*,\s*fill_method\s*=\s*None\s*\)",
            r".diff(periods=\1)",
            normalized,
        )
        normalized = re.sub(
            r"\.shift\(\s*fill_method\s*=\s*None\s*\)",
            ".shift()",
            normalized,
        )
        normalized = re.sub(
            r"\.shift\(\s*periods\s*=\s*(\d+)\s*,\s*fill_method\s*=\s*None\s*\)",
            r".shift(periods=\1)",
            normalized,
        )

        return self._repair_common_module_syntax(normalized)

    @cache_with_pickle(hash_func)
    def execute(self, data_type: str = "Debug") -> Tuple[str, pd.DataFrame]:
        """
        execute the implementation and get the factor value by the following steps:
        1. make the directory in workspace path
        2. write the code to the file in the workspace path
        3. execute factor.py directly inside the workspace
        4. read the factor value from result.h5 in the workspace path folder
        returns the execution feedback as a string and the factor value as a pandas dataframe


        Regarding the cache mechanism:
        1. We will store the function's return value to ensure it behaves as expected.
        - The cached information will include a tuple with the following: (execution_feedback, executed_factor_value_dataframe, Optional[Exception])

        """
        super().execute()
        if self.code_dict is None or "factor.py" not in self.code_dict:
            if self.raise_exception:
                raise CodeFormatError(self.FB_CODE_NOT_SET)
            else:
                return self.FB_CODE_NOT_SET, None
        workspace_dir = Path(self.workspace_path).resolve(strict=False)
        with FileLock(workspace_dir / "execution.lock"):
            code_path = workspace_dir / "factor.py"
            if self._looks_like_minute_factor_module(self.code_dict.get("factor.py", "")):
                normalized_code = self._normalize_minute_factor_code(self.code_dict["factor.py"])
                if normalized_code != self.code_dict["factor.py"]:
                    self.code_dict["factor.py"] = normalized_code
                    code_path.write_text(normalized_code, encoding="utf-8")

            contract_feedback = self._validate_joint_pv_minutes_module_contract(self.code_dict.get("factor.py", ""))
            if contract_feedback is not None:
                if self.raise_exception:
                    raise CodeFormatError(contract_feedback)
                return contract_feedback + self.FB_OUTPUT_FILE_NOT_FOUND, None

            execution_feedback = self.FB_EXECUTION_SUCCEEDED
            execution_success = False
            execution_error = None
            bridge_generated_factor_value = None

            execution_code_path = code_path
            is_minute_module = self._looks_like_minute_factor_module(self.code_dict.get("factor.py", ""))

            if is_minute_module:
                bridge_feedback, bridge_generated_factor_value = self._evaluate_factor_with_bridge(
                    execution_code_path,
                    workspace_dir / "result.h5",
                )
                if bridge_generated_factor_value is not None:
                    execution_feedback += bridge_feedback
                    execution_success = True
                else:
                    execution_feedback += f"\n{bridge_feedback}"

            if not execution_success:
                try:
                    # Set PYTHONPATH to include the project root so quantaalpha can be imported
                    import os
                    env = os.environ.copy()
                    project_root = Path(__file__).resolve().parents[3]
                    pythonpath = str(project_root)
                    if 'PYTHONPATH' in env:
                        env['PYTHONPATH'] = pythonpath + os.pathsep + env['PYTHONPATH']
                    else:
                        env['PYTHONPATH'] = pythonpath
                    
                    subprocess.check_output(
                        [FACTOR_COSTEER_SETTINGS.python_bin, execution_code_path.name],
                        shell=False,
                        cwd=workspace_dir,
                        stderr=subprocess.STDOUT,
                        timeout=FACTOR_COSTEER_SETTINGS.file_based_execution_timeout,
                        env=env,
                    )
                    execution_success = True
                except subprocess.CalledProcessError as e:
                    import site

                    execution_feedback = (
                        e.output.decode()
                        .replace(str(execution_code_path.parent.absolute()), r"/path/to")
                        .replace(str(site.getsitepackages()[0]), r"/path/to/site-packages")
                    )
                    if len(execution_feedback) > 2000:
                        execution_feedback = (
                            execution_feedback[:1000] + "....hidden long error message...." + execution_feedback[-1000:]
                        )
                    if self.raise_exception:
                        raise CustomRuntimeError(execution_feedback)
                    else:
                        execution_error = CustomRuntimeError(execution_feedback)
                except subprocess.TimeoutExpired:
                    execution_feedback += f"Execution timeout error and the timeout is set to {FACTOR_COSTEER_SETTINGS.file_based_execution_timeout} seconds."
                    if self.raise_exception:
                        raise CustomRuntimeError(execution_feedback)
                    else:
                        execution_error = CustomRuntimeError(execution_feedback)

            workspace_output_file_path = workspace_dir / "result.h5"
            if bridge_generated_factor_value is None and (not execution_success or not workspace_output_file_path.exists()):
                bridge_feedback, bridge_generated_factor_value = self._evaluate_factor_with_bridge(
                    execution_code_path,
                    workspace_output_file_path,
                )
                if bridge_generated_factor_value is not None:
                    if execution_feedback != self.FB_EXECUTION_SUCCEEDED:
                        execution_feedback = (
                            f"{self.FB_EXECUTION_SUCCEEDED}"
                            f"{self.FB_BRIDGE_FALLBACK_SUCCESS}\n"
                            f"Initial direct execution feedback:\n{execution_feedback}"
                        )
                    else:
                        execution_feedback += self.FB_BRIDGE_FALLBACK_SUCCESS
                    execution_success = True
                    execution_error = None
                else:
                    execution_feedback += f"\n{bridge_feedback}"
            if workspace_output_file_path.exists() and execution_success:
                try:
                    if bridge_generated_factor_value is not None:
                        executed_factor_value_dataframe = bridge_generated_factor_value
                    else:
                        executed_factor_value_dataframe = pd.read_hdf(workspace_output_file_path)
                    execution_feedback += self.FB_OUTPUT_FILE_FOUND
                except Exception as e:
                    execution_feedback += f"Error found when reading hdf file: {e}"[:1000]
                    executed_factor_value_dataframe = None
            else:
                execution_feedback += self.FB_OUTPUT_FILE_NOT_FOUND
                executed_factor_value_dataframe = None
                if self.raise_exception:
                    raise NoOutputError(execution_feedback)
                else:
                    execution_error = NoOutputError(execution_feedback)

        return execution_feedback, executed_factor_value_dataframe

    def __str__(self) -> str:
        # NOTE:
        # If the code cache works, the workspace will be None.
        return f"File Factor[{self.target_task.factor_name}]: {self.workspace_path}"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_folder(task: FactorTask, path: Union[str, Path], **kwargs):
        path = Path(path)
        code_dict = {}
        for file_path in path.iterdir():
            if file_path.suffix == ".py":
                code_dict[file_path.name] = file_path.read_text()
        return FactorFBWorkspace(target_task=task, code_dict=code_dict, **kwargs)


FactorExperiment = Experiment
FeatureExperiment = Experiment
