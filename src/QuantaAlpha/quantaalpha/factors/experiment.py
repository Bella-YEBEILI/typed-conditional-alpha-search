"""
QuantaAlpha factor experiment module: neutral scenario and experiment entry points.

This layer keeps the upstream rdagent experiment implementation, while exposing
QuantaAlpha-local names, prompt loading, and workspace wiring for the factor
mining main flow.
"""

import os
import sys
from copy import deepcopy
from pathlib import Path

from rdagent.core.scenario import Scenario

from quantaalpha.core.prompts import Prompts
from quantaalpha.factors.factor_runtime_utils import get_data_folder_intro as local_get_data_folder_intro
from quantaalpha.factors.upstream_factor_api import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
    UpstreamFactorExperiment,
    UpstreamFactorScenario,
    upstream_factor_template_path,
)
from quantaalpha.factors.workspace import FactorWorkspace

_LOCAL_PROMPTS = Prompts(file_path=Path(__file__).parent / 'prompts' / 'experiment.yaml')


def _describe_runtime_environment() -> str:
    """Return a runtime summary without assuming a conda-based local environment."""
    use_local = str(os.getenv("USE_LOCAL", "True")).lower() in {"true", "1"}
    if not use_local:
        return "Docker-based QuantaAlpha factor runtime."

    virtual_env = os.getenv("VIRTUAL_ENV")
    if virtual_env:
        return f"Local Python virtual environment at {virtual_env} using interpreter {sys.executable}."

    conda_env = os.getenv("CONDA_DEFAULT_ENV")
    if conda_env:
        return f"Local conda environment '{conda_env}' using interpreter {sys.executable}."

    return f"Local Python runtime using interpreter {sys.executable}."


class FactorMiningExperiment(UpstreamFactorExperiment):
    """Use the upstream factor experiment core with QuantaAlpha local workspace overrides."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = FactorWorkspace(template_folder_path=upstream_factor_template_path())


class FactorScenario(UpstreamFactorScenario):
    """Neutral scenario alias for the upstream factor scenario."""

    def get_runtime_environment(self) -> str:
        return _describe_runtime_environment()


class AlphaAgentFactorScenario(UpstreamFactorScenario):
    """Scenario wrapper for factor mining that uses QuantaAlpha-local prompts and runtime intro."""

    def get_runtime_environment(self) -> str:
        return _describe_runtime_environment()

    def __init__(self, use_local: bool = True, *args, **kwargs):
        del args, kwargs
        Scenario.__init__(self)

        self._background = deepcopy(
            _LOCAL_PROMPTS['factor_background'].format(
                runtime_environment=self.get_runtime_environment(),
            )
        )
        self._source_data = deepcopy(local_get_data_folder_intro(use_local=use_local))
        self._output_format = deepcopy(_LOCAL_PROMPTS['factor_output_format'])
        self._interface = deepcopy(_LOCAL_PROMPTS['factor_interface'])
        self._strategy = deepcopy(_LOCAL_PROMPTS['factor_strategy'])
        self._simulator = deepcopy(_LOCAL_PROMPTS['factor_simulator'])
        self._rich_style_description = deepcopy(_LOCAL_PROMPTS['factor_rich_style_description'])
        self._experiment_setting = deepcopy(_LOCAL_PROMPTS['factor_experiment_setting'])
