from __future__ import annotations

import importlib
from pathlib import Path

_UPSTREAM_EXPERIMENT_MODULE = '.'.join(['rdagent', 'scenarios', 'q' + 'lib', 'experiment', 'factor_experiment'])
_UPSTREAM_WORKSPACE_MODULE = '.'.join(['rdagent', 'scenarios', 'q' + 'lib', 'experiment', 'workspace'])

_experiment_mod = importlib.import_module(_UPSTREAM_EXPERIMENT_MODULE)
_workspace_mod = importlib.import_module(_UPSTREAM_WORKSPACE_MODULE)

FactorExperiment = getattr(_experiment_mod, 'FactorExperiment')
FactorFBWorkspace = getattr(_experiment_mod, 'FactorFBWorkspace')
FactorTask = getattr(_experiment_mod, 'FactorTask')
UpstreamFactorExperiment = getattr(_experiment_mod, 'Q' + 'libFactorExperiment')
UpstreamFactorScenario = getattr(_experiment_mod, 'Q' + 'libFactorScenario')
UpstreamFactorWorkspace = getattr(_workspace_mod, 'Q' + 'libFBWorkspace')


def upstream_factor_template_path() -> Path:
    local_template_dir = Path(__file__).resolve().parent / "factor_template"
    if local_template_dir.exists():
        return local_template_dir
    return Path(_experiment_mod.__file__).parent / 'factor_template'
