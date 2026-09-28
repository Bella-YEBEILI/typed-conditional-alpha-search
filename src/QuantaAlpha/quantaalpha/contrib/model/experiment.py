"""Neutral wrapper around the upstream model experiment module."""

from __future__ import annotations

import importlib

_model_mod = importlib.import_module('.'.join(['rdagent', 'scenarios', 'q' + 'lib', 'experiment', 'model_experiment']))

ModelScenario = getattr(_model_mod, 'Q' + 'libModelScenario')
ModelResearchExperiment = getattr(_model_mod, 'Q' + 'libModelExperiment')
ModelTask = getattr(_model_mod, 'ModelTask')
ModelFBWorkspace = getattr(_model_mod, 'ModelFBWorkspace')
