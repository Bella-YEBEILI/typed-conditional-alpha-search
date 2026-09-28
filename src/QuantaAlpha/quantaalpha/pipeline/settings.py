"""
QuantaAlpha pipeline settings.

Defines class-path configuration for pipeline components.
Components are loaded dynamically via string class paths for flexibility.
"""

from quantaalpha.core.conf import ExtendedBaseSettings, ExtendedSettingsConfigDict


# =============================================================================
# Base setting classes
# =============================================================================

class BasePropSetting(ExtendedBaseSettings):
    """Common base for RD Loop configuration."""

    scen: str = ""
    knowledge_base: str = ""
    knowledge_base_path: str = ""
    hypothesis_gen: str = ""
    hypothesis2experiment: str = ""
    coder: str = ""
    runner: str = ""
    summarizer: str = ""
    evolving_n: int = 10


class BaseFacSetting(ExtendedBaseSettings):
    """Common base for Alpha Agent Loop configuration."""

    scen: str = ""
    knowledge_base: str = ""
    knowledge_base_path: str = ""
    hypothesis_gen: str = ""
    construction: str = ""
    calculation: str = ""
    coder: str = ""
    runner: str = ""
    summarizer: str = ""
    evolving_n: int = 10


# =============================================================================
# Factor mining settings (main experiment)
# =============================================================================

class AlphaAgentFactorBasePropSetting(BasePropSetting):
    """Main experiment: LLM-driven factor mining."""
    model_config = ExtendedSettingsConfigDict(env_prefix="FACTOR_", protected_namespaces=())

    scen: str = "quantaalpha.factors.experiment.AlphaAgentFactorScenario"
    hypothesis_gen: str = "quantaalpha.factors.proposal.AlphaAgentHypothesisGen"
    hypothesis2experiment: str = "quantaalpha.factors.proposal.AlphaAgentHypothesis2FactorExpression"
    coder: str = "quantaalpha.factors.factor_coder.FactorCoder"
    runner: str = "quantaalpha.factors.tq_runner.TQFactorRunner"
    summarizer: str = "quantaalpha.factors.feedback.AlphaAgentFactorHypothesisExperiment2Feedback"
    evolving_n: int = 5


class FactorBasePropSetting(BasePropSetting):
    """Basic factor experiment (traditional RD Loop mode)."""
    model_config = ExtendedSettingsConfigDict(env_prefix="FACTOR_", protected_namespaces=())

    scen: str = "quantaalpha.factors.experiment.FactorScenario"
    hypothesis_gen: str = "quantaalpha.factors.proposal.FactorHypothesisGen"
    hypothesis2experiment: str = "quantaalpha.factors.proposal.FactorHypothesis2Experiment"
    coder: str = "quantaalpha.factors.factor_coder.FactorCoSTEER"
    runner: str = "quantaalpha.factors.tq_runner.TQFactorRunner"
    summarizer: str = "quantaalpha.factors.feedback.FactorHypothesisExperiment2Feedback"
    evolving_n: int = 10


class FactorBackTestBasePropSetting(BasePropSetting):
    """Factor backtest mode."""
    model_config = ExtendedSettingsConfigDict(env_prefix="FACTOR_", protected_namespaces=())

    scen: str = "quantaalpha.factors.experiment.AlphaAgentFactorScenario"
    hypothesis_gen: str = "quantaalpha.factors.proposal.EmptyHypothesisGen"
    hypothesis2experiment: str = "quantaalpha.factors.proposal.BacktestHypothesis2FactorExpression"
    coder: str = "quantaalpha.factors.factor_coder.FactorCoder"
    runner: str = "quantaalpha.factors.tq_runner.TQFactorRunner"
    summarizer: str = "quantaalpha.factors.feedback.FactorHypothesisExperiment2Feedback"
    evolving_n: int = 1


class FactorFromReportPropSetting(FactorBasePropSetting):
    """Factor extraction from research reports."""
    scen: str = "quantaalpha.factors.experiment.FactorScenario"
    report_result_json_file_path: str = "git_ignore_folder/report_list.json"
    max_factors_per_exp: int = 10000
    is_report_limit_enabled: bool = False


# =============================================================================
# Model experiment settings (contrib, optional)
# =============================================================================

class ModelBasePropSetting(BasePropSetting):
    """Model experiment (extended feature)."""
    model_config = ExtendedSettingsConfigDict(env_prefix="MODEL_", protected_namespaces=())

    scen: str = "quantaalpha.contrib.model.experiment.ModelScenario"
    hypothesis_gen: str = "quantaalpha.contrib.model.proposal.ModelHypothesisGen"
    hypothesis2experiment: str = "quantaalpha.contrib.model.proposal.ModelHypothesis2Experiment"
    coder: str = "quantaalpha.contrib.model.model_coder.ModelCoSTEER"
    runner: str = "quantaalpha.contrib.model.runner.ModelRunner"
    summarizer: str = "quantaalpha.factors.feedback.ModelHypothesisExperiment2Feedback"
    evolving_n: int = 10


# =============================================================================
# Singleton instances (global)
# =============================================================================

ALPHA_AGENT_FACTOR_PROP_SETTING = AlphaAgentFactorBasePropSetting()
FACTOR_PROP_SETTING = FactorBasePropSetting()
FACTOR_BACK_TEST_PROP_SETTING = FactorBackTestBasePropSetting()
FACTOR_FROM_REPORT_PROP_SETTING = FactorFromReportPropSetting()
MODEL_PROP_SETTING = ModelBasePropSetting()

