from quantaalpha.backtest.bridge import TQUpstreamBridge
from quantaalpha.backtest.evaluator import TQStandaloneEvaluator
from quantaalpha.factors.data_domains import canonicalize_joint_pv_daily_aliases
import pandas as pd


class _AliasRecordingProvider:
    def __init__(self):
        self.requested = []

    def get_single_data(self, field_name):
        self.requested.append(field_name)
        return field_name


def test_daily_pv_module_contract_rejects_raw_ohlc():
    meta = {"level": "days", "domain": "pv"}
    setting = {"data_needed": ["opens", "volumes"]}
    try:
        TQStandaloneEvaluator._validate_daily_pv_field_contract(meta, setting)
    except ValueError as exc:
        assert "hfq_*" in str(exc)
    else:
        raise AssertionError("expected raw daily OHLC to be rejected")


def test_daily_pv_expression_contract_rejects_raw_ohlc():
    try:
        TQStandaloneEvaluator._validate_daily_pv_expression_fields({"opens", "volumes"}, domains=("pv",))
    except ValueError as exc:
        assert "hfq_*" in str(exc)
    else:
        raise AssertionError("expected raw daily OHLC expression fields to be rejected")


def test_daily_pv_expression_contract_rejects_double_hfq():
    try:
        TQStandaloneEvaluator._validate_daily_pv_expression_fields({"hfq_hfq_closes"}, domains=("pv",))
    except ValueError as exc:
        assert "doubly adjusted" in str(exc)
    else:
        raise AssertionError("expected doubly adjusted hfq fields to be rejected")


def test_evaluator_maps_singular_daily_shared_aliases_to_runtime_fields():
    provider = _AliasRecordingProvider()
    evaluator = TQStandaloneEvaluator.__new__(TQStandaloneEvaluator)
    evaluator.dp = provider
    bridge = TQUpstreamBridge.__new__(TQUpstreamBridge)

    data_ctx = evaluator._prepare_data_ctx(
        ["volume", "vwap", "turnover"],
        univ="unused",
        pasteurization=False,
        field_aliases=bridge.get_factor_field_aliases("mining"),
    )

    assert data_ctx == {"volume": "volumes", "vwap": "vwaps", "turnover": "turnovers"}
    assert provider.requested == ["volumes", "vwaps", "turnovers"]


def test_evaluator_injects_singular_daily_alias_globals_for_modules():
    frame = pd.DataFrame({"A": [1.0, 2.0]})
    provider = _AliasRecordingProvider()
    provider.get_single_data = lambda field_name: (
        pd.DataFrame({"A": [True, True]}) if field_name == "standards" else frame
    )
    evaluator = TQStandaloneEvaluator.__new__(TQStandaloneEvaluator)
    evaluator.dp = provider
    evaluator.analysis = __import__("quantaalpha.backtest.analysis", fromlist=["analysis"])

    module_globals = {
        "TYPE": "regular",
        "META": {"factor_name": "alias_factor", "level": "days", "domain": "pv"},
        "SETTING": {"data_needed": ["turnover"], "universe": "standards"},
    }
    exec("def calc_factor(data_ctx):\n    return turnover + data_ctx['turnover']", module_globals, module_globals)

    result = evaluator.evaluate_factor_module(
        module_globals,
        field_aliases={"turnover": "turnovers"},
    )

    assert result.equals(frame + frame)


def test_published_expression_aliases_are_pluralized():
    expression = "safe_div(ts_mean(returns, 15), vwap) + turnover + volume_mean_20"

    assert (
        canonicalize_joint_pv_daily_aliases(expression)
        == "safe_div(ts_mean(returns, 15), vwaps) + turnovers + volume_mean_20"
    )
