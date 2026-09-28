import os

import pytest

from quantaalpha.factors.coder.evolving_strategy import (
    FactorParsingStrategy,
    _build_joint_pv_minutes_routing_note,
    _render_expression_based_module,
    _render_structured_minute_failure_module,
)
from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
from quantaalpha.factors.coder.factor import FactorTask
from quantaalpha.factors.combined_domain_contract import validate_joint_pv_minutes_contract
from quantaalpha.factors.proposal import _build_minute_runtime_preflight_feedback


def _task(expr: str) -> FactorTask:
    return FactorTask("demo_factor", "demo", "demo", factor_expression=expr)


@pytest.fixture(autouse=True)
def _restore_factor_data_domains():
    original = os.environ.get("FACTOR_DATA_DOMAINS")
    yield
    if original is None:
        os.environ.pop("FACTOR_DATA_DOMAINS", None)
    else:
        os.environ["FACTOR_DATA_DOMAINS"] = original


def test_render_expression_based_module_pv_emits_daily_template():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv"
    module = _render_expression_based_module(_task("ts_return(hfq_closes, 5)"))
    assert module is not None
    assert 'level": "days"' in module
    assert "ts_return(hfq_closes, 5)" in module
    assert "prepare_minute_datas" not in module


def test_render_expression_based_module_fundamental_emits_daily_template():
    os.environ["FACTOR_DATA_DOMAINS"] = "fundamental"
    module = _render_expression_based_module(_task("ts_return(roe, 5)"))
    assert module is not None
    assert 'level": "days"' in module
    assert '"domain": DomainType.fundamental' in module
    assert "prepare_minute_datas" not in module


def test_render_expression_based_module_minutes_emits_minute_template():
    os.environ["FACTOR_DATA_DOMAINS"] = "minutes"
    module = _render_expression_based_module(_task("ts_mean(turnovers, 15)"))
    assert module is not None
    assert 'level": "minutes"' in module
    assert "JOINT_PV_MINUTES_TEMPLATE" not in module
    assert "prepare_minute_datas" in module
    assert '"where": _where_runtime' in module
    assert '"ts_regression": _ts_regression_runtime' in module


def test_render_expression_based_module_minutes_ts_regression2_accepts_min_periods():
    os.environ["FACTOR_DATA_DOMAINS"] = "minutes"
    module = _render_expression_based_module(_task("ts_regression2(lreturns, volumes, 90, None, 2)"))
    assert module is not None
    ns = {}
    exec(module, ns)

    calls = []

    def fake_ts_reg(df1, df2, p=5, min_periods=None, rettype=0):
        calls.append((df1, df2, p, min_periods, rettype))
        return "ok"

    ns["ts_reg"] = fake_ts_reg
    fn = ns["_build_exec_globals"]({"lreturns": "lr", "volumes": "vol"}, {})["ts_regression2"]

    assert fn("lr", "vol", 90, None, 2) == "ok"
    assert calls == [("lr", "vol", 90, None, 2)]


def test_render_expression_based_module_clamps_oversized_minute_windows():
    os.environ["FACTOR_DATA_DOMAINS"] = "minutes"
    module = _render_expression_based_module(_task("ts_std(turnovers, 480)"))
    assert module is not None
    assert "_window = min(480," in module
    assert "shorter than required window 480" not in module


def test_render_expression_based_module_joint_emits_minute_template_with_daily_anchor():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv,minutes"
    module = _render_expression_based_module(
        _task("safe_div(ts_mean(returns, 15), vwaps)")
    )
    assert module is not None
    assert 'level": "minutes"' in module
    assert "JOINT_PV_MINUTES_TEMPLATE = True" in module
    assert "MINUTE_DATA_NEEDED = ['returns']" in module
    assert 'data_ctx["vwaps"]' in module
    assert '"data_needed"' in module
    assert validate_joint_pv_minutes_contract(module, active_domains=("pv", "minutes")) == []


def test_render_expression_based_module_joint_ts_regression2_accepts_min_periods():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv,minutes"
    module = _render_expression_based_module(
        _task("safe_div(ts_regression2(lreturns, volumes, 90, None, 2), vwaps)")
    )
    assert module is not None
    ns = {}
    exec(module, ns)

    calls = []

    def fake_ts_reg(df1, df2, p=5, min_periods=None, rettype=0):
        calls.append((df1, df2, p, min_periods, rettype))
        return "ok"

    ns["ts_reg"] = fake_ts_reg
    fn = ns["_build_exec_globals"](
        {"lreturns": "lr", "volumes": "daily_vol", "vwaps": "daily_vwap"},
        {"volumes": "minute_vol"},
    )["ts_regression2"]

    assert fn("lr", "minute_vol", 90, None, 2) == "ok"
    assert calls == [("lr", "minute_vol", 90, None, 2)]


def test_render_expression_based_module_joint_ts_beta_broadcasts_cs_mean_series():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv,minutes"
    module = _render_expression_based_module(
        _task("mul(ts_beta(ts_return(hfq_closes, 1), cs_mean(ts_return(hfq_closes, 1)), 2), ts_mean(returns, 2))")
    )
    assert module is not None
    ns = {}
    exec(module, ns)
    import pandas as pd

    dates = pd.date_range("2024-01-01", periods=4)
    close = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3], "B": [1.0, 0.9, 1.0, 1.1]}, index=dates)
    minute_feature = pd.DataFrame({"A": [0.1, 0.2, 0.3, 0.4], "B": [0.2, 0.3, 0.4, 0.5]}, index=dates)

    result = ns["calc_factor"]({"hfq_closes": close}, {"returns_mean_2": minute_feature})

    assert list(result.columns) == ["A", "B"]


def test_render_expression_based_module_joint_ts_multireg_broadcasts_series_predictors():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv,minutes"
    module = _render_expression_based_module(
        _task("ts_multireg([cs_mean(ts_return(hfq_closes, 1)), ts_mean(returns, 2)], ts_return(hfq_closes, 1), 2, rettype=0)")
    )
    assert module is not None
    ns = {}
    exec(module, ns)
    import pandas as pd

    dates = pd.date_range("2024-01-01", periods=4)
    close = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3], "B": [1.0, 0.9, 1.0, 1.1]}, index=dates)
    minute_feature = pd.DataFrame({"A": [0.1, 0.2, 0.3, 0.4], "B": [0.2, 0.3, 0.4, 0.5]}, index=dates)

    result = ns["calc_factor"]({"hfq_closes": close}, {"returns_mean_2": minute_feature})

    assert list(result.columns) == ["A", "B"]


def test_render_expression_based_module_minutes_runtime_min_and_where_defaults():
    os.environ["FACTOR_DATA_DOMAINS"] = "minutes"
    module = _render_expression_based_module(_task("min(ts_mean(volumes, 20), 2)"))
    assert module is not None
    ns = {}
    exec(module, ns)
    import pandas as pd

    frame = pd.DataFrame({"A": [1.0, 3.0], "B": [4.0, 0.5]})
    globals_ = ns["_build_exec_globals"]({}, {"volumes_mean_20": frame})

    capped = globals_["min"](frame, 2)
    assert capped["A"].tolist() == [1.0, 2.0]
    assert capped["B"].tolist() == [2.0, 0.5]
    assert globals_["where"](frame > 1.0, 1).equals(pd.DataFrame({"A": [0.0, 1.0], "B": [1.0, 0.0]}))
    assert globals_["where"](frame, 1, 0).equals(pd.DataFrame({"A": [1.0, 1.0], "B": [1.0, 1.0]}))



def test_render_expression_based_module_minutes_does_not_emit_oversized_snapshot_return_feature():
    os.environ["FACTOR_DATA_DOMAINS"] = "minutes"
    module = _render_expression_based_module(_task("ts_return(closes, 60480)"))
    assert module is not None
    assert "minute axis is shorter than required lag 60480" not in module


def test_render_expression_based_module_joint_supports_daily_pct_alias():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv,minutes"
    module = _render_expression_based_module(
        _task("cs_rank(pct(hfq_closes, ts_mean(turnovers, 20))) + cs_rank(ts_mean(returns, 15))")
    )
    assert module is not None
    ns = {}
    exec(module, ns)
    assert "pct" in ns


def test_render_expression_based_module_joint_publishes_plural_daily_aliases():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv,minutes"
    module = _render_expression_based_module(_task("safe_div(ts_mean(returns, 15), vwap)"))
    assert module is not None
    assert 'ORIGINAL_FACTOR_EXPRESSION = """safe_div(ts_mean(returns, 15), vwaps)"""' in module
    assert 'FINAL_EXPRESSION = """safe_div(returns_mean_15, data_ctx["vwaps"])"""' in module
    assert "DATA_NEEDED = ['vwaps']" in module
    assert 'data_ctx["vwap"]' not in module


def test_render_expression_based_module_supports_last_runtime_fallback():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv"
    module = _render_expression_based_module(_task("last(hfq_closes)"))
    assert module is not None
    ns = {}
    exec(module, ns)
    import pandas as pd

    frame = pd.DataFrame({"a": [1.0, 2.0]})
    result = ns["calc_factor"]({"hfq_closes": frame})
    assert result.equals(frame)

    os.environ["FACTOR_DATA_DOMAINS"] = "pv,minutes"
    joint_module = _render_expression_based_module(_task("safe_div(last(returns), vwaps)"))
    assert joint_module is not None
    assert '"last": last' in joint_module


def test_render_expression_based_module_sanitizes_infinite_outputs():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv"
    module = _render_expression_based_module(_task("hfq_closes / (hfq_closes - hfq_closes)"))
    assert module is not None
    assert "DIVIDE(" in module
    assert "SUBTRACT(" in module
    ns = {}
    exec(module, ns)
    import pandas as pd

    frame = pd.DataFrame({"a": [1.0, 2.0]})
    result = ns["calc_factor"]({"hfq_closes": frame})
    assert result.isin([float("inf"), float("-inf")]).sum().sum() == 0
    assert result.isna().sum().sum() == 2


def test_render_expression_based_module_rewrites_daily_infix_to_aligned_ops():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv"
    module = _render_expression_based_module(_task("hfq_closes + vwaps"))
    assert module is not None
    assert 'expr = """ADD(hfq_closes, vwaps)"""' in module
    ns = {}
    exec(module, ns)
    import pandas as pd

    left = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    right = pd.DataFrame({"A": [10.0, 20.0]})
    result = ns["calc_factor"]({"hfq_closes": left, "vwaps": right})

    assert list(result.columns) == ["A", "B"]
    assert result["A"].tolist() == [11.0, 22.0]
    assert result["B"].tolist() == [13.0, 24.0]


def test_render_expression_based_module_rewrites_daily_logic_to_aligned_ops():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv"
    module = _render_expression_based_module(
        _task("(hfq_closes > vwaps) & (turnovers >= 0) ? hfq_closes : vwaps")
    )
    assert module is not None
    assert 'expr = """WHERE(AND(GT(hfq_closes, vwaps), GE(turnovers, 0)), hfq_closes, vwaps)"""' in module
    ns = {}
    exec(module, ns)
    import pandas as pd

    close = pd.DataFrame({"A": [2.0, 1.0], "B": [5.0, 1.0]})
    vwaps = pd.DataFrame({"A": [1.0, 2.0]})
    turnovers = pd.DataFrame({"A": [1.0, 1.0], "B": [1.0, 1.0]})
    result = ns["calc_factor"]({"hfq_closes": close, "vwaps": vwaps, "turnovers": turnovers})

    assert result["A"].tolist() == [2.0, 2.0]
    assert result["B"].tolist() == [5.0, 2.0]


def test_render_expression_based_module_executes_if_else_alias():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv"
    module = _render_expression_based_module(_task("if_else(GT(hfq_closes, vwaps), hfq_closes, vwaps)"))
    assert module is not None
    assert 'expr = """if_else(GT(hfq_closes, vwaps), hfq_closes, vwaps)"""' in module
    assert '"if_else": WHERE' in module
    ns = {}
    exec(module, ns)
    import pandas as pd

    close = pd.DataFrame({"A": [2.0, 1.0], "B": [5.0, 1.0]})
    vwaps = pd.DataFrame({"A": [1.0, 2.0]})
    result = ns["calc_factor"]({"hfq_closes": close, "vwaps": vwaps})

    assert result["A"].tolist() == [2.0, 2.0]
    assert result["B"].tolist() == [5.0, 2.0]


def test_render_expression_based_module_daily_aliases_ts_regression_to_slope_semantics():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv"
    module = _render_expression_based_module(_task("ts_regression(hfq_closes, vwaps, 20)"))
    assert module is not None
    assert '"ts_regression": lambda df1, df2, p=5: ts_reg(df1, df2, p, rettype=1)' in module


def test_render_expression_based_module_normalizes_ts_reg_keyword_argument():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv"
    module = _render_expression_based_module(_task("ts_reg(turnovers, hfq_closes, 20, rettype=0)"))
    assert module is not None
    assert 'expr = """ts_reg(turnovers, hfq_closes, 20, 0)"""' in module


def test_render_expression_based_module_returns_none_for_unparseable_pv_expression():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv"
    module = _render_expression_based_module(_task("!!! not an expression !!!"))
    assert module is None


def test_parsing_strategy_invalid_pv_expression_returns_feedback_module():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv"
    strategy = FactorParsingStrategy(None, FACTOR_COSTEER_SETTINGS)
    module = strategy.implement_one_task(
        _task("ts_corr(cs_rank(hfq_closes), volumes, 20)"),
        queried_knowledge=None,
    )
    assert "raise_expression_render_failure" in module
    assert "invalid expression" in module


def test_parsing_strategy_daily_failure_module_explains_none_literal():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv"
    strategy = FactorParsingStrategy(None, FACTOR_COSTEER_SETTINGS)
    module = strategy.implement_one_task(
        _task("cs_zscore(ts_reg(volumes, ts_return(hfq_closes, 1), 20, min_periods=None, rettype=0))"),
        queried_knowledge=None,
    )

    assert "raise_expression_render_failure" in module
    assert "unsupported fields: None" in module
    assert "remove optional keyword arguments or None literals" in module


def test_render_expression_based_module_returns_none_for_uncompilable_joint_expression():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv,minutes"
    # volumes is shared; top-level bare volumes on the minute side can't be resolved
    # deterministically without a daily anchor, so the compiler returns None.
    module = _render_expression_based_module(_task("volumes + 1"))
    assert module is None


def test_render_structured_minute_failure_module_shape():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv,minutes"
    module = _render_structured_minute_failure_module(
        _task("bad_expression"),
        "compiler_error_message",
    )
    assert 'level": "minutes"' in module
    assert "_raise_structured_render_failure" in module
    assert "compiler_error_message" in module
    assert 'global_name != "_raise_structured_render_failure"' in module


def test_joint_routing_note_documents_pct_and_post_reduce_daily_semantics():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv,minutes"
    note = _build_joint_pv_minutes_routing_note(_task("safe_div(ts_mean(returns, 15), vwaps)"))
    assert "use `ts_return(field, lag)` only for lagged same-field returns" in note
    assert "use `pct(a, b)` only for same-time tensor ratio `a / b - 1`" in note
    assert "uses regular daily semantics, not intraday minute semantics" in note


def test_minute_runtime_preflight_feedback_rejects_stage_mismatched_daily_ops():
    os.environ["FACTOR_DATA_DOMAINS"] = "minutes"
    feedback = _build_minute_runtime_preflight_feedback(
        factor_name="demo_factor",
        expression="ts_delay(ts_corr(winsorize1(amounts, 0.3), winsorize1(highs - lows, 0.3), 240), 2)",
        called_operators=["ts_delay", "ts_corr", "winsorize1"],
        active_domains=("minutes",),
    )
    assert feedback is not None
    assert "cannot be compiled into the structured minute runtime" in feedback
    assert "winsorize1" in feedback
    assert "ts_delay" in feedback


def test_minute_runtime_preflight_feedback_rejects_cross_day_minute_lag():
    os.environ["FACTOR_DATA_DOMAINS"] = "minutes"
    feedback = _build_minute_runtime_preflight_feedback(
        factor_name="demo_factor",
        expression="cs_zscore(ts_return(closes, 60480))",
        called_operators=["cs_zscore", "ts_return"],
        active_domains=("minutes",),
    )
    assert feedback is not None
    assert "cross-day sized minute lags" in feedback
    assert "ts_return(closes, 60480)" in feedback


def test_minute_runtime_preflight_feedback_accepts_compilable_minute_expression():
    os.environ["FACTOR_DATA_DOMAINS"] = "minutes"
    feedback = _build_minute_runtime_preflight_feedback(
        factor_name="demo_factor",
        expression="ts_delay(ts_corr(amounts, highs - lows, 240), 2)",
        called_operators=["ts_delay", "ts_corr"],
        active_domains=("minutes",),
    )
    assert feedback is None
