import os

from quantaalpha.factors.coder.evolving_strategy import _normalize_structured_minute_spec
from quantaalpha.factors.coder.factor import FactorTask
from quantaalpha.factors.coder.minute_spec_compiler import _CompileState, compile_expression_to_minute_spec


def _compile(expr: str, domains: tuple[str, ...]):
    os.environ["FACTOR_DATA_DOMAINS"] = ",".join(domains)
    task = FactorTask("demo_factor", "demo", "demo", factor_expression=expr)
    spec = compile_expression_to_minute_spec(task, active_domains=domains)
    assert spec is not None
    normalized = _normalize_structured_minute_spec(spec, target_task=task)
    return normalized


def test_compile_pure_minute_snapshot_return():
    normalized = _compile("ts_return(returns, 30)", ("minutes",))
    assert normalized["data_needed"] == []
    assert normalized["final_expression"] == "returns_return_30"
    assert normalized["minute_features"][0]["kind"] == "snapshot_return"


def test_compile_pure_minute_window_expression():
    normalized = _compile("safe_div(ts_mean(turnovers, 15), ts_std(vwaps, 20))", ("minutes",))
    assert normalized["data_needed"] == []
    assert normalized["final_expression"] == "safe_div(turnovers_mean_15, vwaps_std_20)"
    assert {feature["name"] for feature in normalized["minute_features"]} == {
        "turnovers_mean_15",
        "vwaps_std_20",
    }


def test_compile_state_renames_duplicate_feature_names_for_distinct_signatures():
    state = _CompileState(minute_fields={"returns"}, daily_fields=set())
    first_name = state.add_feature({"name": "returns_mean_20", "kind": "window"}, ("mean", "returns", 20))
    second_name = state.add_feature({"name": "returns_mean_20", "kind": "window"}, ("mean", "returns", 30))

    assert first_name == "returns_mean_20"
    assert second_name != first_name
    assert len({feature["name"] for feature in state.features}) == 2


def test_compile_joint_expression_keeps_daily_anchor():
    normalized = _compile("safe_div(ts_mean(returns, 15), vwaps)", ("pv", "minutes"))
    assert normalized["data_needed"] == ["vwaps"]
    assert normalized["final_expression"] == 'safe_div(returns_mean_15, data_ctx["vwaps"])'
    assert normalized["minute_features"][0]["field"] == "returns"


def test_compile_registered_minute_operator_to_engine_run():
    normalized = _compile("corr(returns, turnovers, 1)", ("minutes",))
    feature = normalized["minute_features"][0]
    assert feature["kind"] == "engine_run"
    assert feature["operator"] == "corr"
    assert feature["operator_params"] == {"shift": 1}


def test_compile_engine_run_with_unary_preprocess():
    normalized = _compile("corr(abs(returns), turnovers, 1)", ("minutes",))
    feature = normalized["minute_features"][0]
    assert feature["kind"] == "engine_run"
    assert feature["inputs"] == ["returns", "turnovers"]
    assert feature["preprocess"] == ["abs", None]
    assert feature["operator"] == "corr"


def test_compile_window_engine_from_raw_minute_operator():
    normalized = _compile("mean(add(returns, turnovers), 15)", ("minutes",))
    feature = normalized["minute_features"][0]
    assert feature["kind"] == "window_engine"
    assert feature["operator"] == "add"
    assert feature["aggregation"] == "mean"
    assert feature["window"] == 15
    assert feature["inputs"] == ["returns", "turnovers"]


def test_compile_joint_window_engine_keeps_daily_anchor():
    normalized = _compile("safe_div(mean(abs(returns), 15), vwaps)", ("pv", "minutes"))
    feature = normalized["minute_features"][0]
    assert feature["kind"] == "window_engine"
    assert feature["operator"] == "abs"
    assert normalized["data_needed"] == ["vwaps"]
    assert normalized["final_expression"] == 'safe_div(mean_abs_returns_15, data_ctx["vwaps"])'


def test_joint_routes_shared_field_volumes_to_minute_side():
    normalized = _compile("safe_div(ts_mean(volumes, 15), hfq_closes)", ("pv", "minutes"))
    assert normalized["data_needed"] == ["hfq_closes"]
    assert normalized["final_expression"] == 'safe_div(volumes_mean_15, data_ctx["hfq_closes"])'
    assert normalized["minute_features"][0]["field"] == "volumes"


def test_joint_allows_raw_minute_ohlc_inside_minute_subtree():
    normalized = _compile("safe_div(ts_mean(opens, 15), hfq_closes)", ("pv", "minutes"))
    assert normalized["data_needed"] == ["hfq_closes"]
    assert normalized["final_expression"] == 'safe_div(opens_mean_15, data_ctx["hfq_closes"])'
    assert normalized["minute_features"][0]["field"] == "opens"


def test_joint_routes_shared_field_volumes_to_daily_side_when_raw():
    normalized = _compile("safe_div(ts_mean(returns, 15), volumes)", ("pv", "minutes"))
    assert normalized["data_needed"] == ["volumes"]
    assert normalized["final_expression"] == 'safe_div(returns_mean_15, data_ctx["volumes"])'
    assert normalized["minute_features"][0]["field"] == "returns"


def test_joint_accepts_singular_daily_shared_field_aliases_and_publishes_plural_fields():
    normalized = _compile("safe_div(ts_mean(returns, 15), vwap)", ("pv", "minutes"))
    assert normalized["data_needed"] == ["vwaps"]
    assert normalized["final_expression"] == 'safe_div(returns_mean_15, data_ctx["vwaps"])'
    assert normalized["original_expression"] == "safe_div(ts_mean(returns, 15), vwaps)"
    assert normalized["minute_features"][0]["field"] == "returns"


def test_joint_rejects_raw_daily_ohlc():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv,minutes"
    task = FactorTask("demo_factor", "demo", "demo", factor_expression="safe_div(ts_mean(returns, 15), closes)")
    spec = compile_expression_to_minute_spec(task, active_domains=("pv", "minutes"))
    assert spec is None


def test_joint_requires_minute_exclusive_field():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv,minutes"
    task = FactorTask("demo_factor", "demo", "demo", factor_expression="hfq_closes + vwaps")
    spec = compile_expression_to_minute_spec(task, active_domains=("pv", "minutes"))
    assert spec is None


def test_pure_pv_expression_is_not_compiled_in_minutes_mode():
    os.environ["FACTOR_DATA_DOMAINS"] = "minutes"
    task = FactorTask("demo_factor", "demo", "demo", factor_expression="hfq_closes + 1")
    spec = compile_expression_to_minute_spec(task, active_domains=("minutes",))
    assert spec is None


def test_pv_only_mode_never_compiles():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv"
    task = FactorTask("demo_factor", "demo", "demo", factor_expression="ts_return(hfq_closes, 5)")
    spec = compile_expression_to_minute_spec(task, active_domains=("pv",))
    assert spec is None


def test_compile_snapshot_only():
    os.environ["FACTOR_DATA_DOMAINS"] = "minutes"
    task = FactorTask("demo_factor", "demo", "demo", factor_expression="closes")
    spec = compile_expression_to_minute_spec(task, active_domains=("minutes",))
    assert spec is not None
    feature = spec["minute_features"][0]
    assert feature["kind"] == "snapshot"
    assert feature["field"] == "closes"


def test_compile_intraday_pct_aggregate():
    normalized = _compile("ts_mean(pct(returns, 5), 15)", ("minutes",))
    feature = normalized["minute_features"][0]
    assert feature["kind"] == "intraday_pct_aggregate"
    assert feature["field"] == "returns"
    assert feature["lag"] == 5
    assert feature["aggregation"] == "mean"


def test_compile_registered_ts_rank_to_engine_run():
    normalized = _compile("ts_rank(turnovers, 20)", ("minutes",))
    feature = normalized["minute_features"][0]
    assert feature["kind"] == "engine_run"
    assert feature["operator"] == "ts_rank"
    assert feature["operator_params"] == {"n": 20}


def test_compile_registered_ts_corr_to_engine_run():
    normalized = _compile("ts_corr(returns, turnovers, 20)", ("minutes",))
    feature = normalized["minute_features"][0]
    assert feature["kind"] == "engine_run"
    assert feature["operator"] == "ts_corr"
    assert feature["operator_params"] == {"n": 20}


def test_compile_registered_ts_beta_to_engine_run():
    normalized = _compile("ts_beta(returns, turnovers, 20)", ("minutes",))
    feature = normalized["minute_features"][0]
    assert feature["kind"] == "engine_run"
    assert feature["operator"] == "ts_beta"
    assert feature["operator_params"] == {"n": 20}


def test_compile_composite_ts_beta_to_vector_expr():
    normalized = _compile(
        "ts_beta(safe_div(closes - vwaps, vwaps, 1e-8), closes, 60) * (ts_rank(turnovers, 20) > 0.9)",
        ("minutes",),
    )
    feature_names = {feature["name"] for feature in normalized["minute_features"]}
    assert normalized["minute_features"]
    assert normalized["final_expression"]
    assert feature_names


def test_compile_ts_regression_with_multi_input_tensor_subtree():
    normalized = _compile(
        "mul(ts_regression(closes, safe_div(mul(returns, volumes), volumes, 0.05), 240), bucket(ts_rank(volumes, 240), 3))",
        ("minutes",),
    )
    kinds = {feature["kind"] for feature in normalized["minute_features"]}
    assert "vector_expr" in kinds
    assert "engine_run" in kinds
    assert "3" in normalized["final_expression"]
    assert normalized["final_expression"].startswith("mul(")


def test_compile_ts_regression_with_broadcast_pointwise_inputs():
    # The LLM-generated shape that previously triggered the
    # "minute_spec_compiler could not handle the current expression shape"
    # warning. The inner `div(ts_mean(volumes, 20), volumes)` mixes a
    # 1D (stocks,) reduction with a 2D (minutes, stocks) tensor; numpy
    # broadcasts produce a 2D tensor, which flows as input into ts_regression.
    normalized = _compile(
        "extreme_rightmap(ts_zscore(ts_regression(div(ts_mean(volumes, 20), volumes), div(add(opens, closes), 2), 20), 20))",
        ("minutes",),
    )
    features = normalized["minute_features"]
    assert len(features) == 1
    assert features[0]["kind"] == "vector_expr"
    assert set(features[0]["inputs"]) == {"closes", "opens", "volumes"}
    assert '_qa_minute_op("ts_regression"' in features[0]["expression_code"]
    assert '_qa_minute_op("mean"' in features[0]["expression_code"]
    assert normalized["final_expression"].startswith("extreme_rightmap(ts_zscore(")


def test_compile_neg_wrapping_reduced_feature_stays_daily():
    # `neg(ts_regression(...))` keeps neg at the daily layer since the reducer
    # has already collapsed to a per-day vector and the numba `neg` kernel
    # rejects 1D inputs. The inner reducer becomes a vector_expr feature.
    normalized = _compile(
        "extreme_rightmap(ts_zscore(neg(ts_regression(div(ts_mean(volumes, 10), volumes), div(add(highs, lows), 2), 10)), 10))",
        ("minutes",),
    )
    features = normalized["minute_features"]
    assert len(features) == 1
    assert features[0]["kind"] == "vector_expr"
    feature_name = features[0]["name"]
    assert f"neg({feature_name})" in normalized["final_expression"]


def test_compile_where_function_inside_window_aggregation():
    # Prompt constraints allow conditional operators exactly as listed in the
    # runtime allowlist. The compiler must treat the function and ternary forms
    # as the same operator at shape inference and runtime rendering.
    normalized = _compile(
        "mean(WHERE(sub(closes, opens) > 0, sub(highs, closes), 0), 240)",
        ("minutes",),
    )
    features = normalized["minute_features"]
    assert len(features) == 1
    assert features[0]["kind"] == "vector_expr"
    assert "np.where(" in features[0]["expression_code"]
    assert '_qa_minute_op("mean"' in features[0]["expression_code"]
    assert "mean(" not in normalized["final_expression"]
    assert set(features[0]["inputs"]) == {"closes", "opens", "highs"}


def test_compile_if_else_alias_inside_window_aggregation():
    normalized = _compile(
        "ts_sum(IF_ELSE(returns > 0, volumes, 0), 90)",
        ("minutes",),
    )
    features = normalized["minute_features"]
    assert len(features) == 1
    assert features[0]["kind"] == "vector_expr"
    assert "np.where((returns > 0), volumes, 0)" in features[0]["expression_code"]
    assert '_qa_minute_op("sum"' in features[0]["expression_code"]
    assert "ts_sum(" not in normalized["final_expression"]
    assert "IF_ELSE" not in normalized["final_expression"]


def test_compile_two_arg_where_alias_defaults_false_to_zero():
    normalized = _compile(
        "ts_delay(where(returns > 0, 1), 1)",
        ("minutes",),
    )
    features = normalized["minute_features"]
    assert len(features) == 1
    assert features[0]["kind"] == "vector_expr"
    assert "np.where((returns > 0), 1, 0)" in features[0]["expression_code"]
    assert "where(" not in normalized["final_expression"]


def test_compile_complex_tensor_expression_collapses_to_daily_vector():
    normalized = _compile(
        "ts_corr(ts_sum(returns * volumes, 200), volumes, 20) * (volumes / ts_mean(volumes, 200))",
        ("minutes",),
    )
    assert normalized["data_needed"] == []
    assert any(feature["kind"] == "vector_expr" for feature in normalized["minute_features"])
    assert normalized["final_expression"]


def test_compile_vector_expr_renders_raw_division_with_safe_runtime_helper():
    normalized = _compile(
        "ts_mean(abs(turnovers / volumes - 1), 60)",
        ("minutes",),
    )

    vector_features = [
        feature for feature in normalized["minute_features"] if feature["kind"] == "vector_expr"
    ]
    assert len(vector_features) == 1
    expression_code = vector_features[0]["expression_code"]
    assert "_qa_minute_div(" in expression_code
    assert "turnovers / volumes" not in expression_code


def test_compile_joint_complex_tensor_subtree_with_daily_anchor():
    normalized = _compile(
        "ts_corr(ts_sum(returns * volumes, 200), volumes, 20) * (volumes / ts_mean(volumes, 200)) * safe_div(hfq_highs - hfq_lows, hfq_closes, 0.05)",
        ("pv", "minutes"),
    )
    assert {"hfq_closes", "hfq_highs", "hfq_lows"}.issubset(set(normalized["data_needed"]))
    assert normalized["minute_features"]
    assert "data_ctx" in normalized["final_expression"]


def test_compile_minutes_multireg_list_expression():
    normalized = _compile(
        "ts_multireg([ts_mean(returns, 20), ts_mean(turnovers, 20)], ts_return(closes, 1), 20, rettype=0)",
        ("minutes",),
    )

    assert normalized["data_needed"] == []
    assert len(normalized["minute_features"]) >= 3
    assert "ts_multireg([" in normalized["final_expression"]


def test_compile_joint_multireg_list_expression_with_daily_anchor():
    normalized = _compile(
        "ts_multireg([ts_mean(returns, 20), ts_mean(turnovers, 20)], hfq_closes, 20, rettype=0)",
        ("pv", "minutes"),
    )

    assert "hfq_closes" in normalized["data_needed"]
    assert normalized["minute_features"]
    assert "ts_multireg([" in normalized["final_expression"]
    assert 'data_ctx["hfq_closes"]' in normalized["final_expression"]


def test_compile_joint_distinguishes_shared_volume_field_by_context():
    normalized = _compile(
        "safe_div(ts_mean(volumes, 20), volumes)",
        ("pv", "minutes"),
    )

    assert normalized["data_needed"] == ["volumes"]
    assert normalized["minute_features"][0]["field"] == "volumes"
    assert normalized["final_expression"] == 'safe_div(volumes_mean_20, data_ctx["volumes"])'


def test_compile_minutes_final_expression_normalizes_zscore_to_daily_analysis_alias():
    normalized = _compile("zscore(ts_mean(returns, 20))", ("minutes",))

    assert normalized["minute_features"][0]["field"] == "returns"
    assert normalized["final_expression"] == "cs_zscore(returns_mean_20)"


def test_compile_joint_final_expression_normalizes_zscore_to_daily_analysis_alias():
    normalized = _compile(
        "mul(zscore(ts_mean(returns, 20)), hfq_closes)",
        ("pv", "minutes"),
    )

    assert normalized["data_needed"] == ["hfq_closes"]
    assert normalized["minute_features"][0]["field"] == "returns"
    assert normalized["final_expression"] == 'mul(cs_zscore(returns_mean_20), data_ctx["hfq_closes"])'
