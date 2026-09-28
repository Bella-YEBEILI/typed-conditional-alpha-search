from quantaalpha.factors.alignment.preflight_guard import validate_expression_against_registry
from quantaalpha.factors.alignment.registry import get_allowed_field_names, get_allowed_operator_names


def test_validate_expression_against_registry_rejects_fractional_winsorize1_parameter():
    result = validate_expression_against_registry(
        "winsorize1(ts_mean(turnovers, 20), 0.3)",
        allowed_operators=get_allowed_operator_names(("pv",)),
        allowed_fields=get_allowed_field_names(("pv",)),
    )
    assert result["ok"] is False
    assert any(
        "winsorize1 arg#2 is an n-sigma clip width" in message
        for message in result["unsupported_operators"]
    )


def test_validate_expression_against_registry_allows_logical_function_operators():
    allowed_ops = get_allowed_operator_names(("pv",))
    for op in [
        "WHERE",
        "where",
        "IF_ELSE",
        "if_else",
        "ifelse",
        "AND",
        "OR",
        "GT",
        "LT",
        "GE",
        "LE",
        "EQ",
        "NE",
        "gt",
        "lt",
        "ge",
        "le",
        "eq",
        "ne",
    ]:
        assert op in allowed_ops

    result = validate_expression_against_registry(
        "WHERE(AND(GT(hfq_closes, vwaps), GE(turnovers, 0)), hfq_closes, vwaps)",
        allowed_operators=allowed_ops,
        allowed_fields=get_allowed_field_names(("pv",)),
    )
    assert result["ok"] is True

    for expression in [
        "IF_ELSE(GT(hfq_closes, vwaps), hfq_closes, vwaps)",
        "if_else(GT(hfq_closes, vwaps), hfq_closes, vwaps)",
        "ifelse(GT(hfq_closes, vwaps), hfq_closes, vwaps)",
    ]:
        result = validate_expression_against_registry(
            expression,
            allowed_operators=allowed_ops,
            allowed_fields=get_allowed_field_names(("pv",)),
        )
        assert result["ok"] is True


def test_pv_field_allowlist_uses_actual_quant_data_names():
    allowed_fields = get_allowed_field_names(("pv",))
    assert "turnovers" in allowed_fields
    assert "vwaps" in allowed_fields
    assert "turnover" not in allowed_fields
    assert "vwap" not in allowed_fields


def test_validate_expression_against_registry_rejects_bad_cs_zscore_arity():
    result = validate_expression_against_registry(
        "cs_zscore(hfq_closes, 20)",
        allowed_operators=get_allowed_operator_names(("pv",)),
        allowed_fields=get_allowed_field_names(("pv",)),
    )

    assert result["ok"] is False
    assert any("cs_zscore expects 1 args, got 2" in message for message in result["unsupported_operators"])


def test_validate_expression_against_registry_allows_multireg_list_arguments():
    result = validate_expression_against_registry(
        "ts_decay_linear(ts_multireg([safe_div(hfq_closes - hfq_opens, hfq_opens), turnovers], ts_return(hfq_closes, 1), 20, rettype=0), 10)",
        allowed_operators=get_allowed_operator_names(("pv",)),
        allowed_fields=get_allowed_field_names(("pv",)),
    )

    assert result["ok"] is True
    assert result["called_operators"] == [
        "safe_div",
        "ts_decay_linear",
        "ts_multireg",
        "ts_return",
    ]


def test_validate_expression_against_registry_allows_daily_safe_div_arity_in_joint_domains(monkeypatch):
    monkeypatch.setenv("FACTOR_DATA_DOMAINS", "pv,minutes")
    result = validate_expression_against_registry(
        "safe_div(hfq_closes - hfq_opens, hfq_opens)",
        allowed_operators=get_allowed_operator_names(("pv", "minutes")),
        allowed_fields=get_allowed_field_names(("pv", "minutes")),
    )

    assert result["ok"] is True


def test_validate_expression_against_registry_allows_template_arithmetic_aliases():
    result = validate_expression_against_registry(
        "ts_decay_linear(cs_rank(DIVIDE(ts_return(hfq_closes, 5), ADD(ts_std(ts_return(hfq_closes, 1), 20), 1e-08))), 10)",
        allowed_operators=get_allowed_operator_names(("pv",)),
        allowed_fields=get_allowed_field_names(("pv",)),
    )

    assert result["ok"] is True


def test_expr_parser_generated_internal_ops_are_registry_allowed():
    allowed_ops = get_allowed_operator_names(("pv", "minutes"))
    for op in [
        "ADD",
        "SUBTRACT",
        "MULTIPLY",
        "DIVIDE",
        "GT",
        "LT",
        "GE",
        "LE",
        "EQ",
        "NE",
        "AND",
        "OR",
        "WHERE",
    ]:
        assert op in allowed_ops


def test_validate_expression_against_registry_rejects_two_arg_cs_neutralize_with_list_feedback():
    result = validate_expression_against_registry(
        "ema3(cs_neutralize(ts_return(hfq_closes, 1), [safe_div(hfq_closes - hfq_opens, hfq_opens), turnovers]), 0.067)",
        allowed_operators=get_allowed_operator_names(("pv",)),
        allowed_fields=get_allowed_field_names(("pv",)),
    )

    assert result["ok"] is False
    assert not any("<parse_error:" in message for message in result["unsupported_operators"])
    assert any("cs_neutralize expects 1 args, got 2" in message for message in result["unsupported_operators"])
