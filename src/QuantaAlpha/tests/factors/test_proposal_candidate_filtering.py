from quantaalpha.factors import data_domains as factor_data_domains
from quantaalpha.factors.alignment.preflight_guard import validate_expression_against_registry
from quantaalpha.factors.proposal import (
    _filter_similar_factor_payload,
    _get_domain_proposal_policy,
)
from quantaalpha.factors.regulator.factor_regulator import FactorRegulator


def _regulator(domains):
    return FactorRegulator(
        factor_zoo_path=None,
        duplication_threshold=8,
        symbol_length_threshold=300,
        base_features_threshold=6,
        domains=domains,
    )


def test_minute_candidate_filter_rejects_uncompilable_expression(monkeypatch):
    monkeypatch.setattr(
        "quantaalpha.factors.proposal.compile_expression_to_minute_spec",
        lambda task, active_domains=None: None,
    )
    monkeypatch.setattr(factor_data_domains, "resolve_factor_domains", lambda: ("minutes",))

    payload = {
        "MinuteBad": {
            "description": "demo",
            "formulation": "demo",
            "expression": "ts_mean(returns, 15)",
        }
    }

    accepted, skipped = _filter_similar_factor_payload(payload, _regulator(("minutes",)))

    assert accepted == {}
    assert skipped == ["MinuteBad: minute_compile_failed"]


def test_joint_candidate_filter_requires_pv_and_minute_domains(monkeypatch):
    monkeypatch.setattr(
        "quantaalpha.factors.proposal.compile_expression_to_minute_spec",
        lambda task, active_domains=None: {"minute_features": [{"field": "returns"}]},
    )
    monkeypatch.setattr(factor_data_domains, "resolve_factor_domains", lambda: ("pv", "minutes"))

    payload = {
        "JointMissingPv": {
            "description": "demo",
            "formulation": "demo",
            "expression": "ts_mean(returns, 15)",
        }
    }

    accepted, skipped = _filter_similar_factor_payload(payload, _regulator(("pv", "minutes")))

    assert accepted == {}
    assert skipped == ["JointMissingPv: joint_domain_missing"]


def test_candidate_filter_rejects_reserved_and_already_accepted_factor_names(monkeypatch):
    monkeypatch.setattr(factor_data_domains, "resolve_factor_domains", lambda: ("pv",))

    payload = {
        "OldFactor": {
            "description": "demo",
            "formulation": "demo",
            "expression": "ts_mean(hfq_closes, 5)",
        },
        "AcceptedFactor": {
            "description": "demo",
            "formulation": "demo",
            "expression": "ts_mean(hfq_opens, 5)",
        },
        "FreshFactor": {
            "description": "demo",
            "formulation": "demo",
            "expression": "ts_mean(turnovers, 5)",
        },
    }

    accepted, skipped = _filter_similar_factor_payload(
        payload,
        _regulator(("pv",)),
        accepted_payload={
            "AcceptedFactor": {
                "description": "accepted",
                "formulation": "accepted",
                "expression": "ts_mean(vwaps, 5)",
            }
        },
        reserved_factor_names={"OldFactor"},
    )

    assert list(accepted) == ["FreshFactor"]
    assert skipped[:2] == [
        "OldFactor: factor_name_already_used",
        "AcceptedFactor: factor_name_already_used",
    ]


def test_minute_policy_feedback_names_invalid_field_and_allowed_fields():
    regulator = _regulator(("minutes",))
    policy = _get_domain_proposal_policy(("minutes",))
    expression = "volume / ts_mean(volume, 30) * ts_std(returns, 30)"

    registry_check = validate_expression_against_registry(
        expression,
        allowed_operators=regulator.allowed_functions,
        allowed_fields=regulator.allowed_fields,
    )
    feedback = policy.build_registry_feedback(
        factor_name="VolumeRatio_Volatility_Product",
        expression=expression,
        registry_check=registry_check,
        allowed_operators=regulator.allowed_functions,
        allowed_fields=regulator.allowed_fields,
    )

    assert "Invalid field(s): volume" in feedback
    assert "Allowed minute fields" in feedback
    assert "volumes" in feedback
    assert "Regenerate the expression using exact field/operator names" in feedback


def test_joint_policy_feedback_explains_unsupported_sign_rewrite():
    regulator = _regulator(("pv", "minutes"))
    policy = _get_domain_proposal_policy(("pv", "minutes"))
    expression = "mul(sign(returns), ts_mean(hfq_closes, 20))"

    registry_check = validate_expression_against_registry(
        expression,
        allowed_operators=regulator.allowed_functions,
        allowed_fields=regulator.allowed_fields,
    )
    feedback = policy.build_registry_feedback(
        factor_name="Order_Flow_Imbalance_Momentum",
        expression=expression,
        registry_check=registry_check,
        allowed_operators=regulator.allowed_functions,
        allowed_fields=regulator.allowed_fields,
    )

    assert "Unsupported operator/function(s): sign" in feedback
    assert "if_else(gt(x, 0), 1, if_else(lt(x, 0), -1, 0))" in feedback
    assert "Allowed daily PV fields" in feedback
    assert "Allowed minute fields" in feedback


def test_minute_policy_rejects_raw_division_with_safe_div_repair_hint():
    policy = _get_domain_proposal_policy(("minutes",))
    expression = "turnovers / ts_mean(turnovers, 30) * ts_std(returns, 30)"

    feedback = policy.build_preflight_feedback(
        factor_name="TurnoverRatio_Volatility_Product",
        expression=expression,
    )

    assert "Numerical safety issue" in feedback
    assert "safe_div(numerator, denominator, 1e-8)" in feedback
    assert "raw `/`" in feedback
