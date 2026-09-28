import json
from types import SimpleNamespace

from quantaalpha.factors.proposal import AlphaAgentHypothesis2FactorExpression


def test_factor_proposal_records_requested_and_proposed_counts():
    converter = AlphaAgentHypothesis2FactorExpression.__new__(AlphaAgentHypothesis2FactorExpression)
    converter.target_factor_count = 3
    payload = {
        "factor_a": {"description": "a", "formulation": "a", "expression": "ts_mean(close, 5)"},
        "factor_b": {"description": "b", "formulation": "b", "expression": "ts_mean(close, 10)"},
    }

    exp = converter.convert_response(json.dumps(payload), SimpleNamespace(hist=[]))

    assert exp.factor_generation_requested_count == 3
    assert exp.factor_generation_proposed_count == 2
