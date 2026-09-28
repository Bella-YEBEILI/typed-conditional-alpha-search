from quantaalpha.coder.costeer.evaluators import CoSTEERSingleFeedback, _summarize_feedback_failure
from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
from quantaalpha.factors.coder.evaluators import FactorEvaluatorForCoder


class _Workspace:
    def __init__(self, code: str):
        self.code_dict = {"factor.py": code}


def test_ast_regularization_reads_triple_quoted_template_expr():
    evaluator = FactorEvaluatorForCoder(scen=None)
    ok, feedback = evaluator.check_ast_regularization(
        _Workspace(
            '''
expr = """cs_zscore(hfq_closes, 20)"""
'''
        )
    )

    assert ok is False
    assert "cs_zscore expects 1 args, got 2" in feedback


def test_ast_regularization_reads_joint_original_factor_expression():
    evaluator = FactorEvaluatorForCoder(scen=None)
    ok, feedback = evaluator.check_ast_regularization(
        _Workspace(
            '''
ORIGINAL_FACTOR_EXPRESSION = """cs_zscore(safe_div(hfq_closes, ts_mean(vwaps, 240)) - 1, 0.05)"""
FINAL_EXPRESSION = """cs_zscore(safe_div(data_ctx["hfq_closes"], vwaps_mean_240) - 1, 0.05)"""
'''
        )
    )

    assert ok is False
    assert "cs_zscore expects 1 args, got 2" in feedback


def test_ast_regularization_reports_daily_render_failure_message():
    evaluator = FactorEvaluatorForCoder(scen=None)
    ok, feedback = evaluator.check_ast_regularization(
        _Workspace(
            '''
expr = """raise_expression_render_failure()"""
DAILY_RENDER_FAILURE_MESSAGE = "invalid expression: unsupported field None; remove keyword args or None literals"

def raise_expression_render_failure():
    raise RuntimeError(DAILY_RENDER_FAILURE_MESSAGE)
'''
        )
    )

    assert ok is False
    assert "unsupported field None" in feedback
    assert "remove keyword args" in feedback
    assert "Expression cannot be parsed: raise_expression_render_failure()" not in feedback


def test_feedback_summary_prefers_actionable_execution_failure():
    feedback = CoSTEERSingleFeedback(
        final_feedback=(
            "Deterministic final decision failed because execution feedback still contains a hard failure.\n"
            "Execution feedback summary: Execution succeeded without error."
        ),
        execution_feedback=(
            "Execution succeeded without error.\n"
            "Deterministic TQ bridge fallback failed.: invalid expression: "
            "operators=['<semantic:cs_zscore expects 1 args, got 2>'] fields=[]\n"
            "Expected output file not found."
        ),
    )

    summary = _summarize_feedback_failure(feedback)

    assert "cs_zscore expects 1 args, got 2" in summary


def test_factor_coster_default_symbol_length_threshold_is_400():
    assert FACTOR_COSTEER_SETTINGS.symbol_length_threshold == 400


def test_ast_regularization_treats_symbol_length_as_warning_not_failure(monkeypatch):
    evaluator = FactorEvaluatorForCoder(scen=None)
    monkeypatch.setattr(evaluator.factor_regulator, "symbol_length_threshold", 10)

    ok, feedback = evaluator.check_ast_regularization(
        _Workspace(
            '''
expr = "ts_mean(hfq_closes, 5)"
'''
        )
    )

    assert ok is True
    assert "Symbol Length (SL) Warning" in feedback
    assert "exceeds soft threshold (10)" in feedback


def test_ast_regularization_accepts_multireg_list_arguments():
    evaluator = FactorEvaluatorForCoder(scen=None)
    ok, feedback = evaluator.check_ast_regularization(
        _Workspace(
            '''
expr = "ts_decay_linear(ts_multireg([safe_div(hfq_closes - hfq_opens, hfq_opens), turnovers], ts_return(hfq_closes, 1), 20, rettype=0), 10)"
'''
        )
    )

    assert ok is True
    assert "parse_error" not in feedback


def test_ast_regularization_reports_cs_neutralize_arity_instead_of_parse_error():
    evaluator = FactorEvaluatorForCoder(scen=None)
    ok, feedback = evaluator.check_ast_regularization(
        _Workspace(
            '''
expr = "ema3(cs_neutralize(ts_return(hfq_closes, 1), [safe_div(hfq_closes - hfq_opens, hfq_opens), turnovers]), 0.067)"
'''
        )
    )

    assert ok is False
    assert "cs_neutralize expects 1 args, got 2" in feedback
    assert "parse_error" not in feedback


def test_ast_regularization_accepts_daily_template_arithmetic_aliases():
    evaluator = FactorEvaluatorForCoder(scen=None)
    ok, feedback = evaluator.check_ast_regularization(
        _Workspace(
            '''
expr = "ts_decay_linear(cs_rank(DIVIDE(ts_return(hfq_closes, 5), ADD(ts_std(ts_return(hfq_closes, 1), 20), 1e-08))), 10)"
'''
        )
    )

    assert ok is True
    assert "Operator Validation Failed" not in feedback


def test_ast_regularization_accepts_if_else_aliases():
    evaluator = FactorEvaluatorForCoder(scen=None)
    for expression in [
        "IF_ELSE(GT(hfq_closes, vwaps), hfq_closes, vwaps)",
        "if_else(GT(hfq_closes, vwaps), hfq_closes, vwaps)",
        "ifelse(GT(hfq_closes, vwaps), hfq_closes, vwaps)",
    ]:
        ok, feedback = evaluator.check_ast_regularization(
            _Workspace(
                f'''
expr = "{expression}"
'''
            )
        )

        assert ok is True
        assert "Operator Validation Failed" not in feedback
