import os

from quantaalpha.factors.alignment.registry import get_operator_semantic_notes
from quantaalpha.factors.alignment.registry import get_prompt_operator_names
from quantaalpha.factors.coder.prompt_utils import build_runtime_registry_constraints
from quantaalpha.factors.proposal import _build_function_lib_description


def test_operator_semantic_notes_cover_sensitive_regression_and_clipping_ops():
    notes = get_operator_semantic_notes(("pv", "minutes"))
    joined = "\n".join(notes)
    assert "`ts_regression(x, y, n)` returns the rolling regression slope/beta only" in joined
    assert "`ts_regression2(x, y, n, min_periods=None, rettype=0)`" in joined
    assert "`cs_multireg([x1, x2], y, min_cs=30)`" in joined
    assert "`ts_multireg([x1, x2], y, n, min_periods=None, rettype=0)`" in joined
    assert "`cs_neutralize(df)` accepts one argument only" in joined
    assert "`cs_zscore(df)` and `zscore(values)` accept one argument only" in joined
    assert "`winsorize1(df, n)` clips cross-sectional values at `mean +/- n * std`" in joined
    assert "`bucket(df, n)` is a per-date cross-sectional bucketing operator" in joined
    assert "Conditional logic is supported" in joined
    assert "`IF_ELSE(cond, true_value, false_value)`" in joined
    assert "`if_else(...)`" in joined
    assert "Logical composition is supported" in joined
    assert "Comparison operators are supported" in joined


def test_runtime_registry_constraints_include_operator_semantic_notes():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv,minutes"
    constraints = build_runtime_registry_constraints()
    assert "Operator semantic notes:" in constraints
    assert "All operators in the allowlist are equally available" in constraints
    assert "`ts_regression(x, y, n)` returns the rolling regression slope/beta only" in constraints
    assert "`winsorize1(df, n)` clips cross-sectional values" in constraints
    assert "`cs_zscore(df)` and `zscore(values)` accept one argument only" in constraints
    assert "Conditional logic is supported" in constraints
    assert "Comparison operators are supported" in constraints
    assert "ADD(" not in constraints
    assert "DIVIDE(" not in constraints


def test_proposal_function_lib_description_includes_operator_semantic_notes():
    os.environ["FACTOR_DATA_DOMAINS"] = "pv,minutes"
    description = _build_function_lib_description("Base description placeholder.")
    assert "Operator semantic notes:" in description
    assert "`ts_reg(x, y, n, min_periods=None, rettype=0)` returns exactly one panel" in description
    assert "`cs_multireg([x1, x2], y, min_cs=30)` returns the cross-sectional residual" in description
    assert "`winsorize2(df, lower_bound, upper_bound)` clips cross-sectional values by lower/upper quantiles." in description
    assert "ADD(" not in description
    assert "DIVIDE(" not in description


def test_prompt_operator_names_hide_internal_template_aliases():
    prompt_ops = get_prompt_operator_names(("pv", "minutes"))
    assert "ADD" not in prompt_ops
    assert "SUBTRACT" not in prompt_ops
    assert "MULTIPLY" not in prompt_ops
    assert "DIVIDE" not in prompt_ops
    assert "add" in prompt_ops
    assert "div" in prompt_ops
