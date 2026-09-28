import quantaalpha.backtest.minute_ops  # noqa: F401
from quantaalpha.backtest.minute_tools import MinuteFactorEngine
from quantaalpha.factors.alignment.registry import get_operator_arity_constraints
from quantaalpha.factors.coder.minute_op_registry import (
    get_minute_op_registry,
    get_minute_op_spec,
    get_registered_minute_ops_missing_metadata,
    get_minute_operator_arity_constraints,
    get_tensor_output_minute_ops,
    get_vector_output_minute_ops,
)


def test_every_registered_minute_op_has_shape_metadata():
    assert get_registered_minute_ops_missing_metadata() == set()
    registry = get_minute_op_registry()
    assert set(registry.keys()) == set(MinuteFactorEngine._OPS.keys())


def test_minute_op_registry_distinguishes_tensor_and_vector_ops():
    assert "add" in get_tensor_output_minute_ops()
    assert "rank" in get_tensor_output_minute_ops()
    assert "mean" in get_vector_output_minute_ops()
    assert "corr" in get_vector_output_minute_ops()


def test_minute_op_spec_exposes_arity_and_param_order():
    corr = get_minute_op_spec("corr")
    assert corr is not None
    assert corr.arity == 2
    assert corr.input_kinds == ("tensor", "tensor")
    assert corr.param_order == ("shift",)
    assert corr.output_kind == "vector"

    add = get_minute_op_spec("add")
    assert add is not None
    assert add.arity == 2
    assert add.input_kinds == ("tensor", "tensor")
    assert add.output_kind == "tensor"


def test_alignment_minute_arity_constraints_read_same_registry():
    minute_constraints = get_minute_operator_arity_constraints()
    runtime_constraints = get_operator_arity_constraints(("minutes",))
    for name, expected in minute_constraints.items():
        assert runtime_constraints[name] == expected
