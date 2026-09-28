import warnings

import numpy as np
import pandas as pd

from quantaalpha.backtest.analysis import WHERE, ts_corr, ts_multireg, ts_reg, ts_regression, ts_return


def test_ts_regression_skips_degenerate_windows_without_runtime_warnings():
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    columns = ["A", "B"]
    x = pd.DataFrame(
        [
            [1.0, np.inf],
            [np.nan, 2.0],
            [3.0, np.nan],
            [4.0, 5.0],
        ],
        index=index,
        columns=columns,
    )
    y = pd.DataFrame(
        [
            [2.0, 1.0],
            [np.nan, np.inf],
            [6.0, np.nan],
            [8.0, 10.0],
        ],
        index=index,
        columns=columns,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        slope, intercept, eps = ts_regression(x, y, 3)

    assert caught == []
    assert slope.shape == y.shape
    assert intercept.shape == y.shape
    assert eps.shape == y.shape


def test_ts_reg_broadcasts_series_and_scalar_inputs_to_panel_shape():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    y = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [2.0, 4.0, 6.0, 8.0, 10.0],
        },
        index=index,
    )

    beta_from_series = ts_reg(y["A"], y, 3, rettype=1)
    residual_from_scalar = ts_reg(1.0, y, 3, rettype=0)

    assert beta_from_series.shape == y.shape
    assert residual_from_scalar.shape == y.shape


def test_ts_corr_broadcasts_single_column_input_to_panel_shape():
    index = pd.date_range("2024-01-01", periods=5, freq="D")
    left = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]}, index=index, dtype=float)
    right = left[["A"]]

    result = ts_corr(left, right, 3)

    assert result.shape == left.shape


def test_time_series_ops_accept_integral_float_windows_from_llm_expressions():
    index = pd.date_range("2024-01-01", periods=40, freq="D")
    close = pd.DataFrame({"A": np.arange(1.0, 41.0), "B": np.arange(2.0, 42.0)}, index=index)
    turnover = pd.DataFrame({"A": np.linspace(10.0, 20.0, 40), "B": np.linspace(12.0, 22.0, 40)}, index=index)

    short_return = ts_return(close, 5.0)
    long_return = ts_return(close, 20.0)
    residual = ts_multireg([short_return, long_return, turnover], ts_return(close, 1.0), 30.0)

    assert short_return.shape == close.shape
    assert long_return.shape == close.shape
    assert residual.shape == close.shape


def test_where_preserves_boolean_series_condition_when_broadcasting_to_panel():
    index = pd.date_range("2024-01-01", periods=3, freq="D")
    condition = pd.Series([True, False, True], index=index)
    values = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}, index=index)

    result = WHERE(condition, values, 0)

    expected = pd.DataFrame({"A": [1.0, 0.0, 3.0], "B": [4.0, 0.0, 6.0]}, index=index)
    pd.testing.assert_frame_equal(result, expected)
