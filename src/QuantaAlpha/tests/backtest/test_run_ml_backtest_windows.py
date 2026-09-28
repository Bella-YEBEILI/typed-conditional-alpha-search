import pandas as pd

from quantaalpha.backtest.run_ml_backtest import (
    compute_fixed_date_range,
    compute_rolling_window_ranges,
    compute_walk_forward_window_ranges,
    select_top_n_holding,
)


def test_compute_rolling_window_ranges_uses_custom_test_size():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")

    windows = compute_rolling_window_ranges(
        dates,
        rolling_window=10,
        validation_size=4,
        gap_size=2,
        test_size=5,
        max_windows=2,
    )

    assert len(windows) == 2

    first = windows[0]
    assert first["train_start_date"] == dates[0]
    assert first["train_end_date"] == dates[9]
    assert first["val_start_date"] == dates[10]
    assert first["val_end_date"] == dates[13]
    assert first["test_start_date"] == dates[16]
    assert first["test_end_date"] == dates[20]
    assert first["test_start_idx"] == 16
    assert first["test_end_idx"] == 21

    second = windows[1]
    assert second["test_start_date"] == dates[21]
    assert second["test_end_date"] == dates[25]


def test_compute_fixed_date_range_uses_explicit_train_valid_test_dates():
    window = compute_fixed_date_range(
        train_start="2022-01-01",
        train_end="2022-12-31",
        val_start="2023-01-01",
        val_end="2023-03-31",
        test_start="2023-04-01",
        test_end="2023-06-30",
    )

    assert window["window"] == 1
    assert str(window["train_start_date"].date()) == "2022-01-01"
    assert str(window["train_end_date"].date()) == "2022-12-31"
    assert str(window["val_start_date"].date()) == "2023-01-01"
    assert str(window["val_end_date"].date()) == "2023-03-31"
    assert str(window["test_start_date"].date()) == "2023-04-01"
    assert str(window["test_end_date"].date()) == "2023-06-30"


def test_compute_fixed_date_range_rejects_overlapping_dates():
    try:
        compute_fixed_date_range(
            train_start="2022-01-01",
            train_end="2023-02-01",
            val_start="2023-01-01",
            val_end="2023-03-31",
            test_start="2023-04-01",
            test_end="2023-06-30",
        )
    except ValueError as exc:
        assert "train_start <= train_end < val_start <= val_end < test_start <= test_end" in str(exc)
    else:
        raise AssertionError("expected overlapping fixed date ranges to be rejected")


def test_compute_walk_forward_window_ranges_rolls_inside_test_period_with_gap():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")

    windows = compute_walk_forward_window_ranges(
        dates,
        train_lookback=10,
        validation_size=4,
        gap_size=2,
        test_size=3,
        test_start="2024-01-17",
        test_end="2024-01-24",
    )

    assert len(windows) == 3

    first = windows[0]
    assert first["train_start_date"] == pd.Timestamp("2024-01-01")
    assert first["train_end_date"] == pd.Timestamp("2024-01-10")
    assert first["val_start_date"] == pd.Timestamp("2024-01-11")
    assert first["val_end_date"] == pd.Timestamp("2024-01-14")
    assert first["test_start_date"] == pd.Timestamp("2024-01-17")
    assert first["test_end_date"] == pd.Timestamp("2024-01-19")

    last = windows[-1]
    assert last["test_start_date"] == pd.Timestamp("2024-01-23")
    assert last["test_end_date"] == pd.Timestamp("2024-01-24")


def test_select_top_n_holding_uses_highest_predictions():
    pred = pd.Series({"a": 0.2, "b": 0.9, "c": 0.5, "d": 0.7})

    holding = select_top_n_holding(pred, top_n=2)

    assert holding == {"b", "d"}
