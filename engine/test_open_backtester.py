
import pandas as pd
import pytest

from engine.backtester_open import run_open_backtest


def make_prices():
    dates = pd.to_datetime([
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
    ])

    opens = pd.Series(
        [100.0, 110.0, 105.0, 115.0],
        index=dates,
        name="Open",
    )

    closes = pd.Series(
        [105.0, 108.0, 112.0, 118.0],
        index=dates,
        name="Close",
    )

    return opens, closes


def test_signal_timing_and_open_returns():
    opens, closes = make_prices()
    signal = pd.Series([1, 1, 0, 0], index=opens.index)

    results = run_open_backtest(opens, closes, signal, 0.001)

    assert results["Position"].tolist() == [0.0, 1.0, 1.0]

    expected = [
        110.0 / 100.0 - 1,
        105.0 / 110.0 - 1,
        115.0 / 105.0 - 1,
    ]
    assert results["Asset_Return"].tolist() == pytest.approx(expected)


def test_transaction_costs_on_position_changes():
    opens, closes = make_prices()
    signal = pd.Series([1, 1, 0, 0], index=opens.index)

    results = run_open_backtest(opens, closes, signal, 0.001)

    assert results["Turnover"].tolist() == [0.0, 1.0, 0.0]
    assert results["Transaction_Cost"].tolist() == pytest.approx(
        [0.0, 0.001, 0.0]
    )


def test_all_cash_has_zero_returns_and_costs():
    opens, closes = make_prices()
    signal = pd.Series(0, index=opens.index)

    results = run_open_backtest(opens, closes, signal)

    assert (results["Position"] == 0).all()
    assert (results["Net_Return"] == 0).all()
    assert (results["Transaction_Cost"] == 0).all()


def test_invalid_signal_is_rejected():
    opens, closes = make_prices()
    signal = pd.Series([0, 1, 2, 0], index=opens.index)

    with pytest.raises(ValueError, match="Signals"):
        run_open_backtest(opens, closes, signal)


def test_misaligned_signal_dates_are_rejected():
    opens, closes = make_prices()
    signal = pd.Series([0, 1, 0], index=opens.index[:3])

    with pytest.raises(ValueError, match="matching dates"):
        run_open_backtest(opens, closes, signal)


def test_missing_prices_are_rejected():
    opens, closes = make_prices()
    opens.iloc[1] = float("nan")
    signal = pd.Series(0, index=opens.index)

    with pytest.raises(ValueError, match="missing values"):
        run_open_backtest(opens, closes, signal)


def test_negative_transaction_cost_is_rejected():
    opens, closes = make_prices()
    signal = pd.Series(0, index=opens.index)

    with pytest.raises(ValueError, match="Transaction cost"):
        run_open_backtest(opens, closes, signal, -0.001)


def test_final_row_is_removed():
    opens, closes = make_prices()
    signal = pd.Series(0, index=opens.index)

    results = run_open_backtest(opens, closes, signal)

    assert len(results) == len(opens) - 1
    assert results.index[-1] == opens.index[-2]

def test_always_invested_matches_benchmark_after_entry_cost():
    opens, closes = make_prices()
    signal = pd.Series(1, index=opens.index)

    results = run_open_backtest(
        opens, closes, signal, transaction_cost=0.001
    )

    # The first interval is in cash because no prior signal exists.
    assert results["Position"].iloc[0] == 0.0

    # Subsequent intervals remain invested.
    assert (results["Position"].iloc[1:] == 1.0).all()

    # No transaction cost is charged until the position changes.
    assert results["Transaction_Cost"].tolist() == pytest.approx(
        [0.0, 0.001, 0.0]
    )

    # Once invested, strategy returns equal the benchmark returns
    # minus any transaction cost.
    invested_rows = results.iloc[1:]

    expected_net_returns = (
        invested_rows["Asset_Return"]
        - invested_rows["Transaction_Cost"]
    )

    assert invested_rows["Net_Return"].tolist() == pytest.approx(
        expected_net_returns.tolist()
    )

def test_zero_transaction_cost():
    opens, closes = make_prices()
    signal = pd.Series([1, 1, 0, 0], index=opens.index)

    results = run_open_backtest(
        opens,
        closes,
        signal,
        transaction_cost=0.0,
    )

    assert (results["Transaction_Cost"] == 0.0).all()


def test_benchmark_uses_open_to_open_returns():
    opens, closes = make_prices()
    signal = pd.Series(0, index=opens.index)

    results = run_open_backtest(opens, closes, signal)

    expected_asset_returns = [
        110.0 / 100.0 - 1,
        105.0 / 110.0 - 1,
        115.0 / 105.0 - 1,
    ]

    assert results["Asset_Return"].tolist() == pytest.approx(
        expected_asset_returns
    )

    expected_benchmark_equity = (
        1 + pd.Series(expected_asset_returns)
    ).cumprod().tolist()

    assert results["Buy_Hold_Equity"].tolist() == pytest.approx(
        expected_benchmark_equity
    )

def test_single_date_returns_empty_results():
    dates = pd.to_datetime(["2026-01-05"])

    opens = pd.Series([100.0], index=dates, name="Open")
    closes = pd.Series([105.0], index=dates, name="Close")
    signal = pd.Series([0], index=dates)

    results = run_open_backtest(opens, closes, signal)

    assert results.empty
    assert "Net_Return" in results.columns
    assert "Buy_Hold_Equity" in results.columns


def test_transaction_cost_is_charged_when_exiting_position():
    opens, closes = make_prices()
    signal = pd.Series([1, 0, 0, 0], index=opens.index)

    results = run_open_backtest(
        opens,
        closes,
        signal,
        transaction_cost=0.001,
    )

    # The first interval is cash because no previous signal exists.
    # The second interval enters the position.
    # The third interval exits the position.
    assert results["Position"].tolist() == [0.0, 1.0, 0.0]

    assert results["Turnover"].tolist() == pytest.approx(
        [0.0, 1.0, 1.0]
    )

    assert results["Transaction_Cost"].tolist() == pytest.approx(
        [0.0, 0.001, 0.001]
    )
