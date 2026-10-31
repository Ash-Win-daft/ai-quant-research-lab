
import pandas as pd

from engine.data import validate_market_data


def run_open_backtest(
    opens,
    closes,
    signal,
    transaction_cost=0.001,
):
    """
    Long-or-cash backtest using open-to-open returns.

    A signal calculated from the previous session's close
    determines the position held from today's open until
    the next session's open.

    opens and closes: price Series indexed by trading date
    signal: 1 = hold the asset, 0 = hold cash
    transaction_cost: cost per unit of position turnover

    The final row is excluded because it has no next-session
    opening price for calculating an open-to-open return.
    """

    # 1. Validate price data
    market_data = pd.DataFrame({
        "Open": opens,
        "Close": closes,
    })

    validate_market_data(market_data)

    if market_data.isna().any().any():
        raise ValueError(
            "Open and close prices must not contain missing values."
        )

    if (market_data <= 0).any().any():
        raise ValueError(
            "Open and close prices must be positive."
        )

    if not opens.index.equals(closes.index):
        raise ValueError(
            "Open and close prices must have matching dates."
        )

    # 2. Validate signals
    if not isinstance(signal, pd.Series):
        raise TypeError("Signal must be a pandas Series.")

    if not closes.index.equals(signal.index):
        raise ValueError(
            "Prices and signals must have matching dates."
        )

    if signal.isna().any() or not signal.isin([0, 1]).all():
        raise ValueError(
            "Signals must contain only 0 or 1, without missing values."
        )

    if not 0 <= transaction_cost < 1:
        raise ValueError(
            "Transaction cost must be between 0 and 1."
        )

    # 3. Set the position for each open-to-open interval
    results = pd.DataFrame(index=closes.index)
    results["Open"] = opens
    results["Close"] = closes
    results["Signal"] = signal
    results["Position"] = signal.shift(1).fillna(0)

    # 4. Calculate open-to-open returns
    results["Asset_Return"] = (
        results["Open"].shift(-1) / results["Open"] - 1
    )

    # Remove the final row, which has no next opening price
    results = results.iloc[:-1].copy()

    # 5. Calculate turnover and transaction costs
    results["Turnover"] = results["Position"].diff().abs()

    if not results.empty:
        results.iloc[
            0, results.columns.get_loc("Turnover")
        ] = abs(results["Position"].iloc[0])

    results["Transaction_Cost"] = (
        results["Turnover"] * transaction_cost
    )

    # 6. Calculate strategy and benchmark returns
    results["Gross_Return"] = (
        results["Position"] * results["Asset_Return"]
    )

    results["Net_Return"] = (
        results["Gross_Return"] - results["Transaction_Cost"]
    )

    results["Strategy_Equity"] = (
        1 + results["Net_Return"]
    ).cumprod()

    # Benchmark uses the same return intervals, without costs
    results["Buy_Hold_Equity"] = (
        1 + results["Asset_Return"]
    ).cumprod()

    return results
