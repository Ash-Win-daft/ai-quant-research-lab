
import pandas as pd

from engine.data import validate_market_data


def run_backtest(prices, signal, transaction_cost=0.001):
    """
    Backtest a long-or-cash strategy.

    prices: closing-price Series indexed by date
    signal: 1 = want to hold the asset, 0 = want to hold cash
    transaction_cost: cost per unit of position turnover
                      0.001 means 0.1%
    """

    # 1. Validate the price data
    market_data = pd.DataFrame({"Close": prices})
    validate_market_data(market_data)

    # 2. Validate the strategy signals
    if not isinstance(signal, pd.Series):
        raise TypeError("Signal must be a pandas Series.")

    if not prices.index.equals(signal.index):
        raise ValueError("Prices and signals must have matching dates.")

    if signal.isna().any():
        raise ValueError("Signals contain missing values.")

    if not signal.isin([0, 1]).all():
        raise ValueError("Signals must contain only 0 or 1.")

    if not 0 <= transaction_cost < 1:
        raise ValueError("Transaction cost must be between 0 and 1.")

    # 3. Calculate the asset's daily returns
    results = pd.DataFrame(index=prices.index)
    results["Close"] = prices
    results["Asset_Return"] = prices.pct_change().fillna(0)

    # 4. Use yesterday's signal to avoid look-ahead bias
    results["Signal"] = signal
    results["Position"] = results["Signal"].shift(1).fillna(0)

    # 5. Calculate turnover and transaction costs
    results["Turnover"] = results["Position"].diff().abs()
    results.iloc[0, results.columns.get_loc("Turnover")] = (
        abs(results["Position"].iloc[0])
    )

    results["Transaction_Cost"] = (
        results["Turnover"] * transaction_cost
    )

    # 6. Calculate strategy returns
    results["Gross_Return"] = (
        results["Position"] * results["Asset_Return"]
    )

    results["Net_Return"] = (
        results["Gross_Return"] - results["Transaction_Cost"]
    )

    # 7. Calculate the growth of an initial 1-unit investment
    results["Strategy_Equity"] = (
        1 + results["Net_Return"]
    ).cumprod()

    results["Buy_Hold_Equity"] = (
        1 + results["Asset_Return"]
    ).cumprod()

    return results
