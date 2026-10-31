
import pandas as pd

from engine.backtester import run_backtest
from engine.metrics import calculate_performance_metrics


def evaluate_moving_average_parameters(
    prices,
    parameter_pairs,
    transaction_cost=0.001,
):
    """
    Evaluate multiple moving-average parameter combinations.

    Each pair is (short_window, long_window).
    """

    records = []

    for short_window, long_window in parameter_pairs:

        if short_window <= 0 or long_window <= 0:
            raise ValueError("Moving-average windows must be positive.")

        if short_window >= long_window:
            raise ValueError(
                "Short window must be smaller than long window."
            )

        short_sma = prices.rolling(short_window).mean()
        long_sma = prices.rolling(long_window).mean()

        signal = (short_sma > long_sma).astype(int)

        results = run_backtest(
            prices=prices,
            signal=signal,
            transaction_cost=transaction_cost,
        )

        metrics = calculate_performance_metrics(
            results["Net_Return"]
        )

        records.append({
            "Short_Window": short_window,
            "Long_Window": long_window,
            **metrics,
        })

    return pd.DataFrame(records)
