
import yfinance as yf
import pandas as pd

from engine.backtester import run_backtest
from engine.metrics import calculate_performance_metrics

print("Downloading AAPL data...")

history = yf.Ticker("AAPL").history(
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
)

if history.empty:
    raise ValueError("No AAPL data downloaded.")

prices = history["Close"].copy()
prices.index = prices.index.tz_localize(None)

# Test several mean-reversion hypotheses.
# Each tuple contains (lookback window, entry threshold).

parameter_combinations = [
    (10, -0.02),
    (10, -0.03),
    (20, -0.02),
    (20, -0.03),
    (20, -0.05),
    (50, -0.03),
    (50, -0.05),
]

experiment_records = []

for window, entry_threshold in parameter_combinations:
    rolling_average = prices.rolling(window).mean()
    deviation = prices / rolling_average - 1

    signal = (deviation < entry_threshold).astype(int)

    results = run_backtest(
        prices=prices,
        signal=signal,
        transaction_cost=0.001,
    )

    metrics = calculate_performance_metrics(
        results["Net_Return"]
    )

    experiment_records.append({
        "Window": window,
        "Entry_Threshold": entry_threshold,
        **metrics,
    })

experiment_results = pd.DataFrame(experiment_records)

print("\nMEAN-REVERSION PARAMETER EXPERIMENT")
print("----------------------------------")
print(experiment_results.round(4).to_string(index=False))

experiment_results.to_csv(
    "reports/aapl_mean_reversion_parameter_test.csv",
    index=False,
)

print("\nParameter experiment saved successfully.")

# Re-run the selected configuration: 50-day window, -3% threshold.
best_window = 50
best_entry_threshold = -0.03

rolling_average = prices.rolling(best_window).mean()
deviation = prices / rolling_average - 1
best_signal = (deviation < best_entry_threshold).astype(int)

best_results = run_backtest(
    prices=prices,
    signal=best_signal,
    transaction_cost=0.001,
)

# Calculate strategy and benchmark metrics from the same price data.
strategy_metrics = calculate_performance_metrics(
    best_results["Net_Return"]
)

benchmark_metrics = calculate_performance_metrics(
    best_results["Asset_Return"]
)

comparison = pd.DataFrame({
    "Mean Reversion (50-day, -3%)": strategy_metrics,
    "Buy and Hold": benchmark_metrics,
})

print("\nBEST MEAN-REVERSION STRATEGY VS BUY AND HOLD")
print("--------------------------------------------")
print(comparison.round(4))

print("\nDays invested:", int(best_results["Position"].sum()))
print("Position changes:", int(best_results["Turnover"].sum()))

comparison.to_csv(
    "reports/aapl_mean_reversion_best_comparison.csv"
)

best_results.to_csv(
    "reports/aapl_mean_reversion_best_daily.csv"
)

print("\nFresh comparison and daily results saved successfully.")
