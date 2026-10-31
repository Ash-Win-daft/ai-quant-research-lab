
import pandas as pd

from engine.backtester import run_backtest
from engine.metrics import calculate_performance_metrics

print("Loading AAPL data from local CSV...")

history = pd.read_csv(
    "data/aapl_daily.csv",
    parse_dates=["Date"],
    index_col="Date",
)

if history.empty:
    raise ValueError("The local AAPL dataset is empty.")

prices = history["Close"].copy()
opens = history["Open"].copy()

# Split data chronologically.
# The parameter-selection period ends in 2023.
# The out-of-sample test period covers 2024 and 2025.

train_prices = prices.loc["2021-01-01":"2023-12-31"]
test_prices = prices.loc["2024-01-01":"2025-12-31"]

print("\nDATA SPLIT")
print("----------")
print("Research observations:", len(train_prices))
print("Test observations:", len(test_prices))

# Step 1: Select parameters using research-period data only.

parameter_combinations = [
    (10, -0.02),
    (10, -0.03),
    (20, -0.02),
    (20, -0.03),
    (20, -0.05),
    (50, -0.03),
    (50, -0.05),
]

records = []

for window, threshold in parameter_combinations:
    average = train_prices.rolling(window).mean()
    deviation = train_prices / average - 1
    signal = (deviation < threshold).astype(int)

    results = run_backtest(
        prices=train_prices,
        signal=signal,
        transaction_cost=0.001,
    )

    metrics = calculate_performance_metrics(
        results["Net_Return"]
    )

    records.append({
        "Window": window,
        "Threshold": threshold,
        **metrics,
    })

training_results = pd.DataFrame(records)

# Select the configuration with the highest training-period Sharpe ratio.
best_row = training_results.loc[
    training_results["Sharpe Ratio (rf=0)"].idxmax()
]

best_window = int(best_row["Window"])
best_threshold = float(best_row["Threshold"])

print("\nPARAMETER SELECTION (2021-2023)")
print("--------------------------------")
print(training_results.round(4).to_string(index=False))

print("\nSelected window:", best_window)
print("Selected threshold:", best_threshold)
print("Selection criterion: highest research-period Sharpe ratio")

# Step 2: Test the selected parameters on 2024-2025.
# Use earlier prices to initialize the rolling average, but score
# strategy performance only on the test period.

test_history = prices.loc[:"2023-12-31"].tail(best_window - 1)
test_input = pd.concat([test_history, test_prices])

rolling_average = test_input.rolling(best_window).mean()
deviation = test_input / rolling_average - 1
all_signals = (deviation < best_threshold).astype(int)

# Include the previous session's signal for the first test-day position.
# Backtest the combined data, then measure metrics only within the test period.
combined_results = run_backtest(
    prices=test_input,
    signal=all_signals,
    transaction_cost=0.001,
)

test_results = combined_results.loc[test_prices.index].copy()

# Calculate test-period metrics.
strategy_metrics = calculate_performance_metrics(
    test_results["Net_Return"]
)

benchmark_metrics = calculate_performance_metrics(
    test_results["Asset_Return"]
)

comparison = pd.DataFrame({
    "Mean Reversion": strategy_metrics,
    "Buy and Hold": benchmark_metrics,
})

print("\nOUT-OF-SAMPLE RESULTS (2024-2025)")
print("---------------------------------")
print(comparison.round(4))

training_results.to_csv(
    "reports/aapl_mean_reversion_oos_training.csv",
    index=False,
)

comparison.to_csv(
    "reports/aapl_mean_reversion_oos_comparison.csv"
)

test_results.to_csv(
    "reports/aapl_mean_reversion_oos_daily.csv"
)

print("\nOut-of-sample results saved successfully.")
