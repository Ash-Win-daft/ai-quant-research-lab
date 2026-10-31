import pandas as pd

from engine.backtester import run_backtest
from engine.metrics import calculate_performance_metrics


# --------------------------------------------------
# 1. Load historical data from local CSV
# --------------------------------------------------

print("Loading AAPL data from local CSV...")

history = pd.read_csv(
    "data/aapl_daily.csv",
    parse_dates=["Date"],
    index_col="Date",
)

if history.empty:
    raise ValueError("The local AAPL dataset is empty.")

prices = history["Close"].copy()

# Use the same chronological split as our earlier experiment.
train_prices = prices.loc["2021-01-01":"2023-12-31"]
test_prices = prices.loc["2024-01-01":"2025-12-31"]

if train_prices.empty or test_prices.empty:
    raise ValueError("Research or test period contains no data.")


# --------------------------------------------------
# 2. Select parameters using research data only
# --------------------------------------------------

parameter_combinations = [
    (10, -0.02),
    (10, -0.03),
    (20, -0.02),
    (20, -0.03),
    (20, -0.05),
    (50, -0.03),
    (50, -0.05),
]

training_records = []

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

    training_records.append({
        "Window": window,
        "Threshold": threshold,
        **metrics,
    })

training_results = pd.DataFrame(training_records)

best_row = training_results.loc[
    training_results["Sharpe Ratio (rf=0)"].idxmax()
]

best_window = int(best_row["Window"])
best_threshold = float(best_row["Threshold"])

print("\nPARAMETERS SELECTED USING 2021-2023 DATA")
print("----------------------------------------")
print("Rolling window:", best_window)
print("Entry threshold:", f"{best_threshold:.1%}")
print("Training Sharpe:", round(
    float(best_row["Sharpe Ratio (rf=0)"]), 4
))


# --------------------------------------------------
# 3. Prepare the out-of-sample test period
# --------------------------------------------------

# Earlier prices initialize the rolling average.
# They are not included in the test-period performance.
warmup = prices.loc[:"2023-12-31"].tail(best_window - 1)
test_input = pd.concat([warmup, test_prices])

average = test_input.rolling(best_window).mean()
deviation = test_input / average - 1
signal = (deviation < best_threshold).astype(int)


# --------------------------------------------------
# 4. Stress-test different transaction costs
# --------------------------------------------------

cost_scenarios = [
    0.0000,  # 0.00%
    0.0005,  # 0.05%
    0.0010,  # 0.10%
    0.0025,  # 0.25%
    0.0050,  # 0.50%
]

records = []

for cost in cost_scenarios:
    combined_results = run_backtest(
        prices=test_input,
        signal=signal,
        transaction_cost=cost,
    )

    test_results = combined_results.loc[test_prices.index]

    strategy_metrics = calculate_performance_metrics(
        test_results["Net_Return"]
    )

    benchmark_metrics = calculate_performance_metrics(
        test_results["Asset_Return"]
    )

    records.append({
        "Transaction_Cost_Per_Turnover": cost,
        "Strategy_Total_Return": strategy_metrics["Total Return"],
        "Strategy_CAGR": strategy_metrics["CAGR"],
        "Strategy_Sharpe": strategy_metrics["Sharpe Ratio (rf=0)"],
        "Strategy_Max_Drawdown": strategy_metrics["Maximum Drawdown"],
        "Buy_Hold_Total_Return": benchmark_metrics["Total Return"],
        "Buy_Hold_Sharpe": benchmark_metrics["Sharpe Ratio (rf=0)"],
    })

stress_results = pd.DataFrame(records)


# --------------------------------------------------
# 5. Display and save results
# --------------------------------------------------

print("\nTRANSACTION-COST STRESS TEST")
print("----------------------------")
print("Test period: 2024-2025")
print("Selected parameters:", best_window, "sessions,",
      f"{best_threshold:.1%} threshold")
print("\nReturns and drawdowns are decimals in the saved CSV.")
print(stress_results.round(4).to_string(index=False))

output_path = (
    "reports/aapl_mean_reversion_transaction_cost_stress.csv"
)

stress_results.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)
print("Transaction-cost stress test completed.")
