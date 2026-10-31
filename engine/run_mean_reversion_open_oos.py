
import pandas as pd

from engine.backtester_open import run_open_backtest
from engine.metrics import calculate_performance_metrics


TRANSACTION_COST = 0.001
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

print("Loading AAPL data from local CSV...")

history = pd.read_csv(
    "data/aapl_daily_extended.csv",
    parse_dates=["Date"],
    index_col="Date",
)

if history.empty:
    raise ValueError("The local AAPL dataset is empty.")

opens = history["Open"].copy()
closes = history["Close"].copy()

# Ensure both price series use exactly the same dates.
opens = opens.loc[closes.index]

if not opens.index.equals(closes.index):
    raise ValueError("Open and close prices have mismatched dates.")

# --------------------------------------------------
# 1. Select parameters using 2021-2023 only.
# --------------------------------------------------

train_closes = closes.loc[:TRAIN_END]
train_opens = opens.loc[train_closes.index]

if train_closes.empty:
    raise ValueError("No training data available.")

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

    # Signals use closing prices available at each date.
    rolling_average = train_closes.rolling(window).mean()
    deviation = train_closes / rolling_average - 1
    signal = (deviation < threshold).astype(int)

    results = run_open_backtest(
        opens=train_opens,
        closes=train_closes,
        signal=signal,
        transaction_cost=TRANSACTION_COST,
    )

    # The backtester calculates the return from today's open
    # to the next session's open and removes the final row.
    # Therefore, every remaining training return ends no later
    # than the final available training opening price.
    training_returns = results["Net_Return"].dropna()

    if training_returns.empty:
        raise ValueError(
            f"No training returns for window={window}, "
            f"threshold={threshold}."
        )

    metrics = calculate_performance_metrics(training_returns)

    training_records.append({
        "Window": window,
        "Threshold": threshold,
        **metrics,
    })

training_results = pd.DataFrame(training_records)

sharpe_column = "Sharpe Ratio (rf=0)"

if (
    sharpe_column not in training_results.columns
    or training_results[sharpe_column].isna().all()
):
    raise ValueError("Unable to select parameters using training Sharpe ratios.")

best_row = training_results.loc[
    training_results[sharpe_column].idxmax()
]

best_window = int(best_row["Window"])
best_threshold = float(best_row["Threshold"])

print("\nPARAMETER SELECTION: 2021-2023")
print("--------------------------------")
print(training_results.round(4).to_string(index=False))
print("\nSelected window:", best_window)
print("Selected threshold:", best_threshold)
print("Selection criterion: highest training-period Sharpe ratio")

# --------------------------------------------------
# 2. Evaluate the selected parameters on 2024-2025.
# --------------------------------------------------

# Calculate rolling averages across the full chronological
# price series. Rolling calculations use past and current
# observations only, not future prices.
rolling_average = closes.rolling(best_window).mean()
deviation = closes / rolling_average - 1
all_signals = (deviation < best_threshold).astype(int)

all_results = run_open_backtest(
    opens=opens,
    closes=closes,
    signal=all_signals,
    transaction_cost=TRANSACTION_COST,
)

first_test_date = pd.Timestamp("2024-01-02")
first_test_position = all_results.loc[first_test_date, "Position"]

previous_trading_date = all_signals.index[
    all_signals.index < first_test_date
][-1]

previous_close_signal = all_signals.loc[previous_trading_date]

print("\nFIRST TEST-PERIOD SIGNAL CHECK")
print("--------------------------------")
print("Previous trading date:", previous_trading_date.date())
print("Previous close signal:", previous_close_signal)
print("First test position:", first_test_position)

# Keep only intervals that start in the test period
# and end no later than the final 2025 opening price.
last_test_open = opens.loc[:TEST_END].index[-1]

test_results = all_results.loc[
    (all_results.index >= pd.Timestamp(TEST_START))
    & (all_results.index < last_test_open)
].copy()

if test_results.empty:
    raise ValueError("No out-of-sample results were produced.")
# Reset equity curves so they start at 1.0 for the
# out-of-sample evaluation period only.
test_results["Strategy_Equity"] = (
    1 + test_results["Net_Return"]
).cumprod()

test_results["Buy_Hold_Equity"] = (
    1 + test_results["Asset_Return"]
).cumprod()

# Both strategy and benchmark use identical open-to-open
# return intervals.
strategy_metrics = calculate_performance_metrics(
    test_results["Net_Return"]
)

benchmark_metrics = calculate_performance_metrics(
    test_results["Asset_Return"]
)

comparison = pd.DataFrame({
    "Mean Reversion (Open Execution)": strategy_metrics,
    "Buy and Hold (Open-to-Open)": benchmark_metrics,
})

print("\nOUT-OF-SAMPLE RESULTS: 2024-2025")
print("--------------------------------")
print(comparison.round(4))

print("\nTEST PERIOD CHECK")
print("-----------------")
print("First return date:", test_results.index.min().date())
print("Last return date:", test_results.index.max().date())
print("Number of return intervals:", len(test_results))
print("Transaction cost per unit of turnover:", TRANSACTION_COST)

# --------------------------------------------------
# 3. Save separate reports.
# Do not overwrite the original experiment's files.
# --------------------------------------------------

training_results.to_csv(
    "reports/aapl_mean_reversion_open_training.csv",
    index=False,
)

comparison.to_csv(
    "reports/aapl_mean_reversion_open_comparison.csv",
)

test_results.to_csv(
    "reports/aapl_mean_reversion_open_daily.csv",
)

print("\nNew open-execution results saved successfully.")
print("Original reports have not been overwritten.")
