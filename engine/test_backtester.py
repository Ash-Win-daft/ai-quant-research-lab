
import pandas as pd

from engine.backtester import run_backtest

dates = pd.to_datetime([
    "2025-01-02",
    "2025-01-03",
    "2025-01-06",
    "2025-01-07",
])

prices = pd.Series(
    [100.0, 110.0, 105.0, 115.0],
    index=dates,
)

signal = pd.Series(
    [1, 1, 0, 1],
    index=dates,
)

results = run_backtest(
    prices,
    signal,
    transaction_cost=0.001,
)

print(results[
    [
        "Close",
        "Signal",
        "Position",
        "Asset_Return",
        "Turnover",
        "Net_Return",
        "Strategy_Equity",
        "Buy_Hold_Equity",
    ]
].round(4))

# Check timing: first day's position must be zero
assert results["Position"].iloc[0] == 0

# Monday's signal must determine the next listed session's position
assert results["Position"].iloc[1] == 1
assert results["Position"].iloc[2] == 1
assert results["Position"].iloc[3] == 0

# The strategy should pay a cost when its position changes
assert results["Transaction_Cost"].iloc[1] > 0

print("\nAll backtester checks passed.")


# Test 1: Reject invalid signals
invalid_signal = signal.copy()
invalid_signal.iloc[1] = 2

try:
    run_backtest(prices, invalid_signal)
except ValueError as error:
    print("Invalid signal correctly rejected:", error)
else:
    raise AssertionError("Invalid signal was not rejected.")

# Test 2: Reject mismatched dates
wrong_dates_signal = signal.copy()
wrong_dates_signal.index = wrong_dates_signal.index + pd.Timedelta(days=1)

try:
    run_backtest(prices, wrong_dates_signal)
except ValueError as error:
    print("Mismatched dates correctly rejected:", error)
else:
    raise AssertionError("Mismatched dates were not rejected.")

print("All additional validation checks passed.")
