from pathlib import Path

import numpy as np
import pandas as pd

from engine.backtester import run_backtest


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

train_prices = prices.loc["2021-01-01":"2023-12-31"]
test_prices = prices.loc["2024-01-01":"2025-12-31"]

if train_prices.empty or test_prices.empty:
    raise ValueError("Research or test period contains no data.")


# --------------------------------------------------
# 2. Select parameters using training data only
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

    daily_returns = results["Net_Return"].dropna()

    training_records.append({
        "Window": window,
        "Threshold": threshold,
        "Sharpe": (
            daily_returns.mean() / daily_returns.std() * np.sqrt(252)
            if daily_returns.std() > 0
            else np.nan
        ),
    })

training = pd.DataFrame(training_records)

best_row = training.loc[training["Sharpe"].idxmax()]
best_window = int(best_row["Window"])
best_threshold = float(best_row["Threshold"])

print("\nSELECTED USING 2021-2023 DATA")
print("-----------------------------")
print("Window:", best_window)
print("Entry threshold:", f"{best_threshold:.1%}")


# --------------------------------------------------
# 3. Generate out-of-sample daily returns
# --------------------------------------------------

warmup = prices.loc[:"2023-12-31"].tail(best_window - 1)
test_input = pd.concat([warmup, test_prices])

average = test_input.rolling(best_window).mean()
deviation = test_input / average - 1
signal = (deviation < best_threshold).astype(int)

combined_results = run_backtest(
    prices=test_input,
    signal=signal,
    transaction_cost=0.001,
)

test_results = combined_results.loc[test_prices.index].copy()

# Align strategy and benchmark returns by date.
returns = test_results[
    ["Net_Return", "Asset_Return"]
].dropna()

returns = returns.rename(columns={
    "Net_Return": "Strategy",
    "Asset_Return": "Buy_Hold",
})

if len(returns) < 30:
    raise ValueError("Too few observations for this analysis.")


# --------------------------------------------------
# 4. Compare daily returns
# --------------------------------------------------

returns["Daily_Excess_Return"] = (
    returns["Strategy"] - returns["Buy_Hold"]
)

difference = returns["Daily_Excess_Return"]

n = len(difference)
mean_difference = difference.mean()
daily_std = difference.std(ddof=1)
standard_error = daily_std / np.sqrt(n)

if standard_error == 0:
    raise ValueError("Daily return differences have zero variance.")

# Approximate 95% confidence interval using a normal critical value.
ci_low = mean_difference - 1.96 * standard_error
ci_high = mean_difference + 1.96 * standard_error

# Ordinary one-sample t-statistic.
t_statistic = mean_difference / standard_error

# Two-sided p-value using the Student t distribution.
# SciPy is commonly installed alongside scientific Python packages.
try:
    from scipy.stats import t as student_t

    p_value = 2 * student_t.sf(
        abs(t_statistic),
        df=n - 1,
    )
except ImportError:
    p_value = np.nan


# --------------------------------------------------
# 5. Display the findings
# --------------------------------------------------

print("\nDAILY RETURN SIGNIFICANCE TEST")
print("------------------------------")
print("Test period: 2024-2025")
print("Observations:", n)
print("Mean strategy daily return:", f"{returns['Strategy'].mean():.6%}")
print("Mean buy-and-hold daily return:", f"{returns['Buy_Hold'].mean():.6%}")
print("Mean daily excess return:", f"{mean_difference:.6%}")
print("95% confidence interval:", f"[{ci_low:.6%}, {ci_high:.6%}]")
print("t-statistic:", round(t_statistic, 4))

if np.isfinite(p_value):
    print("Two-sided p-value:", round(float(p_value), 6))
else:
    print("Two-sided p-value unavailable: SciPy is not installed.")

if ci_low > 0:
    print("Naive interval interpretation: strategy has higher mean daily returns.")
elif ci_high < 0:
    print("Naive interval interpretation: strategy has lower mean daily returns.")
else:
    print("Naive interval interpretation: no clear mean-return difference.")

print(
    "\nCaution: daily returns can be serially dependent, "
    "and strategy selection adds uncertainty. "
    "This ordinary t-test is diagnostic, not definitive proof."
)


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

summary = pd.DataFrame([{
    "Test_Start": str(returns.index.min().date()),
    "Test_End": str(returns.index.max().date()),
    "Observations": n,
    "Selected_Window": best_window,
    "Selected_Threshold": best_threshold,
    "Transaction_Cost": 0.001,
    "Mean_Strategy_Daily_Return": returns["Strategy"].mean(),
    "Mean_Buy_Hold_Daily_Return": returns["Buy_Hold"].mean(),
    "Mean_Daily_Excess_Return": mean_difference,
    "CI_95_Lower": ci_low,
    "CI_95_Upper": ci_high,
    "T_Statistic": t_statistic,
    "P_Value": p_value,
}])

output_dir = Path(__file__).resolve().parent.parent / "reports"
output_dir.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    output_dir / "aapl_mean_reversion_significance.csv",
    index=False,
)

returns.to_csv(
    output_dir / "aapl_mean_reversion_significance_daily.csv"
)

print("\nResults saved:")
print("reports/aapl_mean_reversion_significance.csv")
print("reports/aapl_mean_reversion_significance_daily.csv")
