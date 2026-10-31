
import yfinance as yf
import pandas as pd

from engine.robustness import evaluate_moving_average_parameters
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

parameter_pairs = [
    (10, 30),
    (15, 40),
    (20, 50),
    (25, 60),
    (30, 100),
    (50, 150),
]

# Test each moving-average configuration
results = evaluate_moving_average_parameters(
    prices=prices,
    parameter_pairs=parameter_pairs,
    transaction_cost=0.001,
)

# Calculate buy-and-hold benchmark metrics
asset_returns = prices.pct_change().fillna(0)
benchmark = calculate_performance_metrics(asset_returns)

results["Buy_Hold_Total_Return"] = benchmark["Total Return"]
results["Buy_Hold_Sharpe"] = benchmark["Sharpe Ratio (rf=0)"]

# Flag configurations that fail these simple checks
results["Positive_Return"] = results["Total Return"] > 0

results["Beats_Buy_Hold_Return"] = (
    results["Total Return"] > results["Buy_Hold_Total_Return"]
)

results["Beats_Buy_Hold_Sharpe"] = (
    results["Sharpe Ratio (rf=0)"] > results["Buy_Hold_Sharpe"]
)

print("\nPARAMETER ROBUSTNESS AND BENCHMARK")
print("----------------------------------")
print(results.round(4).to_string(index=False))

results.to_csv(
    "reports/aapl_parameter_robustness.csv",
    index=False,
)

print("\nResults saved to reports/aapl_parameter_robustness.csv")
print("\nDiagnostic counts:")
print("Configurations with positive returns:", results["Positive_Return"].sum())
print("Configurations beating buy-and-hold return:", results["Beats_Buy_Hold_Return"].sum())
print("Configurations beating buy-and-hold Sharpe:", results["Beats_Buy_Hold_Sharpe"].sum())
