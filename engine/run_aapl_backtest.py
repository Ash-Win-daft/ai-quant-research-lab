
from engine.metrics import calculate_performance_metrics
import yfinance as yf
import pandas as pd

from engine.backtester import run_backtest
from engine.data import validate_market_data

print("Downloading AAPL historical data...")

ticker = yf.Ticker("AAPL")
history = ticker.history(
    start="2021-01-01",
    end="2026-01-01",
    auto_adjust=True,
)

if history.empty:
    raise ValueError("No AAPL data downloaded. Check your connection.")

# Use adjusted prices and remove timezone information from the dates
prices = history["Close"].copy()
prices.index = prices.index.tz_localize(None)
prices.name = "Close"

# Validate the data
validate_market_data(pd.DataFrame({"Close": prices}))

# Create the moving-average strategy
sma_20 = prices.rolling(20).mean()
sma_50 = prices.rolling(50).mean()

signal = (sma_20 > sma_50).astype(int)

# Run the shared engine
results = run_backtest(
    prices=prices,
    signal=signal,
    transaction_cost=0.001,
)

# Compare strategy and buy-and-hold results
strategy_return = results["Strategy_Equity"].iloc[-1] - 1
buy_hold_return = results["Buy_Hold_Equity"].iloc[-1] - 1

print("\nAAPL BACKTEST")
print("-------------")
print(f"Observations: {len(results)}")
print(f"First date: {results.index[0].date()}")
print(f"Last date: {results.index[-1].date()}")
print(f"Strategy total return: {strategy_return:.2%}")
print(f"Buy-and-hold total return: {buy_hold_return:.2%}")
print(f"Total turnover: {results['Turnover'].sum():.0f}")

print("\nLast five observations:")
print(
    results[
        ["Close", "Signal", "Position", "Net_Return"]
    ].tail().round(4)
)

print("\nReal-data backtest completed.")


# Save the complete daily backtest
results.to_csv("reports/aapl_moving_average_daily.csv")

# Save a compact performance summary
summary = pd.DataFrame({
    "Metric": [
        "Observations",
        "Strategy Total Return",
        "Buy-and-Hold Total Return",
        "Total Turnover",
    ],
    "Value": [
        len(results),
        strategy_return,
        buy_hold_return,
        results["Turnover"].sum(),
    ],
})

summary.to_csv(
    "reports/aapl_moving_average_summary.csv",
    index=False,
)

print("\nResults saved to the reports folder.")


# Calculate standardized performance metrics
strategy_metrics = calculate_performance_metrics(
    results["Net_Return"]
)

benchmark_metrics = calculate_performance_metrics(
    results["Asset_Return"]
)

comparison = pd.DataFrame({
    "Strategy": strategy_metrics,
    "Buy and Hold": benchmark_metrics,
})

print("\nPERFORMANCE COMPARISON")
print("----------------------")
print(comparison.round(4))

# Save the comparison
comparison.to_csv("reports/aapl_performance_metrics.csv")

print("\nPerformance metrics saved successfully.")
