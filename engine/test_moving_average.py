
import pandas as pd
from engine.backtester import run_backtest
from engine.data import validate_market_data

# Create a small, artificial price history for testing
dates = pd.bdate_range(start="2025-01-01", periods=100)

prices = pd.Series(
    [100 + i * 0.2 + (i % 7) * 0.5 for i in range(100)],
    index=dates,
    name="Close",
)

# Validate prices
validate_market_data(pd.DataFrame({"Close": prices}))

# Calculate moving averages
sma_5 = prices.rolling(window=5).mean()
sma_15 = prices.rolling(window=15).mean()

# Signal: hold the asset when the short SMA is above the long SMA
signal = (sma_5 > sma_15).astype(int)

# Run the shared backtester
results = run_backtest(
    prices=prices,
    signal=signal,
    transaction_cost=0.001,
)

# Print summary statistics
strategy_return = results["Strategy_Equity"].iloc[-1] - 1
buy_hold_return = results["Buy_Hold_Equity"].iloc[-1] - 1

print("MOVING-AVERAGE STRATEGY TEST")
print("----------------------------")
print(f"Observations: {len(results)}")
print(f"Strategy total return: {strategy_return:.2%}")
print(f"Buy-and-hold total return: {buy_hold_return:.2%}")
print(f"Total turnover: {results['Turnover'].sum():.0f}")
print(f"Total transaction costs: {results['Transaction_Cost'].sum():.2%}")

print("\nLast five observations:")
print(results[["Close", "Signal", "Position", "Net_Return"]].tail())

print("\nMoving-average strategy test completed.")
