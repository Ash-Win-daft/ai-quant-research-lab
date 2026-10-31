
import pandas as pd

from engine.metrics import calculate_performance_metrics

sample_returns = pd.Series([
    0.01,
    -0.02,
    0.015,
    0.005,
    -0.01,
])

metrics = calculate_performance_metrics(sample_returns)

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# Basic sanity checks
assert metrics["Total Return"] > -1
assert metrics["Maximum Drawdown"] <= 0

print("\nAll metrics checks passed.")
