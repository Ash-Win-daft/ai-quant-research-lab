
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# PATHS AND INPUT
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "reports" / "multi_stock_mean_reversion_open_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_FILE = OUTPUT_DIR / "cross_stock_performance.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# LOAD AND VALIDATE DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "ticker",
    "Strategy_Total Return",
    "BuyHold_Total Return",
    "Excess_Total_Return",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

if len(df) != 10:
    raise ValueError(f"Expected 10 stocks, found {len(df)}")

if df["ticker"].duplicated().any():
    raise ValueError("Duplicate tickers found.")

if df[required_columns[1:]].isna().any().any():
    raise ValueError("Missing return values detected.")

# Sort by relative strategy performance
df = df.sort_values("Excess_Total_Return").reset_index(drop=True)

tickers = df["ticker"].tolist()

strategy_returns = df["Strategy_Total Return"].to_numpy() * 100
benchmark_returns = df["BuyHold_Total Return"].to_numpy() * 100

# Verify that excess return matches the difference
calculated_excess = (
    df["Strategy_Total Return"] - df["BuyHold_Total Return"]
).to_numpy()

if not np.allclose(
    calculated_excess,
    df["Excess_Total_Return"].to_numpy(),
    atol=1e-8,
):
    raise ValueError("Excess return validation failed.")


# ------------------------------------------------------------
# CREATE CHART
# ------------------------------------------------------------

y = np.arange(len(tickers))
bar_height = 0.36

fig, ax = plt.subplots(figsize=(13, 8))

strategy_bars = ax.barh(
    y - bar_height / 2,
    strategy_returns,
    height=bar_height,
    label="Mean-Reversion Strategy",
)

benchmark_bars = ax.barh(
    y + bar_height / 2,
    benchmark_returns,
    height=bar_height,
    label="Buy-and-Hold Benchmark",
)

ax.set_yticks(y)
ax.set_yticklabels(tickers)

ax.axvline(0, linestyle="--", linewidth=1)

ax.set_title(
    "Mean-Reversion Strategy vs Buy-and-Hold Across 10 US Stocks",
    fontsize=16,
    fontweight="bold",
    pad=18,
)

ax.set_xlabel("Cumulative Return (%)", fontsize=11)
ax.set_ylabel("Stock Ticker", fontsize=11)

ax.legend(loc="upper right")
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)

# Add return labels to each bar
for bars in [strategy_bars, benchmark_bars]:
    for bar in bars:
        width = bar.get_width()

        ax.annotate(
            f"{width:.1f}%",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5 if width >= 0 else -5, 0),
            textcoords="offset points",
            ha="left" if width >= 0 else "right",
            va="center",
            fontsize=8,
        )

# Add sufficient room for labels
max_return = max(strategy_returns.max(), benchmark_returns.max())
min_return = min(strategy_returns.min(), benchmark_returns.min())

ax.set_xlim(
    min(-10, min_return - 20),
    max_return + 45,
)

fig.text(
    0.5,
    0.015,
    "Out-of-sample: 2024–2025 | 10 US equities | "
    "Window: 20 | Threshold: -5% | "
    "Transaction cost: 0.10% per turnover",
    ha="center",
    fontsize=9,
    color="dimgray",
)

plt.tight_layout(rect=[0, 0.045, 1, 1])

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("=" * 60)
print("CROSS-STOCK PERFORMANCE VISUAL GENERATED")
print("=" * 60)
print(f"Stocks: {len(df)}")
print(
    "Stocks outperforming buy-and-hold:",
    int((df["Beats_Buy_Hold"] == True).sum()),
)
print(f"Saved to: {OUTPUT_FILE}")
