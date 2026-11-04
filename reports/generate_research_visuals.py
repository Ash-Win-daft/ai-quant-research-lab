
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "multi_stock_mean_reversion_open_daily.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

OUTPUT_FILE = OUTPUT_DIR / "portfolio_performance.png"


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "Date",
    "ticker",
    "Net_Return",
    "Asset_Return",
]

missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["Date"] = pd.to_datetime(df["Date"])

if df[["Net_Return", "Asset_Return"]].isna().any().any():
    raise ValueError("Missing portfolio return observations.")

if df.duplicated(["Date", "ticker"]).any():
    raise ValueError("Duplicate stock-date observations detected.")

stocks_per_date = df.groupby("Date")["ticker"].nunique()

if not (stocks_per_date == 10).all():
    raise ValueError("Expected exactly 10 stocks on every date.")


# ============================================================
# RECONSTRUCT EQUAL-WEIGHTED DAILY PORTFOLIO
# ============================================================

portfolio = (
    df.groupby("Date")
    .agg(
        Strategy_Return=("Net_Return", "mean"),
        Benchmark_Return=("Asset_Return", "mean"),
    )
    .sort_index()
)

portfolio["Strategy_Equity"] = (
    1 + portfolio["Strategy_Return"]
).cumprod()

portfolio["Benchmark_Equity"] = (
    1 + portfolio["Benchmark_Return"]
).cumprod()

strategy_total_return = (
    portfolio["Strategy_Equity"].iloc[-1] - 1
)

benchmark_total_return = (
    portfolio["Benchmark_Equity"].iloc[-1] - 1
)


# ============================================================
# VERIFY EXPECTED RESEARCH RESULTS
# ============================================================

expected_strategy_return = 0.176861
expected_benchmark_return = 0.973356

if not np.isclose(
    strategy_total_return,
    expected_strategy_return,
    atol=1e-5,
):
    raise ValueError("Strategy return does not match research output.")

if not np.isclose(
    benchmark_total_return,
    expected_benchmark_return,
    atol=1e-5,
):
    raise ValueError("Benchmark return does not match research output.")


# ============================================================
# CREATE VISUALIZATION
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 7))

ax.plot(
    portfolio.index,
    portfolio["Strategy_Equity"],
    label="Mean-Reversion Strategy",
    linewidth=2.4,
)

ax.plot(
    portfolio.index,
    portfolio["Benchmark_Equity"],
    label="Equal-Weighted Benchmark",
    linewidth=2.4,
)

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1,
    alpha=0.6,
)

ax.set_title(
    "Mean-Reversion Strategy vs Equal-Weighted Benchmark",
    fontsize=16,
    fontweight="bold",
    pad=18,
)

ax.set_xlabel("Evaluation Date", fontsize=11)

ax.set_ylabel(
    "Portfolio Growth (Initial Value = 1.0)",
    fontsize=11,
)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

ax.grid(
    True,
    alpha=0.25,
)

ax.legend(
    loc="upper left",
    frameon=True,
)

# Final performance annotations
ax.annotate(
    f"Strategy: {strategy_total_return:.2%}",
    xy=(portfolio.index[-1], portfolio["Strategy_Equity"].iloc[-1]),
    xytext=(-135, 12),
    textcoords="offset points",
    fontsize=10,
    fontweight="bold",
)

ax.annotate(
    f"Benchmark: {benchmark_total_return:.2%}",
    xy=(portfolio.index[-1], portfolio["Benchmark_Equity"].iloc[-1]),
    xytext=(-145, 12),
    textcoords="offset points",
    fontsize=10,
    fontweight="bold",
)

fig.text(
    0.01,
    0.01,
    "10 US equities | Equal-weighted daily rebalancing | "
    "2024–2025 | Transaction cost: 0.10% per turnover",
    fontsize=9,
    alpha=0.75,
)

fig.tight_layout(rect=[0, 0.04, 1, 1])

fig.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# REPORT
# ============================================================

print("=" * 60)
print("PORTFOLIO PERFORMANCE VISUAL GENERATED")
print("=" * 60)

print(f"Observations: {len(portfolio)}")
print(f"Stocks: {df['ticker'].nunique()}")
print(f"Start date: {portfolio.index.min().date()}")
print(f"End date: {portfolio.index.max().date()}")

print(f"\nStrategy return: {strategy_total_return:.4%}")
print(f"Benchmark return: {benchmark_total_return:.4%}")

print(f"\nSaved to: {OUTPUT_FILE}")
