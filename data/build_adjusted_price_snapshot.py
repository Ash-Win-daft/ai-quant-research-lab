
from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "aapl_adjusted_close.csv"


def build_snapshot():
    """Download and save a fixed AAPL adjusted-close dataset."""

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Snapshot already exists: {OUTPUT_PATH}\n"
            "Delete it manually only if you intentionally want "
            "to create a replacement."
        )

    print("Downloading AAPL historical data...")

    raw = yf.download(
        "AAPL",
        start="2021-01-01",
        end="2026-01-01",
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            "No data downloaded. Check your internet connection."
        )

    # Handle possible multi-level column names.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.loc[:, ~raw.columns.duplicated()].sort_index()

    if "Adj Close" not in raw.columns:
        raise ValueError("Adjusted Close is unavailable.")

    prices = raw["Adj Close"].dropna()

    if prices.empty:
        raise ValueError("Adjusted Close contains no valid prices.")

    if prices.isna().any() or (prices <= 0).any():
        raise ValueError("Invalid adjusted prices detected.")

    output = prices.rename("Adj_Close").to_frame()
    output.index.name = "Date"

    output.to_csv(OUTPUT_PATH)

    print("\nAdjusted-price snapshot created successfully.")
    print(f"File: {OUTPUT_PATH}")
    print(f"Observations: {len(output)}")
    print(f"First date: {output.index.min().date()}")
    print(f"Last date: {output.index.max().date()}")


if __name__ == "__main__":
    build_snapshot()
