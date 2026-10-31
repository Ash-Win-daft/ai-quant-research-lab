
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "reports" / "aapl_mean_reversion_oos_daily.csv"
OUTPUT_FILE = ROOT / "reports" / "april_2025_regime_analysis.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    required = [
        "Date",
        "Close",
        "Asset_Return",
        "Signal",
        "Position",
        "Turnover",
        "Transaction_Cost",
        "Net_Return",
        "Strategy_Equity",
        "Buy_Hold_Equity",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="raise")

    start = pd.Timestamp("2025-04-01")
    end = pd.Timestamp("2025-04-30")

    april = df.loc[df["Date"].between(start, end), required].copy()

    if april.empty:
        raise ValueError("No April 2025 observations found.")

    april["Asset_Return_Pct"] = april["Asset_Return"] * 100
    april["Net_Return_Pct"] = april["Net_Return"] * 100
    april["Transaction_Cost_Pct"] = april["Transaction_Cost"] * 100

    output_columns = [
        "Date",
        "Close",
        "Asset_Return_Pct",
        "Signal",
        "Position",
        "Turnover",
        "Transaction_Cost_Pct",
        "Net_Return_Pct",
        "Strategy_Equity",
        "Buy_Hold_Equity",
    ]

    april[output_columns].to_csv(OUTPUT_FILE, index=False)

    display = april[output_columns].copy()
    for col in [
        "Asset_Return_Pct",
        "Net_Return_Pct",
        "Transaction_Cost_Pct",
    ]:
        display[col] = display[col].map(lambda x: f"{x:+.3f}%")

    print("\nAAPL — APRIL 2025 REGIME ANALYSIS")
    print("=" * 110)
    print(display.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
