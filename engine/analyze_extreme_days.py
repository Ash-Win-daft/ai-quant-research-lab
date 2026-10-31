
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

INPUT_FILE = REPORTS / "aapl_mean_reversion_oos_daily.csv"
OUTPUT_FILE = REPORTS / "extreme_day_analysis.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = [
        "Date",
        "Asset_Return",
        "Position",
        "Transaction_Cost",
        "Gross_Return",
        "Net_Return",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="raise")

    numeric_columns = [
        "Asset_Return",
        "Position",
        "Transaction_Cost",
        "Gross_Return",
        "Net_Return",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    df = df.dropna(subset=required).sort_values("Date").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid daily observations found.")

    # Rank the strategy's best and worst days by net return.
    best = df.nlargest(10, "Net_Return").copy()
    best["Extreme_Group"] = "Best 10"

    worst = df.nsmallest(10, "Net_Return").copy()
    worst["Extreme_Group"] = "Worst 10"

    result = pd.concat([best, worst], ignore_index=True)
    result = result.sort_values(
        ["Extreme_Group", "Net_Return"],
        ascending=[True, False],
    )

    result["Asset_Return_Pct"] = result["Asset_Return"] * 100
    result["Gross_Return_Pct"] = result["Gross_Return"] * 100
    result["Net_Return_Pct"] = result["Net_Return"] * 100
    result["Transaction_Cost_Pct"] = result["Transaction_Cost"] * 100

    output_columns = [
        "Extreme_Group",
        "Date",
        "Position",
        "Asset_Return_Pct",
        "Gross_Return_Pct",
        "Transaction_Cost_Pct",
        "Net_Return_Pct",
    ]

    result[output_columns].to_csv(OUTPUT_FILE, index=False)

    display = result[output_columns].copy()
    for column in [
        "Asset_Return_Pct",
        "Gross_Return_Pct",
        "Transaction_Cost_Pct",
        "Net_Return_Pct",
    ]:
        display[column] = display[column].map(lambda value: f"{value:+.3f}%")

    print("\nEXTREME-DAY ANALYSIS")
    print("=" * 100)
    print(display.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
