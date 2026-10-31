
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

INPUT_FILE = REPORTS / "aapl_mean_reversion_oos_daily.csv"
OUTPUT_FILE = REPORTS / "best_day_dependency_comparison.csv"


def compounded_return(returns):
    if returns.empty:
        return float("nan")
    if (returns <= -1).any():
        raise ValueError("Returns must be greater than -100%.")
    return (1 + returns).prod() - 1


def run_dependency_test(df, return_column, strategy_name):
    returns = df[return_column]
    baseline = compounded_return(returns)

    rows = [{
        "Series": strategy_name,
        "Test": "Baseline",
        "Days_Removed": 0,
        "Total_Return": baseline,
        "Return_Change": 0.0,
        "Observations_Remaining": len(returns),
    }]

    ranked = df.sort_values(return_column, ascending=False)

    for k in [1, 3, 5, 10]:
        if k >= len(df):
            continue

        removed_dates = set(ranked.head(k)["Date"])
        remaining = df.loc[~df["Date"].isin(removed_dates), return_column]
        adjusted_return = compounded_return(remaining)

        rows.append({
            "Series": strategy_name,
            "Test": f"Remove top {k} days",
            "Days_Removed": k,
            "Total_Return": adjusted_return,
            "Return_Change": adjusted_return - baseline,
            "Observations_Remaining": len(remaining),
        })

    return rows


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = ["Date", "Net_Return", "Asset_Return"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="raise")

    for column in ["Net_Return", "Asset_Return"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    df = (
        df.dropna(subset=required)
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("No valid observations found.")

    results = []
    results.extend(
        run_dependency_test(df, "Net_Return", "Mean Reversion Strategy")
    )
    results.extend(
        run_dependency_test(df, "Asset_Return", "Buy and Hold")
    )

    result = pd.DataFrame(results)
    result.to_csv(OUTPUT_FILE, index=False)

    display = result.copy()
    display["Total_Return"] = display["Total_Return"].map(
        lambda value: f"{value:.2%}"
    )
    display["Return_Change"] = display["Return_Change"].map(
        lambda value: f"{value:+.2%}"
    )

    print("\nBEST-DAY DEPENDENCY COMPARISON")
    print("=" * 95)
    print(display.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
