from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_PATH = (
    REPORTS_DIR / "multi_stock_mean_reversion_open_summary.csv"
)

OUTPUT_PATH = (
    REPORTS_DIR / "multi_stock_cross_sectional_statistics.csv"
)

SECTOR_OUTPUT_PATH = (
    REPORTS_DIR / "multi_stock_sector_statistics.csv"
)


# --------------------------------------------------
# LOAD AND VALIDATE EXPERIMENT RESULTS
# --------------------------------------------------

def load_results():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Experiment summary not found: {INPUT_PATH}"
        )

    results = pd.read_csv(INPUT_PATH)

    required_columns = {
        "ticker",
        "sector",
        "Strategy_Total Return",
        "BuyHold_Total Return",
        "Strategy_Sharpe Ratio (rf=0)",
        "BuyHold_Sharpe Ratio (rf=0)",
        "Strategy_Maximum Drawdown",
        "BuyHold_Maximum Drawdown",
        "Excess_Total_Return",
        "Beats_Buy_Hold",
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if results.empty:
        raise ValueError("Experiment summary is empty.")

    if results["ticker"].duplicated().any():
        raise ValueError("Duplicate ticker records found.")

    numeric_columns = [
        "Strategy_Total Return",
        "BuyHold_Total Return",
        "Strategy_Sharpe Ratio (rf=0)",
        "BuyHold_Sharpe Ratio (rf=0)",
        "Strategy_Maximum Drawdown",
        "BuyHold_Maximum Drawdown",
        "Excess_Total_Return",
    ]

    for column in numeric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="coerce",
        )

        if results[column].isna().any():
            raise ValueError(
                f"Missing or invalid numeric values in {column}."
            )

        if not np.isfinite(results[column]).all():
            raise ValueError(
                f"Non-finite values found in {column}."
            )

    results["Beats_Buy_Hold"] = (
        results["Beats_Buy_Hold"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )

    if results["Beats_Buy_Hold"].isna().any():
        raise ValueError("Invalid Beats_Buy_Hold values.")

    # Confirm the stored excess returns are consistent.
    calculated_excess = (
        results["Strategy_Total Return"]
        - results["BuyHold_Total Return"]
    )

    if not np.allclose(
        calculated_excess,
        results["Excess_Total_Return"],
        atol=1e-8,
    ):
        raise ValueError(
            "Stored excess returns do not match the return differences."
        )

    return results


# --------------------------------------------------
# CROSS-STOCK DESCRIPTIVE STATISTICS
# --------------------------------------------------

def calculate_cross_stock_statistics(results):

    metrics = {
        "Strategy Total Return": "Strategy_Total Return",
        "Buy-and-Hold Total Return": "BuyHold_Total Return",
        "Excess Total Return": "Excess_Total_Return",
        "Strategy Sharpe Ratio": "Strategy_Sharpe Ratio (rf=0)",
        "Buy-and-Hold Sharpe Ratio": "BuyHold_Sharpe Ratio (rf=0)",
        "Strategy Maximum Drawdown": "Strategy_Maximum Drawdown",
        "Buy-and-Hold Maximum Drawdown": "BuyHold_Maximum Drawdown",
    }

    records = []

    for metric_name, column in metrics.items():

        values = results[column]

        records.append({
            "Metric": metric_name,
            "Mean": values.mean(),
            "Median": values.median(),
            "Standard Deviation": values.std(ddof=1),
            "Minimum": values.min(),
            "25th Percentile": values.quantile(0.25),
            "75th Percentile": values.quantile(0.75),
            "Maximum": values.max(),
        })

    statistics = pd.DataFrame(records)

    return statistics


# --------------------------------------------------
# SECTOR-LEVEL DESCRIPTIVE STATISTICS
# --------------------------------------------------

def calculate_sector_statistics(results):

    sector_statistics = (
        results.groupby("sector")
        .agg(
            Stocks=("ticker", "count"),
            Mean_Strategy_Return=("Strategy_Total Return", "mean"),
            Median_Strategy_Return=("Strategy_Total Return", "median"),
            Mean_BuyHold_Return=("BuyHold_Total Return", "mean"),
            Median_BuyHold_Return=("BuyHold_Total Return", "median"),
            Mean_Excess_Return=("Excess_Total_Return", "mean"),
            Median_Excess_Return=("Excess_Total_Return", "median"),
            Mean_Strategy_Sharpe=(
                "Strategy_Sharpe Ratio (rf=0)", "mean"
            ),
            Mean_BuyHold_Sharpe=(
                "BuyHold_Sharpe Ratio (rf=0)", "mean"
            ),
            Stocks_Beating_BuyHold=("Beats_Buy_Hold", "sum"),
        )
        .reset_index()
    )

    sector_statistics["Percent_Beating_BuyHold"] = (
        sector_statistics["Stocks_Beating_BuyHold"]
        / sector_statistics["Stocks"]
        * 100
    )

    return sector_statistics


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 75)
    print("CROSS-STOCK ROBUSTNESS: DESCRIPTIVE ANALYSIS")
    print("=" * 75)

    results = load_results()

    statistics = calculate_cross_stock_statistics(results)
    sector_statistics = calculate_sector_statistics(results)

    statistics.to_csv(OUTPUT_PATH, index=False)
    sector_statistics.to_csv(SECTOR_OUTPUT_PATH, index=False)

    outperforming = int(results["Beats_Buy_Hold"].sum())
    underperforming = len(results) - outperforming

    print("\nUNIVERSE SUMMARY")
    print("-" * 75)
    print("Stocks analysed:", len(results))
    print("Stocks outperforming buy-and-hold:", outperforming)
    print("Stocks underperforming buy-and-hold:", underperforming)

    print(
        "Percentage outperforming buy-and-hold:",
        f"{outperforming / len(results) * 100:.2f}%",
    )

    print("\nCROSS-STOCK STATISTICS")
    print("-" * 75)
    print(statistics.round(4).to_string(index=False))

    print("\nSECTOR-LEVEL STATISTICS")
    print("-" * 75)
    print(sector_statistics.round(4).to_string(index=False))

    print("\nSTOCKS RANKED BY EXCESS RETURN")
    print("-" * 75)

    ranked = results[
        [
            "ticker",
            "sector",
            "Strategy_Total Return",
            "BuyHold_Total Return",
            "Excess_Total_Return",
        ]
    ].sort_values(
        "Excess_Total_Return",
        ascending=False,
    )

    print(ranked.round(4).to_string(index=False))

    print("\nREPORTS SAVED")
    print("-" * 75)
    print(OUTPUT_PATH)
    print(SECTOR_OUTPUT_PATH)

    print("\nCROSS-STOCK DESCRIPTIVE ANALYSIS COMPLETED.")


if __name__ == "__main__":
    main()