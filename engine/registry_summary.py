
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_FILE = PROJECT_ROOT / "reports" / "experiment_registry.csv"
SUMMARY_FILE = PROJECT_ROOT / "reports" / "registry_summary.csv"


def generate_summary():
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_FILE}")

    registry = pd.read_csv(REGISTRY_FILE)

    if registry.empty:
        raise ValueError("The experiment registry is empty.")

    required = [
        "Experiment_ID",
        "Asset",
        "Strategy",
        "Evaluation_Period",
        "Strategy_Total_Return",
        "Buy_Hold_Total_Return",
        "Strategy_Sharpe",
        "Buy_Hold_Sharpe",
        "Strategy_Max_Drawdown",
        "HAC_P_Value",
        "Decision",
    ]

    missing = [c for c in required if c not in registry.columns]
    if missing:
        raise ValueError(f"Registry is missing columns: {missing}")

    registry["Return_Difference"] = (
        registry["Strategy_Total_Return"]
        - registry["Buy_Hold_Total_Return"]
    )
    registry["Sharpe_Difference"] = (
        registry["Strategy_Sharpe"]
        - registry["Buy_Hold_Sharpe"]
    )

    # Unknown metrics remain unknown rather than being labelled False.
    registry["Beats_Benchmark_Return"] = (
        registry["Return_Difference"].gt(0).astype("boolean")
    ).where(registry["Return_Difference"].notna(), pd.NA)

    registry["Beats_Benchmark_Sharpe"] = (
        registry["Sharpe_Difference"].gt(0).astype("boolean")
    ).where(registry["Sharpe_Difference"].notna(), pd.NA)

    registry["Statistically_Significant_At_5pct"] = (
        registry["HAC_P_Value"].lt(0.05).astype("boolean")
    ).where(registry["HAC_P_Value"].notna(), pd.NA)

    summary_columns = [
        "Experiment_ID",
        "Asset",
        "Strategy",
        "Evaluation_Period",
        "Strategy_Total_Return",
        "Buy_Hold_Total_Return",
        "Return_Difference",
        "Strategy_Sharpe",
        "Buy_Hold_Sharpe",
        "Sharpe_Difference",
        "Strategy_Max_Drawdown",
        "HAC_P_Value",
        "Beats_Benchmark_Return",
        "Beats_Benchmark_Sharpe",
        "Statistically_Significant_At_5pct",
        "Decision",
    ]

    summary = registry[summary_columns].copy()
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\nAI QUANT RESEARCH LAB")
    print("REGISTRY SUMMARY")
    print("=" * 90)
    print(summary.to_string(index=False, na_rep="UNKNOWN"))

    print("\nEXPERIMENT COUNTS")
    print("-" * 70)
    print(f"Total experiments: {len(summary)}")
    print(f"Rejected: {(summary['Decision'] == 'REJECT').sum()}")
    print(
        "Return comparison available: "
        f"{summary['Return_Difference'].notna().sum()}"
    )
    print(
        "Sharpe comparison available: "
        f"{summary['Sharpe_Difference'].notna().sum()}"
    )
    print(
        "HAC significance tests available: "
        f"{summary['HAC_P_Value'].notna().sum()}"
    )
    print(f"\nSummary saved to: {SUMMARY_FILE}")


if __name__ == "__main__":
    generate_summary()
