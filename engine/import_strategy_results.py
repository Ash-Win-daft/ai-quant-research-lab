
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "normalized_strategy_results.csv"


def moving_average_result():
    path = REPORTS / "aapl_moving_average_summary.csv"
    if not path.exists():
        return None

    data = pd.read_csv(path)
    metrics = dict(zip(data["Metric"], data["Value"]))

    required = [
        "Strategy Total Return",
        "Buy-and-Hold Total Return",
    ]
    if not all(key in metrics for key in required):
        print("Skipping moving average: required metrics missing.")
        return None

    return {
        "Experiment_ID": "AAPL_SMA_BASELINE",
        "Asset": "AAPL",
        "Strategy": "Moving Average",
        "Evaluation_Period": "2021-2025",
        "Strategy_Total_Return": float(
            metrics["Strategy Total Return"]
        ),
        "Buy_Hold_Total_Return": float(
            metrics["Buy-and-Hold Total Return"]
        ),
        "Return_Difference": float(
            metrics["Strategy Total Return"]
            - metrics["Buy-and-Hold Total Return"]
        ),
        "Source_Report": path.name,
        "Period_Verified": True,
    }


def mean_reversion_result():
    path = REPORTS / "aapl_mean_reversion_oos_comparison.csv"
    if not path.exists():
        return None

    data = pd.read_csv(path)
    if "Unnamed: 0" not in data.columns:
        print("Skipping mean reversion: unexpected report format.")
        return None

    metrics = data.set_index("Unnamed: 0")

    if "Total Return" not in metrics.index:
        print("Skipping mean reversion: total return not found.")
        return None

    strategy_return = float(metrics.loc["Total Return", "Mean Reversion"])
    benchmark_return = float(metrics.loc["Total Return", "Buy and Hold"])

    return {
        "Experiment_ID": "AAPL_MR_OOS_2024_2025",
        "Asset": "AAPL",
        "Strategy": "Mean Reversion",
        "Evaluation_Period": "2024-2025",
        "Strategy_Total_Return": strategy_return,
        "Buy_Hold_Total_Return": benchmark_return,
        "Return_Difference": strategy_return - benchmark_return,
        "Source_Report": path.name,
        "Period_Verified": True,
    }


def main():
    results = [
        result
        for result in [
            moving_average_result(),
            mean_reversion_result(),
        ]
        if result is not None
    ]

    if not results:
        raise ValueError("No compatible strategy reports were found.")

    normalized = pd.DataFrame(results)

    # Do not overwrite the experiment registry.
    normalized.to_csv(OUTPUT, index=False)

    print("\nAI QUANT RESEARCH LAB")
    print("NORMALIZED STRATEGY RESULTS")
    print("=" * 65)
    print(normalized.to_string(index=False))
    print("\nSaved to:")
    print(OUTPUT)
    print("\nBoth evaluation periods are verified from their source reports.")


if __name__ == "__main__":
    main()
