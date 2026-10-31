
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REGISTRY_PATH = REPORTS_DIR / "experiment_registry.csv"

COLUMNS = [
    "Experiment_ID",
    "Asset",
    "Strategy",
    "Hypothesis",
    "Training_Period",
    "Evaluation_Period",
    "Reference_Transaction_Cost",
    "Strategy_Total_Return",
    "Buy_Hold_Total_Return",
    "Strategy_Sharpe",
    "Buy_Hold_Sharpe",
    "Strategy_Max_Drawdown",
    "HAC_P_Value",
    "Decision",
    "Reason",
]


def build_records():
    records = []

    # Existing, fully evaluated mean-reversion experiment.
    evaluation_file = REPORTS_DIR / "aapl_mean_reversion_evaluation.csv"
    hac_file = REPORTS_DIR / "aapl_mean_reversion_hac_significance.csv"

    if not evaluation_file.exists() or not hac_file.exists():
        raise FileNotFoundError(
            "Mean-reversion evaluation or HAC report is missing."
        )

    evaluation = pd.read_csv(evaluation_file)
    hac = pd.read_csv(hac_file)

    if evaluation.empty or hac.empty:
        raise ValueError("Mean-reversion source report is empty.")

    e = evaluation.iloc[0]
    h = hac.iloc[0]

    records.append({
        "Experiment_ID": "AAPL_MR_OOS_2024_2025",
        "Asset": "AAPL",
        "Strategy": "Mean Reversion",
        "Hypothesis": (
            "Prices sufficiently below a rolling average "
            "may recover toward that average."
        ),
        "Training_Period": "2021-2023",
        "Evaluation_Period": "2024-2025",
        "Reference_Transaction_Cost": 0.001,
        "Strategy_Total_Return": e.get("Strategy_Total_Return"),
        "Buy_Hold_Total_Return": e.get("Buy_Hold_Total_Return"),
        "Strategy_Sharpe": e.get("Strategy_Sharpe"),
        "Buy_Hold_Sharpe": e.get("Buy_Hold_Sharpe"),
        "Strategy_Max_Drawdown": e.get("Strategy_Max_Drawdown"),
        "HAC_P_Value": h.get("HAC_P_Value"),
        "Decision": "REJECT",
        "Reason": (
            "Underperformed buy-and-hold on total return and Sharpe; "
            "HAC test found no statistically significant mean daily "
            "excess return."
        ),
    })

    # Moving-average baseline: import only known, verified metrics.
    normalized_file = REPORTS_DIR / "normalized_strategy_results.csv"

    if not normalized_file.exists():
        raise FileNotFoundError(
            f"Missing normalized results: {normalized_file}"
        )

    normalized = pd.read_csv(normalized_file)
    sma = normalized.loc[
        normalized["Experiment_ID"] == "AAPL_SMA_BASELINE"
    ]

    if len(sma) != 1:
        raise ValueError(
            "Expected exactly one AAPL_SMA_BASELINE record."
        )

    sma = sma.iloc[0]

    if (
        str(sma["Evaluation_Period"]) != "2021-2025"
        or not bool(sma["Period_Verified"])
    ):
        raise ValueError(
            "Moving-average evaluation period is not verified."
        )

    
    # Load the existing Moving Average performance metrics.
    metrics_file = REPORTS_DIR / "aapl_performance_metrics.csv"

    if not metrics_file.exists():
        raise FileNotFoundError(
            f"Missing metrics report: {metrics_file}"
        )

    metrics = pd.read_csv(metrics_file).set_index("Unnamed: 0")

    records.append({
        "Experiment_ID": "AAPL_SMA_BASELINE",
        "Asset": "AAPL",
        "Strategy": "Moving Average",
        "Hypothesis": (
            "Moving-average trend signals may capture persistent "
            "price trends."
        ),
        "Training_Period": "Not separately defined",
        "Evaluation_Period": "2021-2025",
        "Reference_Transaction_Cost": 0.001,
        "Strategy_Total_Return": sma["Strategy_Total_Return"],
        "Buy_Hold_Total_Return": sma["Buy_Hold_Total_Return"],
        "Strategy_Sharpe": metrics.loc[
            "Sharpe Ratio (rf=0)", "Strategy"
        ],
        "Buy_Hold_Sharpe": metrics.loc[
            "Sharpe Ratio (rf=0)", "Buy and Hold"
        ],
        "Strategy_Max_Drawdown": metrics.loc[
            "Maximum Drawdown", "Strategy"
        ],
        "HAC_P_Value": None,
        "Decision": "REJECT",
        "Reason": (
            "Total return was below buy-and-hold over 2021-2025. "
            "Transaction-cost assumptions and statistical evidence "
            "are not yet verified for this registry record."
        ),
    })

    return pd.DataFrame(records, columns=COLUMNS)


def create_experiment_registry():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    new_records = build_records()

    if REGISTRY_PATH.exists():
        existing = pd.read_csv(REGISTRY_PATH)

        # Reject unexpected schemas rather than silently dropping fields.
        missing = set(COLUMNS) - set(existing.columns)
        if missing:
            raise ValueError(
                f"Existing registry is missing columns: {sorted(missing)}"
            )

        # Preserve unrelated records; replace only the two records rebuilt here.
        managed_ids = set(new_records["Experiment_ID"])
        existing = existing[
            ~existing["Experiment_ID"].isin(managed_ids)
        ]
        registry = pd.concat(
            [existing[COLUMNS], new_records],
            ignore_index=True,
        )
    else:
        registry = new_records

    registry.to_csv(REGISTRY_PATH, index=False)

    print("\nAI QUANT RESEARCH LAB")
    print("EXPERIMENT REGISTRY")
    print("=" * 70)
    print(registry.to_string(index=False))
    print(f"\nSaved to: {REGISTRY_PATH}")


if __name__ == "__main__":
    create_experiment_registry()
