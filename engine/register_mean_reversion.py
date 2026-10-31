
from pathlib import Path

import pandas as pd

from engine.register_experiment import register_experiment


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_first_row(filename):
    path = REPORTS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")

    frame = pd.read_csv(path)

    if frame.empty:
        raise ValueError(f"Report contains no rows: {path}")

    return frame.iloc[0]


def main():
    evaluation = load_first_row(
        "aapl_mean_reversion_evaluation.csv"
    )
    hac = load_first_row(
        "aapl_mean_reversion_hac_significance.csv"
    )

    experiment = {
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
        "Strategy_Total_Return": evaluation["Strategy_Total_Return"],
        "Buy_Hold_Total_Return": evaluation["Buy_Hold_Total_Return"],
        "Strategy_Sharpe": evaluation["Strategy_Sharpe"],
        "Buy_Hold_Sharpe": evaluation["Buy_Hold_Sharpe"],
        "Strategy_Max_Drawdown": evaluation["Strategy_Max_Drawdown"],
        "HAC_P_Value": hac["HAC_P_Value"],
        "Decision": "REJECT",
        "Reason": (
            "Underperformed buy-and-hold on total return and "
            "Sharpe ratio; the HAC test found no statistically "
            "significant mean daily excess return."
        ),
    }

    
    registry_path = REPORTS_DIR / "experiment_registry.csv"

    if registry_path.exists():
        registry = pd.read_csv(registry_path)

        if (
            "Experiment_ID" in registry.columns
            and experiment["Experiment_ID"]
            in registry["Experiment_ID"].astype(str).values
        ):
            raise ValueError(
                "This experiment already exists in the registry. "
                "Refusing to overwrite it automatically. "
                "Update the existing record deliberately if needed."
            )

    register_experiment(experiment)



if __name__ == "__main__":
    main()
