
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_FILE = PROJECT_ROOT / "reports" / "experiment_registry.csv"


REQUIRED_COLUMNS = [
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


def register_experiment(experiment):
    """
    Validate and register one experiment.

    If its Experiment_ID already exists, update that record.
    Otherwise, append a new record.
    """

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in experiment
    ]

    if missing:
        raise ValueError(
            f"Experiment is missing required fields: {missing}"
        )

    if not str(experiment["Experiment_ID"]).strip():
        raise ValueError("Experiment_ID cannot be empty.")

    if not 0 <= float(
        experiment["Reference_Transaction_Cost"]
    ):
        raise ValueError("Transaction cost cannot be negative.")

    if not 0 <= float(experiment["HAC_P_Value"]) <= 1:
        raise ValueError("HAC_P_Value must be between 0 and 1.")

    REGISTRY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_record = pd.DataFrame([experiment], columns=REQUIRED_COLUMNS)

    if REGISTRY_FILE.exists():
        registry = pd.read_csv(REGISTRY_FILE)

        missing_columns = [
            column for column in REQUIRED_COLUMNS
            if column not in registry.columns
        ]

        if missing_columns:
            raise ValueError(
                "Existing registry has unexpected schema. "
                f"Missing columns: {missing_columns}"
            )

        registry = registry[
            registry["Experiment_ID"]
            != experiment["Experiment_ID"]
        ]

        registry = pd.concat(
            [registry, new_record],
            ignore_index=True,
        )
    else:
        registry = new_record

    registry.to_csv(REGISTRY_FILE, index=False)

    print(
        f"Registered experiment: "
        f"{experiment['Experiment_ID']}"
    )
    print(f"Registry now contains {len(registry)} experiment(s).")


if __name__ == "__main__":
    print("Experiment registration module is ready.")
    print("Import register_experiment() to register a new experiment.")
