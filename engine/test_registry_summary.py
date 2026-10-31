
import pandas as pd

from engine.registry_summary import generate_summary


def test_summary_preserves_experiments_and_unknown_metrics(tmp_path, monkeypatch):
    registry_file = tmp_path / "experiment_registry.csv"
    summary_file = tmp_path / "registry_summary.csv"

    registry = pd.DataFrame([
        {
            "Experiment_ID": "AAPL_MR_OOS_2024_2025",
            "Asset": "AAPL",
            "Strategy": "Mean Reversion",
            "Evaluation_Period": "2024-2025",
            "Strategy_Total_Return": 0.08,
            "Buy_Hold_Total_Return": 0.20,
            "Strategy_Sharpe": 0.30,
            "Buy_Hold_Sharpe": 0.60,
            "Strategy_Max_Drawdown": -0.15,
            "HAC_P_Value": 0.40,
            "Decision": "REJECT",
        },
        {
            "Experiment_ID": "AAPL_MR_OPEN_OOS_2024_2025",
            "Asset": "AAPL",
            "Strategy": "Mean Reversion (Open Execution)",
            "Evaluation_Period": "2024-2025",
            "Strategy_Total_Return": 0.11,
            "Buy_Hold_Total_Return": 0.47,
            "Strategy_Sharpe": 0.40,
            "Buy_Hold_Sharpe": 0.80,
            "Strategy_Max_Drawdown": -0.13,
            "HAC_P_Value": 0.34,
            "Decision": "REJECT",
        },
        {
            "Experiment_ID": "AAPL_SMA_BASELINE",
            "Asset": "AAPL",
            "Strategy": "Moving Average",
            "Evaluation_Period": "2021-2025",
            "Strategy_Total_Return": 0.33,
            "Buy_Hold_Total_Return": 1.16,
            "Strategy_Sharpe": 0.41,
            "Buy_Hold_Sharpe": 0.69,
            "Strategy_Max_Drawdown": -0.30,
            "HAC_P_Value": float("nan"),
            "Decision": "REJECT",
        },
    ])

    registry.to_csv(registry_file, index=False)

    monkeypatch.setattr(
        "engine.registry_summary.REGISTRY_FILE",
        registry_file,
    )
    monkeypatch.setattr(
        "engine.registry_summary.SUMMARY_FILE",
        summary_file,
    )

    generate_summary()

    summary = pd.read_csv(summary_file)

    assert len(summary) == 3
    assert set(summary["Experiment_ID"]) == set(registry["Experiment_ID"])

    sma = summary.loc[
        summary["Experiment_ID"] == "AAPL_SMA_BASELINE"
    ].iloc[0]

    assert pd.isna(sma["HAC_P_Value"])
    assert pd.isna(sma["Statistically_Significant_At_5pct"])

    open_mr = summary.loc[
        summary["Experiment_ID"] == "AAPL_MR_OPEN_OOS_2024_2025"
    ].iloc[0]

    assert open_mr["Beats_Benchmark_Return"] == False
    assert open_mr["Beats_Benchmark_Sharpe"] == False
    assert open_mr["Statistically_Significant_At_5pct"] == False
