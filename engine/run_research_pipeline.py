
from pathlib import Path
import subprocess
import sys

import pandas as pd

from data.validate_market_data import validate_market_data


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ============================================================
# REQUIRED INPUT REPORTS
# ============================================================

REQUIRED_INPUT_REPORTS = [
    # Original AAPL research
    "experiment_registry.csv",
    "aapl_moving_average_summary.csv",
    "aapl_parameter_robustness.csv",
    "robustness_results.csv",
    "strategy_dependency_analysis.csv",
    "aapl_mean_reversion_oos_comparison.csv",
    "aapl_mean_reversion_hac_significance.csv",
    "aapl_mean_reversion_transaction_cost_stress.csv",
    "aapl_mean_reversion_open_comparison.csv",
    "ml_prediction_metrics.csv",
    "ml_performance.csv",
    "walk_forward_metrics.csv",
    "walk_forward_performance.csv",
    "best_day_dependency_comparison.csv",
    "extreme_day_analysis.csv",
    "april_2025_regime_analysis.csv",

    # Multi-stock research
    "multi_stock_validation_summary.csv",
    "multi_stock_mean_reversion_open_summary.csv",
    "multi_stock_mean_reversion_open_daily.csv",
    "multi_stock_cross_sectional_statistics.csv",
    "multi_stock_sector_statistics.csv",
    "multi_stock_cost_sensitivity_detailed.csv",
    "multi_stock_cost_sensitivity_summary.csv",
    "multi_stock_hac_significance.csv",
    "multi_stock_portfolio_hac_significance.csv",
    "multi_stock_hac_lag_sensitivity.csv",
]


# ============================================================
# EXPECTED GENERATED OUTPUTS
# ============================================================

EXPECTED_OUTPUTS = [
    REPORTS_DIR / "registry_summary.csv",
    REPORTS_DIR / "consolidated_research_report.md",
    REPORTS_DIR / "comprehensive_research_report.md",
    PROJECT_ROOT / "AI_Quant_Research_Lab_Final_Report_v2.pdf",
]


# ============================================================
# 1. VALIDATE MARKET DATA
# ============================================================

def validate_market_datasets():
    """Validate original AAPL historical market datasets."""

    print("\nValidating original AAPL market data...", flush=True)

    validate_market_data(DATA_DIR / "aapl_daily.csv")

    validate_market_data(DATA_DIR / "aapl_daily_extended.csv")

    adjusted_data = DATA_DIR / "aapl_adjusted_close.csv"

    if not adjusted_data.is_file() or adjusted_data.stat().st_size == 0:
        raise FileNotFoundError(
            f"Adjusted-price dataset is missing or empty: {adjusted_data}"
        )

    print("Original AAPL market data validation completed.", flush=True)


# ============================================================
# 2. RUN PROJECT MODULES
# ============================================================

def run_module(module_name):
    """Execute a project module using the active Python interpreter."""

    print(f"\nRunning {module_name}...", flush=True)

    subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        check=True,
    )

    print(f"Completed: {module_name}", flush=True)


# ============================================================
# 3. REGISTER MACHINE-LEARNING EXPERIMENTS
# ============================================================

def sync_ml_experiments_to_registry():
    """Synchronize static and walk-forward ML results with the registry."""

    print("\nSynchronizing ML experiments with registry...", flush=True)

    registry_path = REPORTS_DIR / "experiment_registry.csv"

    static_path = REPORTS_DIR / "ml_performance.csv"
    walk_forward_path = REPORTS_DIR / "walk_forward_performance.csv"

    registry = pd.read_csv(registry_path)

    static = pd.read_csv(static_path, index_col=0)
    walk_forward = pd.read_csv(walk_forward_path, index_col=0)

    def get_metrics(performance, strategy_name):
        return {
            "Strategy_Total_Return": performance.loc[
                "Total Return", strategy_name
            ],
            "Buy_Hold_Total_Return": performance.loc[
                "Total Return", "Buy and Hold"
            ],
            "Strategy_Sharpe": performance.loc[
                "Sharpe Ratio", strategy_name
            ],
            "Buy_Hold_Sharpe": performance.loc[
                "Sharpe Ratio", "Buy and Hold"
            ],
            "Strategy_Max_Drawdown": performance.loc[
                "Max Drawdown", strategy_name
            ],
        }

    static_metrics = get_metrics(static, "ML Strategy")

    walk_forward_metrics = get_metrics(
        walk_forward, "Walk-Forward ML"
    )

    new_records = [
        {
            "Experiment_ID": "AAPL_STATIC_ML_OOS_2024_2025",
            "Asset": "AAPL",
            "Strategy": "Static Logistic Regression",
            "Hypothesis": (
                "Historical price and volatility features may predict "
                "the direction of the next trading session."
            ),
            "Training_Period": "2021-2023",
            "Evaluation_Period": "2024-2025",
            "Reference_Transaction_Cost": 0.001,
            **static_metrics,
            "HAC_P_Value": None,
            "Decision": "REJECT",
            "Reason": (
                "Underperformed buy-and-hold on total return and Sharpe. "
                "ROC AUC was below 0.50 and Brier score was worse than "
                "the historical-rate baseline. No predictive edge "
                "was demonstrated."
            ),
        },
        {
            "Experiment_ID": "AAPL_WALK_FORWARD_ML_OOS_2024_2025",
            "Asset": "AAPL",
            "Strategy": "Walk-Forward Logistic Regression",
            "Hypothesis": (
                "Repeatedly retraining a logistic regression model "
                "using expanding historical data may improve "
                "next-session return-direction predictions."
            ),
            "Training_Period": "Expanding historical window",
            "Evaluation_Period": "2024-2025",
            "Reference_Transaction_Cost": 0.001,
            **walk_forward_metrics,
            "HAC_P_Value": None,
            "Decision": "REJECT",
            "Reason": (
                "Underperformed buy-and-hold on total return and Sharpe. "
                "ROC AUC was below 0.50 and Brier score was worse than "
                "the historical-rate baseline. Expanding-window "
                "retraining did not establish a predictive edge."
            ),
        },
    ]

    new_ids = {
        record["Experiment_ID"]
        for record in new_records
    }

    registry = registry[
        ~registry["Experiment_ID"].isin(new_ids)
    ]

    registry = pd.concat(
        [registry, pd.DataFrame(new_records)],
        ignore_index=True,
    )

    registry.to_csv(registry_path, index=False)

    print(
        f"Registry synchronized: {len(new_records)} ML experiments.",
        flush=True,
    )


# ============================================================
# 4. VALIDATE REQUIRED RESEARCH REPORTS
# ============================================================

def validate_required_reports():
    """Check that all required research reports exist and are non-empty."""

    print("\nValidating research report inputs...", flush=True)

    missing = []

    for filename in REQUIRED_INPUT_REPORTS:
        path = REPORTS_DIR / filename

        if not path.is_file() or path.stat().st_size == 0:
            missing.append(filename)

    if missing:
        raise FileNotFoundError(
            "Required research reports are missing or empty:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )

    print(
        f"Research report validation: PASSED "
        f"({len(REQUIRED_INPUT_REPORTS)} files)",
        flush=True,
    )


# ============================================================
# 5. VERIFY GENERATED OUTPUTS
# ============================================================

def check_output(path):
    """Verify that a generated output exists and is non-empty."""

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(
            f"Expected output was not generated correctly: {path}"
        )

    print(
        f"Verified: {path.relative_to(PROJECT_ROOT)}",
        flush=True,
    )


# ============================================================
# MAIN RESEARCH PIPELINE
# ============================================================

def main():

    print("=" * 70, flush=True)
    print("AI QUANT RESEARCH LAB", flush=True)
    print("UNIFIED RESEARCH PIPELINE", flush=True)
    print("=" * 70, flush=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # STAGE 1: Original AAPL market data validation
    # --------------------------------------------------------

    print("\nSTAGE 1: ORIGINAL AAPL DATA VALIDATION", flush=True)

    validate_market_datasets()

    # --------------------------------------------------------
    # STAGE 2: Multi-stock data validation
    # --------------------------------------------------------

    print("\nSTAGE 2: MULTI-STOCK DATA VALIDATION", flush=True)

    run_module("data.validate_multi_stock_data")

    # --------------------------------------------------------
    # STAGE 3: Single-stock machine-learning experiments
    # --------------------------------------------------------

    print("\nSTAGE 3: MACHINE-LEARNING EXPERIMENTS", flush=True)

    run_module("engine.run_ml_experiment")

    run_module("engine.run_walk_forward_experiment")

    sync_ml_experiments_to_registry()

    # --------------------------------------------------------
    # STAGE 4: Multi-stock mean-reversion experiment
    # --------------------------------------------------------

    print("\nSTAGE 4: MULTI-STOCK MEAN-REVERSION RESEARCH", flush=True)

    run_module("engine.run_multi_stock_mean_reversion")

    # --------------------------------------------------------
    # STAGE 5: Multi-stock descriptive analysis
    # --------------------------------------------------------

    print("\nSTAGE 5: CROSS-SECTIONAL ANALYSIS", flush=True)

    run_module("engine.analyze_multi_stock_results")

    # --------------------------------------------------------
    # STAGE 6: Multi-stock transaction-cost sensitivity
    # --------------------------------------------------------

    print("\nSTAGE 6: TRANSACTION-COST SENSITIVITY", flush=True)

    run_module("engine.analyze_multi_stock_cost_sensitivity")

    # --------------------------------------------------------
    # STAGE 7: Multi-stock statistical significance
    # --------------------------------------------------------

    print("\nSTAGE 7: MULTI-STOCK HAC SIGNIFICANCE", flush=True)

    run_module("engine.analyze_multi_stock_significance")

    # --------------------------------------------------------
    # STAGE 8: HAC lag sensitivity
    # --------------------------------------------------------

    print("\nSTAGE 8: HAC LAG SENSITIVITY", flush=True)

    run_module("engine.analyze_hac_lag_sensitivity")

    # --------------------------------------------------------
    # STAGE 9: Validate all research report inputs
    # --------------------------------------------------------

    print("\nSTAGE 9: RESEARCH REPORT VALIDATION", flush=True)

    validate_required_reports()

    # --------------------------------------------------------
    # STAGE 10: Regenerate experiment registry summary
    # --------------------------------------------------------

    print("\nSTAGE 10: EXPERIMENT REGISTRY", flush=True)

    run_module("engine.registry_summary")

    check_output(EXPECTED_OUTPUTS[0])

    # --------------------------------------------------------
    # STAGE 11: Generate consolidated Markdown report
    # --------------------------------------------------------

    print("\nSTAGE 11: CONSOLIDATED RESEARCH REPORT", flush=True)

    run_module("engine.generate_registry_report")

    check_output(EXPECTED_OUTPUTS[1])

    # --------------------------------------------------------
    # STAGE 12: Generate comprehensive Markdown report
    # --------------------------------------------------------

    print("\nSTAGE 12: COMPREHENSIVE RESEARCH REPORT", flush=True)

    run_module("engine.generate_final_report")

    check_output(EXPECTED_OUTPUTS[2])

    # --------------------------------------------------------
    # STAGE 13: Generate final PDF
    # --------------------------------------------------------

    print("\nSTAGE 13: FINAL PDF GENERATION", flush=True)

    run_module("engine.generate_pdf_report")

    check_output(EXPECTED_OUTPUTS[3])

    # --------------------------------------------------------
    # STAGE 14: Final verification
    # --------------------------------------------------------

    print("\n" + "=" * 70, flush=True)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 70, flush=True)

    print("\nGENERATED OUTPUTS:")

    for path in EXPECTED_OUTPUTS:
        size_kb = path.stat().st_size / 1024

        print(
            f"{path.name}: {size_kb:.2f} KB",
            flush=True,
        )

    print("\nResearch pipeline execution finished.", flush=True)


if __name__ == "__main__":
    main()
