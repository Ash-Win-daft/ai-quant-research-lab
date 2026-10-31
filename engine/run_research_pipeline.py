
from pathlib import Path
import subprocess
import sys


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


# ============================================================
# REQUIRED SOURCE FILES
# ============================================================

REQUIRED_INPUT_REPORTS = [
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
]


# ============================================================
# EXPECTED GENERATED FILES
# ============================================================

EXPECTED_OUTPUTS = [
    REPORTS_DIR / "registry_summary.csv",
    REPORTS_DIR / "consolidated_research_report.md",
    REPORTS_DIR / "comprehensive_research_report.md",
    PROJECT_ROOT / "AI_Quant_Research_Lab_Final_Report_v2.pdf",
]


# ============================================================
# VALIDATION
# ============================================================

def check_input_reports():
    """Check that all source reports exist and are non-empty."""

    missing = []

    for filename in REQUIRED_INPUT_REPORTS:
        path = REPORTS_DIR / filename

        if not path.is_file() or path.stat().st_size == 0:
            missing.append(filename)

    if missing:
        raise FileNotFoundError(
            "Required input reports are missing or empty:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )

    print(
        f"Input report check: PASSED "
        f"({len(REQUIRED_INPUT_REPORTS)} files)",
        flush=True,
    )


def check_output(path):
    """Verify that a generated output exists and is non-empty."""

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(
            f"Expected output was not generated correctly: {path}"
        )

    print(f"Verified: {path.relative_to(PROJECT_ROOT)}", flush=True)


# ============================================================
# MODULE EXECUTION
# ============================================================

def run_module(module_name):
    """Run a project module using the current Python interpreter."""

    print(f"\nRunning {module_name}...", flush=True)

    subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        check=True,
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60, flush=True)
    print("AI QUANT RESEARCH LAB", flush=True)
    print("UNIFIED RESEARCH PIPELINE", flush=True)
    print("=" * 60, flush=True)

    # 1. Validate original research inputs.
    check_input_reports()

    # 2. Regenerate experiment registry summary.
    run_module("engine.registry_summary")
    check_output(EXPECTED_OUTPUTS[0])

    # 3. Generate registry-based Markdown report.
    run_module("engine.generate_registry_report")
    check_output(EXPECTED_OUTPUTS[1])

    # 4. Generate comprehensive research report.
    run_module("engine.generate_final_report")
    check_output(EXPECTED_OUTPUTS[2])

    # 5. Generate the final PDF.
    run_module("engine.generate_pdf_report")
    check_output(EXPECTED_OUTPUTS[3])

    # 6. Final verification.
    print("\n" + "=" * 60, flush=True)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 60, flush=True)

    for path in EXPECTED_OUTPUTS:
        size_kb = path.stat().st_size / 1024
        print(
            f"{path.name}: {size_kb:.2f} KB",
            flush=True,
        )


if __name__ == "__main__":
    main()
