
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

REGISTRY_FILE = REPORTS / "experiment_registry.csv"
OUTPUT_FILE = REPORTS / "consolidated_research_report.md"


def format_pct(value):
    if pd.isna(value):
        return "Unknown"
    return f"{value:.2%}"


def format_num(value):
    if pd.isna(value):
        return "Unknown"
    return f"{value:.4f}"


def main():
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            f"Experiment registry not found: {REGISTRY_FILE}"
        )

    df = pd.read_csv(REGISTRY_FILE)

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
        "Reason",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing registry columns: {missing}")

    lines = [
        "# AI Quant Research Lab",
        "",
        "## Consolidated Research Report",
        "",
        "This report is generated from the experiment registry.",
        "Returns are historical results, not predictions of future performance.",
        "",
        "## Experiment summary",
        "",
        f"- Experiments recorded: {len(df)}",
        f"- Rejected experiments: {(df['Decision'] == 'REJECT').sum()}",
        "",
        "## Results",
        "",
        "| Experiment | Asset | Strategy | Evaluation period | Strategy return | Buy-and-hold return | Return difference | Strategy Sharpe | Buy-and-hold Sharpe | Max drawdown | HAC p-value | Decision |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for _, row in df.iterrows():
        strategy_return = row["Strategy_Total_Return"]
        benchmark_return = row["Buy_Hold_Total_Return"]

        if pd.notna(strategy_return) and pd.notna(benchmark_return):
            return_difference = strategy_return - benchmark_return
        else:
            return_difference = float("nan")

        p_value = pd.to_numeric(
            pd.Series([row["HAC_P_Value"]]), errors="coerce"
        ).iloc[0]

        
        table_values = [
            row["Experiment_ID"],
            row["Asset"],
            row["Strategy"],
            row["Evaluation_Period"],
            format_pct(strategy_return),
            format_pct(benchmark_return),
            format_pct(return_difference),
            format_num(row["Strategy_Sharpe"]),
            format_num(row["Buy_Hold_Sharpe"]),
            format_pct(row["Strategy_Max_Drawdown"]),
            format_num(p_value),
            row["Decision"],
        ]

        lines.append("| " + " | ".join(str(value) for value in table_values) + " |")
        

    lines.extend([
        "",
        "## Interpretation",
        "",
        "### Benchmark comparisons",
        "",
    ])

    for _, row in df.iterrows():
        strategy_return = row["Strategy_Total_Return"]
        benchmark_return = row["Buy_Hold_Total_Return"]

        lines.append(f"### {row['Experiment_ID']}")
        lines.append("")

        if pd.notna(strategy_return) and pd.notna(benchmark_return):
            difference = strategy_return - benchmark_return

            if difference > 0:
                lines.append(
                    f"The strategy outperformed buy-and-hold by "
                    f"{format_pct(difference)} in total return."
                )
            else:
                lines.append(
                    f"The strategy underperformed buy-and-hold by "
                    f"{format_pct(abs(difference))} in total return."
                )
        else:
            lines.append("The total-return comparison is unavailable.")

        lines.append("")
        lines.append(
            f"- Recorded decision: **{row['Decision']}**"
        )

        row_p_value = pd.to_numeric(
            pd.Series([row["HAC_P_Value"]]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(row_p_value):
            lines.append(
                f"- HAC p-value: {format_num(row_p_value)}"
            )
        else:
            lines.append(
                "- HAC significance: **Not tested or not recorded.**"
            )

        reason = row.get("Reason")
        if pd.notna(reason) and str(reason).strip():
            lines.append(f"- Recorded rationale: {reason}")

        lines.append("")

    lines.extend([
        "## Limitations",
        "",
        "- These results describe the tested historical periods only.",
        "- Strategies were evaluated over different date ranges; "
        "returns should not be compared without considering that difference.",
        "- Missing statistical tests are unknown, not evidence of significance.",
        "- Repeatedly testing strategies on the same data can introduce "
        "selection bias and overfitting.",
        "- Transaction costs, execution assumptions, slippage, and market "
        "impact may affect real-world results.",
        "- Past performance does not guarantee future results.",
        "",
    ])

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"Research report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
