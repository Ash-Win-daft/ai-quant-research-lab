
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

OUTPUT_FILE = REPORTS / "comprehensive_research_report.md"


# ============================================================
# INPUT FILES
# ============================================================

INPUT_FILES = {
    "registry": "experiment_registry.csv",
    "registry_summary": "registry_summary.csv",
    "sma_summary": "aapl_moving_average_summary.csv",
    "sma_robustness": "aapl_parameter_robustness.csv",
    "old_sma_robustness": "robustness_results.csv",
    "old_sma_dependency": "strategy_dependency_analysis.csv",
    "close_mr": "aapl_mean_reversion_oos_comparison.csv",
    "close_mr_hac": "aapl_mean_reversion_hac_significance.csv",
    "close_mr_costs": "aapl_mean_reversion_transaction_cost_stress.csv",
    "open_mr": "aapl_mean_reversion_open_comparison.csv",
    "ml_metrics": "ml_prediction_metrics.csv",
    "ml_performance": "ml_performance.csv",
    "walk_forward_metrics": "walk_forward_metrics.csv",
    "walk_forward_performance": "walk_forward_performance.csv",
    "best_day_dependency": "best_day_dependency_comparison.csv",
    "extreme_days": "extreme_day_analysis.csv",
    "april_regime": "april_2025_regime_analysis.csv",
    "multi_stock_summary": "multi_stock_mean_reversion_open_summary.csv",
    "multi_stock_cross_section": "multi_stock_cross_sectional_statistics.csv",
    "multi_stock_sector": "multi_stock_sector_statistics.csv",
    "multi_stock_costs": "multi_stock_cost_sensitivity_summary.csv",
    "multi_stock_hac": "multi_stock_hac_significance.csv",
    "multi_stock_portfolio_hac": "multi_stock_portfolio_hac_significance.csv",
    "multi_stock_hac_lags": "multi_stock_hac_lag_sensitivity.csv",
    "multi_stock_validation": "multi_stock_validation_summary.csv"
}


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

def load_reports():
    """Load available report CSVs without modifying source files."""

    data = {}
    missing = []

    for name, filename in INPUT_FILES.items():
        path = REPORTS / filename

        if not path.exists():
            missing.append(filename)
            continue

        data[name] = pd.read_csv(path)

    if missing:
        print("WARNING: The following report files are missing:")
        for filename in missing:
            print(f"  - {filename}")

    return data


def validate_registry(data):
    """Validate the experiment registry."""

    required = {
        "Experiment_ID",
        "Asset",
        "Strategy",
        "Evaluation_Period",
        "Strategy_Total_Return",
        "Buy_Hold_Total_Return",
        "Decision",
    }

    if "registry" not in data:
        raise FileNotFoundError(
            "experiment_registry.csv is required."
        )

    missing = required - set(data["registry"].columns)

    if missing:
        raise ValueError(
            f"Experiment registry is missing columns: "
            f"{sorted(missing)}"
        )


# ============================================================
# FORMATTING HELPERS
# ============================================================

def clean_dataframe(df):
    """
    Remove exported numeric index columns.

    If an unnamed column contains meaningful text, rename it
    to Metric rather than deleting it.
    """

    if df is None or df.empty:
        return df

    df = df.copy()

    unnamed_columns = [
        column
        for column in df.columns
        if str(column).startswith("Unnamed:")
    ]

    for column in unnamed_columns:
        numeric_values = pd.to_numeric(
            df[column], errors="coerce"
        )

        if numeric_values.notna().all():
            df = df.drop(columns=[column])
        else:
            df = df.rename(columns={column: "Metric"})

    return df


def is_percentage_column(column):
    """Identify columns containing fractional percentage values."""

    name = str(column).lower()

    if name.endswith("_pct"):
        return True

    percentage_terms = (
        "return",
        "cagr",
        "volatility",
        "drawdown",
        "cost",
        "excess",
        "standard_error",
        "ci_95_lower",
        "ci_95_upper",
    )

    return any(term in name for term in percentage_terms)


def format_table_value(value, column, metric=None):
    """Format a value according to its meaning."""

    if pd.isna(value):
        return "Not available"

    if isinstance(value, bool):
        return str(value)

    if not isinstance(value, (int, float)):
        return str(value)

    name = str(column).lower()
    metric_name = str(metric).lower() if metric is not None else ""

    # Columns ending in _Pct already contain percentage units.
    # Example: 15.3289 means 15.3289%, not 1532.89%.
    if name.endswith("_pct"):
        return f"{value:.2f}%"

    # Identify percentage-based columns.
    if is_percentage_column(column):
        return f"{value:.2%}"

    # Performance tables sometimes use a generic first column
    # containing metric names and separate numeric value columns.
    if any(
        term in metric_name
        for term in (
            "return",
            "cagr",
            "volatility",
            "drawdown",
            "transaction cost",
            "excess return",
        )
    ):
        return f"{value:.2%}"

    # Sharpe ratios, p-values, accuracy and test statistics
    # remain decimal values.
    return f"{value:.4f}"


def comparison_table(df):
    """Create a Markdown table with readable numeric formatting."""

    if df is None or df.empty:
        return "Results unavailable.\n"

    table = clean_dataframe(df)

    if table is None or table.empty:
        return "Results unavailable.\n"

    formatted = table.copy()

    metric_column = (
        "Metric" if "Metric" in formatted.columns else None
    )

    for column in formatted.columns:
        if column == metric_column:
            continue

        formatted[column] = [
            format_table_value(
                value,
                column,
                metric=(
                    formatted.iloc[index][metric_column]
                    if metric_column
                    else None
                ),
            )
            for index, value in enumerate(formatted[column])
        ]

    return formatted.to_markdown(index=False)


def fmt_pct(value):
    if pd.isna(value):
        return "Not available"
    return f"{value:.2%}"


def fmt_num(value, decimals=3):
    if pd.isna(value):
        return "Not available"
    return f"{value:.{decimals}f}"


def registry_value(experiment_id, column):
    """Retrieve a value from a registered experiment."""

    registry_path = REPORTS / INPUT_FILES["registry"]

    if not registry_path.exists():
        return None

    registry = pd.read_csv(registry_path)

    if not {
        "Experiment_ID",
        column,
    }.issubset(registry.columns):
        return None

    matches = registry.loc[
        registry["Experiment_ID"] == experiment_id,
        column,
    ]

    if matches.empty:
        return None

    return pd.to_numeric(matches.iloc[0], errors="coerce")


# ============================================================
# REPORT SECTIONS
# ============================================================


def build_executive_summary(data):
    registry = data["registry"]

    total = len(registry)
    rejected = int(
        (registry["Decision"] == "REJECT").sum()
    )

    multi_stock_count = 0
    if "multi_stock_summary" in data:
        multi_stock_count = len(data["multi_stock_summary"])

    return f"""
# AI Quant Research Lab
## Comprehensive Research Report

**Research assets:** AAPL and a selected universe of 10 US equities
**Research period:** 2021–2025, with primary out-of-sample evaluation during 2024–2025
**Report type:** Historical quantitative research and strategy evaluation

---

## 1. Executive Summary

This project investigates whether systematic trading strategies and
machine-learning models can identify repeatable opportunities in
historical equity-market data.

The research began with single-stock experiments on Apple Inc. (AAPL)
and subsequently expanded to a selected ten-stock US equity universe.

The multi-stock universe includes technology, consumer, financial,
energy and healthcare companies. The portfolio experiment uses
equal-weighted daily rebalancing.

The research emphasizes out-of-sample evaluation, transaction costs,
execution timing, statistical testing, parameter sensitivity and
explicit rejection of strategies that fail to demonstrate convincing
evidence.

The experiment registry currently contains **{total} experiments**,
of which **{rejected} are rejected**.

The multi-stock mean-reversion experiment evaluated
**{multi_stock_count} stocks** over the 2024–2025 out-of-sample period.
None of the ten individual stocks outperformed their respective
benchmarks on total return.

The equal-weighted portfolio also underperformed its benchmark.
Its mean daily excess return was significantly negative under
HAC inference, with a p-value of approximately 0.0086.

The findings do not establish a reliable benchmark-beating trading edge.
Instead, the experiments demonstrate the importance of realistic
execution assumptions, statistical validation and disciplined
research methodology.
"""



def build_research_methodology():
    return """
## 2. Research Objective

The objective is to develop a research framework that attempts to
disprove its own trading hypotheses rather than simply identify
historical strategies with attractive returns.

The framework evaluates:

- Signal generation and execution timing.
- Historical performance against buy-and-hold.
- Transaction-cost sensitivity.
- Out-of-sample performance.
- Statistical significance.
- Parameter sensitivity.
- Dependence on unusually strong trading days.
- Predictive performance of machine-learning models.

### Research principle

A strategy is not accepted merely because it produces positive
historical returns. It must demonstrate credible performance after
costs, against an appropriate benchmark, and under out-of-sample
evaluation.
"""


def build_sma_section(data):
    text = """
## 3. Moving-Average Strategy

The moving-average strategy uses a short-term simple moving average
(SMA) and a long-term SMA.

A long position is held when the short-term average is above the
long-term average. The previous session's signal determines the
current session's position.

The historical backtest uses adjusted closing prices and assumes
a 0.10% transaction cost per position change.

### Earlier parameter sensitivity experiment

An earlier experiment evaluated four parameter combinations:
10/30, 20/50, 30/100 and 50/150.

All configurations use a common evaluation start determined by
the first valid 150-session moving average. The notebook's research
period is January 2021 to December 2025.

"""

    if "old_sma_robustness" in data:
        text += comparison_table(
            data["old_sma_robustness"]
        ) + "\n"

    text += """
The earlier parameter experiment is a historical sensitivity
analysis, not an independent out-of-sample test.

### Later six-configuration analysis

A separate analysis evaluated six moving-average configurations:

"""

    if "sma_robustness" in data:
        text += comparison_table(
            data["sma_robustness"]
        ) + "\n"

    text += """
The two parameter experiments should be interpreted separately.
Their results should not be combined because their evaluation
procedures and source data may differ.

### Interpretation

The moving-average experiments did not establish consistent
outperformance over buy-and-hold. Parameter sensitivity also
demonstrates that historical performance depends on the selected
moving-average windows.

The six-configuration analysis identified a stronger historical
configuration, but it still did not outperform buy-and-hold
on total return or Sharpe ratio.
"""

    return text


def build_mean_reversion_section(data):
    text = """
## 4. Mean-Reversion Strategy — Close Execution

The close-execution mean-reversion experiment evaluates whether
prices sufficiently below a rolling average subsequently recover.

Parameters were selected using the 2021–2023 training period and
evaluated on 2024–2025 data.

The selected configuration was a 20-session window with a
-5% entry threshold.

### Out-of-sample performance

"""

    if "close_mr" in data:
        text += comparison_table(
            data["close_mr"]
        ) + "\n"

    if "close_mr_hac" in data:
        text += "\n### HAC significance test\n\n"
        text += comparison_table(
            data["close_mr_hac"]
        ) + "\n"

    if "close_mr_costs" in data:
        text += "\n### Transaction-cost sensitivity\n\n"
        text += comparison_table(
            data["close_mr_costs"]
        ) + "\n"

    text += """
The HAC p-value was approximately 0.3546. The experiment therefore
did not demonstrate statistically significant mean daily excess
returns at the 5% level.

The strategy also underperformed buy-and-hold on total return
and Sharpe ratio.

**Decision: REJECT.**
"""

    return text


def build_open_execution_section(data):
    text = """
## 5. Mean-Reversion Strategy — Open Execution

A separate implementation evaluates next-session execution using
opening prices.

A signal observed at the previous close determines the position
held during the next open-to-open return interval.

Parameters were selected using 2021–2023 training data and evaluated
out of sample over 2024–2025.

The selected configuration was a 20-session window with a
-5% entry threshold.

### Out-of-sample performance

"""

    if "open_mr" in data:
        text += comparison_table(
            data["open_mr"]
        ) + "\n"

    open_hac_p = registry_value(
        "AAPL_MR_OPEN_OOS_2024_2025",
        "HAC_P_Value",
    )

    text += """
The open-execution experiment produced a total return of approximately
11.50%, compared with 47.28% for its open-to-open buy-and-hold
benchmark.

"""

    if open_hac_p is not None and pd.notna(open_hac_p):
        text += (
            f"Its HAC p-value, as recorded in the experiment registry, "
            f"was approximately {open_hac_p:.4f}. This does not provide "
            "statistically significant evidence of positive mean daily "
            "excess returns at the 5% level.\n\n"
        )
    else:
        text += """
A verified HAC p-value was not available in the experiment registry
for this experiment. Statistical significance is therefore not
asserted here.

"""

    text += """
Lower volatility and drawdown do not compensate for the substantial
underperformance against the benchmark.

**Decision: REJECT.**
"""

    return text


def build_ml_section(data):
    text = """
## 6. Static Machine-Learning Experiment

A logistic-regression classifier was evaluated using historical
price-derived features.

The experiment investigated whether the model could predict the
direction of subsequent returns.

### Prediction metrics

"""

    if "ml_metrics" in data:
        text += comparison_table(
            data["ml_metrics"]
        ) + "\n"

    text += "\n### Strategy performance\n\n"

    if "ml_performance" in data:
        text += comparison_table(
            data["ml_performance"]
        ) + "\n"

    text += """
The model's ROC AUC was approximately 0.4661, below the
0.50 level associated with random ranking.

The resulting strategy also underperformed buy-and-hold.

These results do not support the use of this model as a profitable
trading signal in its tested configuration.
"""

    return text


def build_walk_forward_section(data):
    text = """
## 7. Walk-Forward Validation

Walk-forward validation evaluates a model sequentially using
historical information available before each prediction.

This is a more realistic evaluation design than fitting a model
once and evaluating it on a fixed holdout period.

### Prediction metrics

"""

    if "walk_forward_metrics" in data:
        text += comparison_table(
            data["walk_forward_metrics"]
        ) + "\n"

    text += "\n### Strategy performance\n\n"

    if "walk_forward_performance" in data:
        text += comparison_table(
            data["walk_forward_performance"]
        ) + "\n"

    text += """
The walk-forward model did not establish predictive performance
above its simple baselines. Its trading strategy also underperformed
buy-and-hold.

Walk-forward evaluation improves the testing design, but does not
by itself establish that a model is free from data leakage or
implementation bias. Those properties depend on the complete
feature-generation and execution pipeline.
"""

    return text


# ============================================================
# DIAGNOSTICS
# ============================================================

def build_extreme_day_table(data):
    """Show the five strongest and five weakest recorded days."""

    if "extreme_days" not in data:
        return "Extreme-day results unavailable.\n"

    df = clean_dataframe(data["extreme_days"])

    if df is None or df.empty:
        return "Extreme-day results unavailable.\n"

    # The source CSV is already ordered into best-day and
    # worst-day groups.
    selected = pd.concat(
        [df.head(5), df.tail(5)]
    ).drop_duplicates()

    return comparison_table(selected)


def build_april_regime_table(data):
    """
    Show the eight sessions with the largest absolute
    underlying asset moves during April 2025.
    """

    if "april_regime" not in data:
        return "April 2025 regime results unavailable.\n"

    df = data["april_regime"].copy()

    if df.empty or "Asset_Return_Pct" not in df.columns:
        return comparison_table(df)

    df["Asset_Return_Pct"] = pd.to_numeric(
        df["Asset_Return_Pct"], errors="coerce"
    )

    selected = (
        df.assign(
            Absolute_Move=df["Asset_Return_Pct"].abs()
        )
        .nlargest(8, "Absolute_Move")
        .sort_values("Date")
        .drop(columns=["Absolute_Move"])
    )

    preferred_columns = [
        "Date",
        "Asset_Return_Pct",
        "Signal",
        "Position",
        "Transaction_Cost_Pct",
        "Net_Return_Pct",
        "Strategy_Equity",
        "Buy_Hold_Equity",
    ]

    available = [
        column
        for column in preferred_columns
        if column in selected.columns
    ]

    if available:
        selected = selected[available]

    return comparison_table(selected)


def build_multi_stock_section(data):
    text = """
### Multi-Stock Mean-Reversion Research

The research universe was expanded from AAPL to ten selected US
equities:

AAPL, MSFT, NVDA, GOOGL, AMZN, WMT, JPM, GS, XOM and JNJ.

All ten datasets passed market-data validation and had aligned
trading dates. The common dataset contained 1,261 observations
per stock, covering 2021-01-04 through 2026-01-09.

The mean-reversion strategy used the 20-session rolling window
and -5% entry threshold selected using AAPL training data
from 2021–2023. These parameters were held fixed across the
ten-stock evaluation. The out-of-sample period was 2024–2025.

The use of AAPL-selected parameters across other stocks is a
deliberate fixed-parameter transfer experiment. It is not
independent parameter optimization for each stock.

#### Individual-stock performance

"""

    if "multi_stock_summary" in data:
        text += comparison_table(
            data["multi_stock_summary"]
        ) + "\n"
    else:
        text += "Multi-stock performance summary unavailable.\n"

    text += """
All ten stocks underperformed their respective benchmarks
on total return at the reference transaction cost of 0.10%.

#### Cross-sectional results

"""

    if "multi_stock_cross_section" in data:
        text += comparison_table(
            data["multi_stock_cross_section"]
        ) + "\n"
    else:
        text += "Cross-sectional statistics unavailable.\n"

    text += """
#### Sector-level descriptive results

"""

    if "multi_stock_sector" in data:
        text += comparison_table(
            data["multi_stock_sector"]
        ) + "\n"
    else:
        text += "Sector statistics unavailable.\n"

    text += """
Sector groups are small and unevenly represented. These
descriptive results should not be interpreted as reliable
sector-wide conclusions.

#### Transaction-cost sensitivity

The experiment was reconstructed at several transaction-cost
assumptions. The 0.10% reconstruction was checked against the
original strategy results.

"""

    if "multi_stock_costs" in data:
        text += comparison_table(
            data["multi_stock_costs"]
        ) + "\n"
    else:
        text += "Multi-stock cost-sensitivity results unavailable.\n"

    text += """
Even with zero transaction costs, none of the ten stocks
outperformed its benchmark. Transaction costs worsened the
results but did not explain the underlying underperformance.

#### Individual-stock HAC significance tests

The individual-stock tests evaluate whether each stock's
mean daily strategy excess return differs from zero.

Holm correction is applied across the ten individual tests
to control the family-wise error rate.

"""

    if "multi_stock_hac" in data:
        text += comparison_table(
            data["multi_stock_hac"]
        ) + "\n"
    else:
        text += "Individual-stock HAC results unavailable.\n"

    text += """
Four stocks had unadjusted p-values below 0.05. After Holm
correction, only WMT remained significant at the 5% level.
Its significant result represents negative mean daily excess
return, not evidence of a profitable strategy.

#### Equal-weighted portfolio HAC test

The portfolio is constructed by averaging the ten constituent
strategy returns each day. The benchmark is constructed by
averaging the ten constituent asset returns each day. Both
are therefore equal-weighted and daily rebalanced.

"""

    if "multi_stock_portfolio_hac" in data:
        text += comparison_table(
            data["multi_stock_portfolio_hac"]
        ) + "\n"
    else:
        text += "Portfolio HAC results unavailable.\n"

    text += """
The portfolio contained 501 daily observations. Its mean
daily excess return was approximately -0.1081%, with a HAC
p-value of 0.008642. The 95% confidence interval was entirely
negative.

#### HAC lag sensitivity

"""

    if "multi_stock_hac_lags" in data:
        text += comparison_table(
            data["multi_stock_hac_lags"]
        ) + "\n"
    else:
        text += "HAC lag-sensitivity results unavailable.\n"

    text += """
The negative portfolio excess-return result remained
statistically significant at the 5% level across HAC
lag specifications of 1, 5, 10 and 20.

#### Equal-weighted portfolio performance

The equal-weighted portfolio's compounded results were:

| Metric | Strategy | Equal-weighted benchmark |
|---|---:|---:|
| Total return | 17.69% | 97.34% |
| Sharpe ratio | 0.8649 | 1.9342 |
| Maximum drawdown | -5.39% | -22.45% |

The strategy experienced a smaller maximum drawdown, but
substantially underperformed the benchmark on total return
and Sharpe ratio.

**Decision: REJECT.**

#### Interpretation

The multi-stock experiment provides broader evidence than
the original AAPL-only research, but it does not establish
a profitable or benchmark-beating mean-reversion strategy.

The ten-stock universe is selected rather than a point-in-time
historical universe. The findings are conditional on the
selected assets, fixed AAPL-derived parameters, transaction-cost
assumptions and 2024–2025 evaluation period.
"""

    return text


def build_diagnostics_section(data):
    text = """
## 8. Strategy Diagnostics and Multi-Stock Research

### Earlier moving-average best-day dependency

The earlier SMA experiment replaced its five strongest daily strategy
returns with zero and recalculated performance.

This is an observation-deletion sensitivity test, not an executable
alternative strategy.

"""

    if "old_sma_dependency" in data:
        text += comparison_table(
            data["old_sma_dependency"]
        ) + "\n"

    text += """
### Mean-reversion best-day dependency

A separate diagnostic examined the effect of removing the strongest
daily returns from the close-execution mean-reversion strategy and
its benchmark.

"""

    if "best_day_dependency" in data:
        text += comparison_table(
            data["best_day_dependency"]
        ) + "\n"

    text += """
The two dependency analyses use different procedures and should
not be directly compared.

### Extreme-day analysis

The following table summarizes five of the strongest and five
of the weakest recorded daily observations.

"""

    text += build_extreme_day_table(data)

    text += """

### April 2025 market-regime observations

The following table selects the eight April 2025 sessions with
the largest absolute underlying asset moves. It includes the
strategy's signal, position, transaction cost and net return
where those fields are available.

These are descriptive observations. They do not establish causal
explanations for strategy performance.

"""

    text += build_april_regime_table(data)
    text += "\n" + build_multi_stock_section(data)
    return text


# ============================================================
# EXPERIMENT REGISTRY
# ============================================================

def build_registry_section(data):
    registry = data["registry"]

    text = """
## 9. Experiment Registry

The registry records formal strategy experiments and their decisions.

"""

    columns = [
        "Experiment_ID",
        "Strategy",
        "Evaluation_Period",
        "Strategy_Total_Return",
        "Buy_Hold_Total_Return",
        "Strategy_Sharpe",
        "Buy_Hold_Sharpe",
        "HAC_P_Value",
        "Decision",
    ]

    available = [
        column
        for column in columns
        if column in registry.columns
    ]

    table = registry[available].copy()

    text += comparison_table(table) + "\n"

    total = len(registry)
    rejected = int(
        (registry["Decision"] == "REJECT").sum()
    )


    text += f"""
The registry currently contains {total} experiments,
of which {rejected} are rejected.

The registry includes five AAPL experiments and one
ten-stock equal-weighted portfolio experiment.

The moving-average registry entry does not have a verified
HAC significance result. Its unavailable p-value is not
interpreted as evidence of statistical significance.

The close-execution and open-execution AAPL mean-reversion
experiments did not demonstrate statistically significant
mean daily excess returns at the 5% level.

The ten-stock portfolio experiment recorded a HAC p-value
of approximately 0.0086, indicating significantly negative
mean daily excess returns at the 5% level. This result
remained significant across the tested HAC lag specifications.

The portfolio's statistical result is conditional on the
selected ten-stock universe and 2024–2025 evaluation period.
"""

    return text

# ============================================================
# LIMITATIONS, FUTURE RESEARCH AND CONCLUSION
# ============================================================


def build_limitations_section():
    return """
## 10. Limitations

The research has several important limitations:

1. The initial experiments use AAPL. The later multi-stock
   experiment expands the universe to ten selected US equities,
   but this is not a broad-market sample.

2. The ten-stock universe is selected using a present-day
   configuration rather than a point-in-time historical
   constituent universe. Survivorship and selection bias
   may therefore affect the findings.

3. The universe is not sector-balanced. Technology accounts
   for four of the ten stocks, while several other sectors
   have only one or two representatives. Sector-level
   statistics are descriptive, not generalizable.

4. Multi-stock mean-reversion parameters were selected using
   AAPL training data and transferred unchanged to the other
   stocks. This does not establish that the parameters are
   optimal for each constituent.

5. The portfolio benchmark is an equal-weighted, daily
   rebalanced basket. It should not be interpreted as a
   conventional buy-and-hold portfolio with fixed share
   quantities.

6. Transaction-cost assumptions simplify real-world execution.
   Market impact, bid-ask spreads, liquidity constraints and
   partial fills are not fully modeled.

7. Some experiments use different execution conventions and
   evaluation periods, so their returns are not directly
   comparable.

8. Parameter sensitivity and repeated experimentation do not
   eliminate data-snooping risk or multiple-testing concerns.

9. Machine-learning features are primarily derived from
   historical price data. Further independent validation
   is required.

10. The HAC tests evaluate mean daily arithmetic excess returns.
    They do not directly test whether compounded total returns
    or CAGR are statistically different.

11. The annualized mean excess return reported in the HAC
    analysis is the arithmetic daily mean multiplied by 252.
    It is not a compounded annual return.

12. The framework is a research prototype, not a production
    trading system.

## 11. Future Research

Potential extensions include:

- Expanding to a broader, point-in-time US equity universe.
- Controlling survivorship and selection bias.
- Testing across multiple assets and market regimes.
- Evaluating parameters independently using nested or
  properly separated training and validation procedures.
- Adding point-in-time fundamental and alternative data.
- Modeling spreads, slippage and market impact.
- Applying purged and embargoed cross-validation where appropriate.
- Evaluating multiple-testing and data-snooping adjustments.
- Testing portfolio-level risk and exposure constraints.
- Improving feature engineering and model calibration.
- Comparing against stronger statistical and economic baselines.
- Requiring independent validation before accepting a strategy.

## 12. Final Conclusion

The experiments did not identify a strategy that convincingly
outperformed its benchmark after the tested transaction costs
and evaluation procedures.

The single-stock and multi-stock mean-reversion experiments
underperformed their respective benchmarks. The equal-weighted
ten-stock portfolio also produced significantly negative mean
daily excess returns under the tested HAC specifications.

The machine-learning experiments failed to demonstrate
reliable predictive superiority over simple baselines.

The main contribution of this project is its research framework:
strategies are tested under explicit assumptions, evaluated
against benchmarks, examined for statistical support and
rejected when the evidence is insufficient.

The results should be interpreted as historical research findings,
not investment advice or evidence of a deployable trading edge.
"""



# ============================================================
# MAIN
# ============================================================

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("Loading research reports...")
    data = load_reports()

    validate_registry(data)

    sections = [
        build_executive_summary(data),
        build_research_methodology(),
        build_sma_section(data),
        build_mean_reversion_section(data),
        build_open_execution_section(data),
        build_ml_section(data),
        build_walk_forward_section(data),
        build_diagnostics_section(data),
        build_registry_section(data),
        build_limitations_section(),
    ]

    report = "\n\n".join(sections)

    OUTPUT_FILE.write_text(report, encoding="utf-8")

    numbered_sections = sum(
        1
        for line in report.splitlines()
        if line.startswith("## ")
        and line[3:4].isdigit()
    )

    print("\nCOMPREHENSIVE REPORT GENERATED")
    print("=" * 60)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Characters: {len(report):,}")
    print(f"Numbered sections: {numbered_sections}")

    if numbered_sections != 12:
        raise RuntimeError(
            f"Expected 12 numbered sections, "
            f"but generated {numbered_sections}."
        )


if __name__ == "__main__":
    main()
