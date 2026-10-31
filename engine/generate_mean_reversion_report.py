
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. Define the project files
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAINING_FILE = REPORTS_DIR / "aapl_mean_reversion_oos_training.csv"
COMPARISON_FILE = REPORTS_DIR / "aapl_mean_reversion_oos_comparison.csv"

OUTPUT_FILE = REPORTS_DIR / "mean_reversion_research_report_generated.md"


# --------------------------------------------------
# 2. Check that the required experiment results exist
# --------------------------------------------------

for file_path in [TRAINING_FILE, COMPARISON_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required results file not found: {file_path}"
        )


# --------------------------------------------------
# 3. Load the saved results
# --------------------------------------------------

training = pd.read_csv(TRAINING_FILE)

comparison = pd.read_csv(
    COMPARISON_FILE,
    index_col=0,
)


# --------------------------------------------------
# 4. Find the configuration selected during training
# --------------------------------------------------

sharpe_column = "Sharpe Ratio (rf=0)"

best_row = training.loc[
    training[sharpe_column].idxmax()
]

best_window = int(best_row["Window"])
best_threshold = float(best_row["Threshold"])
training_sharpe = float(best_row[sharpe_column])


# --------------------------------------------------
# 5. Helper functions for readable report values
# --------------------------------------------------

def percentage(value):
    return f"{float(value):.2%}"


def decimal(value):
    return f"{float(value):.4f}"


def metric(metric_name, strategy_name):
    return float(comparison.loc[metric_name, strategy_name])


# --------------------------------------------------
# 6. Build a table of all training experiments
# --------------------------------------------------

training_table = [
    "| Window | Entry threshold | Total return | Sharpe ratio |",
    "|---:|---:|---:|---:|",
]

for _, row in training.iterrows():
    training_table.append(
        f"| {int(row['Window'])} "
        f"| {float(row['Threshold']):.1%} "
        f"| {percentage(row['Total Return'])} "
        f"| {decimal(row[sharpe_column])} |"
    )

training_table_text = "\n".join(training_table)


# --------------------------------------------------
# 7. Extract out-of-sample comparison metrics
# --------------------------------------------------

strategy = "Mean Reversion"
benchmark = "Buy and Hold"

strategy_return = metric("Total Return", strategy)
benchmark_return = metric("Total Return", benchmark)

strategy_cagr = metric("CAGR", strategy)
benchmark_cagr = metric("CAGR", benchmark)

strategy_volatility = metric("Annualized Volatility", strategy)
benchmark_volatility = metric("Annualized Volatility", benchmark)

strategy_sharpe = metric(sharpe_column, strategy)
benchmark_sharpe = metric(sharpe_column, benchmark)

strategy_drawdown = metric("Maximum Drawdown", strategy)
benchmark_drawdown = metric("Maximum Drawdown", benchmark)


# --------------------------------------------------
# 8. Write the research report
# --------------------------------------------------

report = f"""# AI Quant Research Lab
## Mean-Reversion Strategy: Research Report

**Asset:** Apple Inc. (AAPL)  
**Price data:** Yahoo Finance adjusted historical prices  
**Research period:** January 2021 – December 2023  
**Out-of-sample test period:** January 2024 – December 2025  
**Transaction cost assumption:** 0.1% per unit of turnover

---

## 1. Research hypothesis

The strategy tests whether a stock price that falls sufficiently below
its recent rolling average tends to recover.

A position is opened when:

- The closing price is below its rolling average by more than the
  selected negative threshold.
- The strategy holds the position until the entry condition is no
  longer satisfied.

This is a simple hypothesis to test, not evidence that mean reversion
actually exists.

## 2. Experimental method

Seven combinations of rolling-average windows and entry thresholds
were evaluated using the research period only.

The selected configuration was the one with the highest research-period
Sharpe ratio.

**Selected window:** {best_window} trading sessions  
**Selected entry threshold:** {best_threshold:.1%}  
**Research-period Sharpe ratio:** {training_sharpe:.4f}

### Research-period parameter results

{training_table_text}

These results describe the training period. They are not out-of-sample
evidence of future performance.

## 3. Out-of-sample evaluation

The selected parameters were applied to the separate 2024–2025 test
period. The test period was not used to select the configuration.

### Performance comparison

| Metric | Mean reversion | Buy and hold |
|---|---:|---:|
| Total return | {percentage(strategy_return)} | {percentage(benchmark_return)} |
| CAGR | {percentage(strategy_cagr)} | {percentage(benchmark_cagr)} |
| Annualized volatility | {percentage(strategy_volatility)} | {percentage(benchmark_volatility)} |
| Sharpe ratio (risk-free rate = 0) | {decimal(strategy_sharpe)} | {decimal(benchmark_sharpe)} |
| Maximum drawdown | {percentage(strategy_drawdown)} | {percentage(benchmark_drawdown)} |

## 4. Interpretation

The mean-reversion strategy returned {percentage(strategy_return)}
during the out-of-sample period, compared with
{percentage(benchmark_return)} for buy and hold.

Its Sharpe ratio was {decimal(strategy_sharpe)}, compared with
{decimal(benchmark_sharpe)} for buy and hold.

The strategy's maximum drawdown was {percentage(strategy_drawdown)},
while the benchmark's was {percentage(benchmark_drawdown)}.

These results suggest that the strategy did not outperform buy and
hold on total return or Sharpe ratio during this test period. Its
drawdown was smaller, but lower drawdown alone does not establish
that the strategy is better.

## 5. Limitations

1. The experiment uses only one stock.
2. A small set of parameter combinations cannot establish a general
   market effect.
3. The chosen parameters may be sensitive to the selected research period.
4. The assumed transaction cost may not capture all real trading costs,
   including slippage and market impact.
5. Historical performance does not guarantee future performance.
6. The strategy is long-or-cash and does not model every practical
   portfolio constraint.
7. Results depend on the backtesting implementation and its assumptions.

## 6. Conclusion

This experiment demonstrates a repeatable research workflow:
define a hypothesis, test multiple configurations, select parameters
using research data, and evaluate the selected configuration on
separate historical data.

The current results do not establish that this mean-reversion strategy
has a predictive advantage over buy and hold.

The appropriate next step is further testing across different assets
and market conditions, with realistic transaction costs and strict
protection against data leakage and repeated use of the test period.

---

*Generated automatically from the saved AI Quant Research Lab CSV results.*
"""

OUTPUT_FILE.write_text(report, encoding="utf-8")

print("Research report generated successfully.")
print(f"Selected parameters: window={best_window}, threshold={best_threshold:.1%}")
print(f"Output file: {OUTPUT_FILE}")
