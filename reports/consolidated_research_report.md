# AI Quant Research Lab

## Consolidated Research Report

This report is generated from the experiment registry.
Returns are historical results, not predictions of future performance.

## Experiment summary

- Experiments recorded: 5
- Rejected experiments: 5

## Results

| Experiment | Asset | Strategy | Evaluation period | Strategy return | Buy-and-hold return | Return difference | Strategy Sharpe | Buy-and-hold Sharpe | Max drawdown | HAC p-value | Decision |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| AAPL_MR_OOS_2024_2025 | AAPL | Mean Reversion | 2024-2025 | 8.72% | 42.54% | -33.81% | 0.3495 | 0.7742 | -18.96% | 0.3546 | REJECT |
| AAPL_SMA_BASELINE | AAPL | Moving Average | 2021-2025 | 33.15% | 115.81% | -82.66% | 0.4113 | 0.6932 | -29.80% | Unknown | REJECT |
| AAPL_MR_OPEN_OOS_2024_2025 | AAPL | Mean Reversion (Open Execution) | 2024-2025 | 11.50% | 47.28% | -35.78% | 0.4018 | 0.7998 | -13.44% | 0.3404 | REJECT |
| AAPL_STATIC_ML_OOS_2024_2025 | AAPL | Static Logistic Regression | 2024-2025 | 3.24% | 47.83% | -44.59% | 0.1838 | 0.8429 | -31.18% | Unknown | REJECT |
| AAPL_WALK_FORWARD_ML_OOS_2024_2025 | AAPL | Walk-Forward Logistic Regression | 2024-2025 | 13.00% | 47.83% | -34.82% | 0.3676 | 0.8429 | -31.73% | Unknown | REJECT |

## Interpretation

### Benchmark comparisons

### AAPL_MR_OOS_2024_2025

The strategy underperformed buy-and-hold by 33.81% in total return.

- Recorded decision: **REJECT**
- HAC p-value: 0.3546
- Recorded rationale: Underperformed buy-and-hold on total return and Sharpe; HAC test found no statistically significant mean daily excess return.

### AAPL_SMA_BASELINE

The strategy underperformed buy-and-hold by 82.66% in total return.

- Recorded decision: **REJECT**
- HAC significance: **Not tested or not recorded.**
- Recorded rationale: Total return was below buy-and-hold over 2021-2025. Transaction-cost assumptions and statistical evidence are not yet verified for this registry record.

### AAPL_MR_OPEN_OOS_2024_2025

The strategy underperformed buy-and-hold by 35.78% in total return.

- Recorded decision: **REJECT**
- HAC p-value: 0.3404
- Recorded rationale: Underperformed open-to-open buy-and-hold on total return and Sharpe ratio. HAC test found no statistically significant mean daily excess return at the 5% level. Lower volatility and drawdown do not establish a benchmark-beating edge.

### AAPL_STATIC_ML_OOS_2024_2025

The strategy underperformed buy-and-hold by 44.59% in total return.

- Recorded decision: **REJECT**
- HAC significance: **Not tested or not recorded.**
- Recorded rationale: Underperformed buy-and-hold on total return and Sharpe. ROC AUC was below 0.50 and Brier score was worse than the historical-rate baseline. No predictive edge was demonstrated.

### AAPL_WALK_FORWARD_ML_OOS_2024_2025

The strategy underperformed buy-and-hold by 34.82% in total return.

- Recorded decision: **REJECT**
- HAC significance: **Not tested or not recorded.**
- Recorded rationale: Underperformed buy-and-hold on total return and Sharpe. ROC AUC was below 0.50 and Brier score was worse than the historical-rate baseline. Expanding-window retraining did not establish a predictive edge.

## Limitations

- These results describe the tested historical periods only.
- Strategies were evaluated over different date ranges; returns should not be compared without considering that difference.
- Missing statistical tests are unknown, not evidence of significance.
- Repeatedly testing strategies on the same data can introduce selection bias and overfitting.
- Transaction costs, execution assumptions, slippage, and market impact may affect real-world results.
- Past performance does not guarantee future results.
