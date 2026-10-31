
# AI Quant Research Lab
## Comprehensive Research Report

**Research asset:** Apple Inc. (AAPL)  
**Research period:** 2021–2025  
**Report type:** Historical quantitative research and strategy evaluation

---

## 1. Executive Summary

This project investigates whether systematic trading strategies and
machine-learning models can identify repeatable opportunities in
historical equity-market data.

The research emphasizes out-of-sample evaluation, transaction costs,
statistical testing, parameter sensitivity and explicit rejection
of strategies that fail to demonstrate convincing evidence.

The experiment registry currently contains **3 experiments**,
of which **3 are rejected**.

The findings do not establish a reliable benchmark-beating trading edge.
Instead, the experiments demonstrate the importance of realistic
execution assumptions, statistical validation and disciplined
research methodology.



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

|   Short MA |   Long MA | Strategy Return   | Strategy CAGR   |   Strategy Sharpe | Strategy Max Drawdown   | Benchmark Return   |   Number of Position Changes |
|-----------:|----------:|:------------------|:----------------|------------------:|:------------------------|:-------------------|-----------------------------:|
|         10 |        30 | 22.30%            | 4.69%           |            0.3408 | -27.95%                 | 89.30%             |                           41 |
|         20 |        50 | 29.88%            | 6.14%           |            0.4157 | -29.80%                 | 89.30%             |                           26 |
|         30 |       100 | 33.24%            | 6.76%           |            0.4372 | -31.41%                 | 89.30%             |                           12 |
|         50 |       150 | 11.79%            | 2.57%           |            0.2275 | -28.40%                 | 89.30%             |                           11 |

The earlier parameter experiment is a historical sensitivity
analysis, not an independent out-of-sample test.

### Later six-configuration analysis

A separate analysis evaluated six moving-average configurations:

|   Short_Window |   Long_Window | Total Return   | CAGR   | Annualized Volatility   |   Sharpe Ratio (rf=0) | Maximum Drawdown   | Buy_Hold_Total_Return   |   Buy_Hold_Sharpe | Positive_Return   | Beats_Buy_Hold_Return   | Beats_Buy_Hold_Sharpe   |
|---------------:|--------------:|:---------------|:-------|:------------------------|----------------------:|:-------------------|:------------------------|------------------:|:------------------|:------------------------|:------------------------|
|             10 |            30 | 28.77%         | 5.21%  | 17.94%                  |                0.3723 | -27.95%            | 115.81%                 |            0.6932 | True              | False                   | False                   |
|             15 |            40 | 38.51%         | 6.76%  | 17.28%                  |                0.4648 | -32.34%            | 115.81%                 |            0.6932 | True              | False                   | False                   |
|             20 |            50 | 33.15%         | 5.92%  | 17.83%                  |                0.4113 | -29.80%            | 115.81%                 |            0.6932 | True              | False                   | False                   |
|             25 |            60 | 67.38%         | 10.90% | 17.77%                  |                0.6709 | -24.22%            | 115.81%                 |            0.6932 | True              | False                   | False                   |
|             30 |           100 | 44.69%         | 7.70%  | 18.23%                  |                0.4979 | -31.41%            | 115.81%                 |            0.6932 | True              | False                   | False                   |
|             50 |           150 | 11.79%         | 2.26%  | 18.52%                  |                0.2136 | -28.40%            | 115.81%                 |            0.6932 | True              | False                   | False                   |

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



## 4. Mean-Reversion Strategy — Close Execution

The close-execution mean-reversion experiment evaluates whether
prices sufficiently below a rolling average subsequently recover.

Parameters were selected using the 2021–2023 training period and
evaluated on 2024–2025 data.

The selected configuration was a 20-session window with a
-5% entry threshold.

### Out-of-sample performance

| Metric                | Mean Reversion   | Buy and Hold   |
|:----------------------|:-----------------|:---------------|
| Total Return          | 8.72%            | 42.54%         |
| CAGR                  | 4.29%            | 19.47%         |
| Annualized Volatility | 15.18%           | 27.99%         |
| Sharpe Ratio (rf=0)   | 0.3495           | 0.7742         |
| Maximum Drawdown      | -18.96%          | -33.36%        |

### HAC significance test

|   Observations | Mean_Daily_Excess_Return   |   Ordinary_T_Statistic |   Ordinary_P_Value |   HAC_Lags | HAC_Standard_Error   |   HAC_T_Statistic |   HAC_P_Value | HAC_CI_95_Lower   | HAC_CI_95_Upper   |
|---------------:|:---------------------------|-----------------------:|-------------------:|-----------:|:---------------------|------------------:|--------------:|:------------------|:------------------|
|            502 | -0.06%                     |                -0.9808 |             0.3272 |          5 | 0.07%                |           -0.9266 |        0.3546 | -0.20%            | 0.07%             |

### Transaction-cost sensitivity

| Transaction_Cost_Per_Turnover   | Strategy_Total_Return   | Strategy_CAGR   |   Strategy_Sharpe | Strategy_Max_Drawdown   | Buy_Hold_Total_Return   |   Buy_Hold_Sharpe |
|:--------------------------------|:------------------------|:----------------|------------------:|:------------------------|:------------------------|------------------:|
| 0.00%                           | 10.93%                  | 5.34%           |            0.4162 | -18.62%                 | 42.54%                  |            0.7742 |
| 0.05%                           | 9.82%                   | 4.82%           |            0.3828 | -18.79%                 | 42.54%                  |            0.7742 |
| 0.10%                           | 8.72%                   | 4.29%           |            0.3495 | -18.96%                 | 42.54%                  |            0.7742 |
| 0.25%                           | 5.50%                   | 2.72%           |            0.2496 | -19.45%                 | 42.54%                  |            0.7742 |
| 0.50%                           | 0.32%                   | 0.16%           |            0.0841 | -20.48%                 | 42.54%                  |            0.7742 |

The HAC p-value was approximately 0.3546. The experiment therefore
did not demonstrate statistically significant mean daily excess
returns at the 5% level.

The strategy also underperformed buy-and-hold on total return
and Sharpe ratio.

**Decision: REJECT.**



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

| Metric                | Mean Reversion (Open Execution)   | Buy and Hold (Open-to-Open)   |
|:----------------------|:----------------------------------|:------------------------------|
| Total Return          | 11.50%                            | 47.28%                        |
| CAGR                  | 5.63%                             | 21.50%                        |
| Annualized Volatility | 17.21%                            | 29.90%                        |
| Sharpe Ratio (rf=0)   | 0.4018                            | 0.7998                        |
| Maximum Drawdown      | -13.44%                           | -33.33%                       |

The open-execution experiment produced a total return of approximately
11.50%, compared with 47.28% for its open-to-open buy-and-hold
benchmark.

Its HAC p-value, as recorded in the experiment registry, was approximately 0.3404. This does not provide statistically significant evidence of positive mean daily excess returns at the 5% level.


Lower volatility and drawdown do not compensate for the substantial
underperformance against the benchmark.

**Decision: REJECT.**



## 6. Static Machine-Learning Experiment

A logistic-regression classifier was evaluated using historical
price-derived features.

The experiment investigated whether the model could predict the
direction of subsequent returns.

### Prediction metrics

| Model                   |   Accuracy |   Balanced Accuracy |   Precision |   Recall | ROC AUC       |
|:------------------------|-----------:|--------------------:|------------:|---------:|:--------------|
| Logistic Regression     |     0.4731 |              0.4583 |      0.5136 |   0.6227 | 0.4661        |
| Training Majority Class |     0.5449 |              0.5    |      0.5449 |   1      | Not available |
| Always Predict Up       |     0.5449 |              0.5    |      0.5449 |   1      | Not available |

### Strategy performance

| Metric                      | ML Strategy   | Buy and Hold   |
|:----------------------------|:--------------|:---------------|
| Total Return                | -0.82%        | 47.83%         |
| CAGR                        | -0.41%        | 21.73%         |
| Annualized Volatility       | 23.64%        | 27.89%         |
| Sharpe Ratio (0% risk-free) | 0.0990        | 0.8429         |
| Maximum Drawdown            | -31.18%       | -33.36%        |

The model's ROC AUC was approximately 0.4661, below the
0.50 level associated with random ranking.

The resulting strategy also underperformed buy-and-hold.

These results do not support the use of this model as a profitable
trading signal in its tested configuration.



## 7. Walk-Forward Validation

Walk-forward validation evaluates a model sequentially using
historical information available before each prediction.

This is a more realistic evaluation design than fitting a model
once and evaluating it on a fixed holdout period.

### Prediction metrics

| Metric                               |   Value |
|:-------------------------------------|--------:|
| Accuracy                             |  0.513  |
| Balanced Accuracy                    |  0.4848 |
| ROC AUC                              |  0.4469 |
| Brier Score                          |  0.2542 |
| Majority-Class Baseline Accuracy     |  0.5449 |
| Historical-Rate Baseline Brier Score |  0.2485 |

### Strategy performance

| Metric                | Walk-Forward Strategy   | Buy and Hold   |
|:----------------------|:------------------------|:---------------|
| Total Return          | 13.00%                  | 47.83%         |
| CAGR                  | 6.34%                   | 21.73%         |
| Annualized Volatility | 25.38%                  | 27.89%         |
| Sharpe Ratio          | 0.3676                  | 0.8429         |
| Max Drawdown          | -31.73%                 | -33.36%        |

The walk-forward model did not establish predictive performance
above its simple baselines. Its trading strategy also underperformed
buy-and-hold.

Walk-forward evaluation improves the testing design, but does not
by itself establish that a model is free from data leakage or
implementation bias. Those properties depend on the complete
feature-generation and execution pipeline.



## 8. Strategy Dependency and Market-Regime Diagnostics

### Earlier moving-average best-day dependency

The earlier SMA experiment replaced its five strongest daily strategy
returns with zero and recalculated performance.

This is an observation-deletion sensitivity test, not an executable
alternative strategy.

| Metric                      | Original Strategy   | Without Best 5 Days   |
|:----------------------------|:--------------------|:----------------------|
| Total Return                | 26.54%              | -0.68%                |
| CAGR                        | 12.54%              | -0.34%                |
| Annualized Volatility       | 18.52%              | 16.67%                |
| Sharpe Ratio (0% risk-free) | 0.7304              | 0.0631                |
| Maximum Drawdown            | -25.03%             | -25.03%               |

### Mean-reversion best-day dependency

A separate diagnostic examined the effect of removing the strongest
daily returns from the close-execution mean-reversion strategy and
its benchmark.

| Series                  | Test               |   Days_Removed | Total_Return   | Return_Change   |   Observations_Remaining |
|:------------------------|:-------------------|---------------:|:---------------|:----------------|-------------------------:|
| Mean Reversion Strategy | Baseline           |              0 | 8.72%          | 0.00%           |                      502 |
| Mean Reversion Strategy | Remove top 1 days  |              1 | -5.73%         | -14.45%         |                      501 |
| Mean Reversion Strategy | Remove top 3 days  |              3 | -12.30%        | -21.03%         |                      499 |
| Mean Reversion Strategy | Remove top 5 days  |              5 | -17.02%        | -25.75%         |                      497 |
| Mean Reversion Strategy | Remove top 10 days |             10 | -25.10%        | -33.83%         |                      492 |
| Buy and Hold            | Baseline           |              0 | 42.54%         | 0.00%           |                      502 |
| Buy and Hold            | Remove top 1 days  |              1 | 23.59%         | -18.95%         |                      501 |
| Buy and Hold            | Remove top 3 days  |              3 | 8.38%          | -34.16%         |                      499 |
| Buy and Hold            | Remove top 5 days  |              5 | -2.69%         | -45.23%         |                      497 |
| Buy and Hold            | Remove top 10 days |             10 | -20.69%        | -63.23%         |                      492 |

The two dependency analyses use different procedures and should
not be directly compared.

### Extreme-day analysis

The following table summarizes five of the strongest and five
of the weakest recorded daily observations.

| Extreme_Group   | Date       |   Position | Asset_Return_Pct   | Gross_Return_Pct   | Transaction_Cost_Pct   | Net_Return_Pct   |
|:----------------|:-----------|-----------:|:-------------------|:-------------------|:-----------------------|:-----------------|
| Best 10         | 2025-04-09 |          1 | 15.33%             | 15.33%             | 0.00%                  | 15.33%           |
| Best 10         | 2025-04-11 |          1 | 4.06%              | 4.06%              | 0.00%                  | 4.06%            |
| Best 10         | 2025-04-22 |          1 | 3.41%              | 3.41%              | 0.10%                  | 3.31%            |
| Best 10         | 2025-01-27 |          1 | 3.18%              | 3.18%              | 0.00%                  | 3.18%            |
| Best 10         | 2025-05-27 |          1 | 2.53%              | 2.53%              | 0.10%                  | 2.43%            |
| Worst 10        | 2025-03-13 |          1 | -3.36%             | -3.36%             | 0.00%                  | -3.36%           |
| Worst 10        | 2025-04-07 |          1 | -3.67%             | -3.67%             | 0.00%                  | -3.67%           |
| Worst 10        | 2025-04-10 |          1 | -4.24%             | -4.24%             | 0.00%                  | -4.24%           |
| Worst 10        | 2025-04-08 |          1 | -4.98%             | -4.98%             | 0.00%                  | -4.98%           |
| Worst 10        | 2025-04-04 |          1 | -7.29%             | -7.29%             | 0.10%                  | -7.39%           |

### April 2025 market-regime observations

The following table selects the eight April 2025 sessions with
the largest absolute underlying asset moves. It includes the
strategy's signal, position, transaction cost and net return
where those fields are available.

These are descriptive observations. They do not establish causal
explanations for strategy performance.

| Date       | Asset_Return_Pct   |   Signal |   Position | Transaction_Cost_Pct   | Net_Return_Pct   |   Strategy_Equity |   Buy_Hold_Equity |
|:-----------|:-------------------|---------:|-----------:|:-----------------------|:-----------------|------------------:|------------------:|
| 2025-04-03 | -9.25%             |        1 |          0 | 0.00%                  | -0.00%           |            1.0229 |            1.0791 |
| 2025-04-04 | -7.29%             |        1 |          1 | 0.10%                  | -7.39%           |            0.9473 |            1.0004 |
| 2025-04-07 | -3.67%             |        1 |          1 | 0.00%                  | -3.67%           |            0.9125 |            0.9637 |
| 2025-04-08 | -4.98%             |        1 |          1 | 0.00%                  | -4.98%           |            0.867  |            0.9157 |
| 2025-04-09 | 15.33%             |        1 |          1 | 0.00%                  | 15.33%           |            1      |            1.056  |
| 2025-04-10 | -4.24%             |        1 |          1 | 0.00%                  | -4.24%           |            0.9576 |            1.0113 |
| 2025-04-11 | 4.06%              |        1 |          1 | 0.00%                  | 4.06%            |            0.9964 |            1.0523 |
| 2025-04-16 | -3.89%             |        1 |          0 | 0.00%                  | -0.00%           |            1.0174 |            1.0317 |


## 9. Experiment Registry

The registry records formal strategy experiments and their decisions.

| Experiment_ID              | Strategy                        | Evaluation_Period   | Strategy_Total_Return   | Buy_Hold_Total_Return   |   Strategy_Sharpe |   Buy_Hold_Sharpe | HAC_P_Value   | Decision   |
|:---------------------------|:--------------------------------|:--------------------|:------------------------|:------------------------|------------------:|------------------:|:--------------|:-----------|
| AAPL_MR_OOS_2024_2025      | Mean Reversion                  | 2024-2025           | 8.72%                   | 42.54%                  |            0.3495 |            0.7742 | 0.3546        | REJECT     |
| AAPL_SMA_BASELINE          | Moving Average                  | 2021-2025           | 33.15%                  | 115.81%                 |            0.4113 |            0.6932 | Not available | REJECT     |
| AAPL_MR_OPEN_OOS_2024_2025 | Mean Reversion (Open Execution) | 2024-2025           | 11.50%                  | 47.28%                  |            0.4018 |            0.7998 | 0.3404        | REJECT     |

The registry currently contains 3 experiments,
of which 3 are rejected.

The moving-average registry entry does not have a verified
HAC significance result. Its unavailable p-value is not
interpreted as evidence of statistical significance.

The close-execution and open-execution mean-reversion entries
have recorded HAC p-values, neither of which indicates
statistical significance at the 5% level.



## 10. Limitations

The research has several important limitations:

1. The primary experiments use a single equity asset, AAPL.
2. Historical performance does not guarantee future performance.
3. Transaction-cost assumptions simplify real-world execution.
4. Market impact, bid-ask spreads, liquidity constraints and
   partial fills are not fully modeled.
5. Some experiments use different execution conventions and
   evaluation periods, so their returns are not directly comparable.
6. Parameter sensitivity does not eliminate data-snooping risk.
7. Machine-learning features are primarily derived from historical
   price data.
8. The experiment registry contains a moving-average record without
   a verified HAC significance result.
9. Some older diagnostic outputs were generated independently
   in notebooks and have not been fully integrated into the
   formal experiment registry.
10. The framework is a research prototype, not a production
    trading system.

## 11. Future Research

Potential extensions include:

- Testing across multiple assets and market regimes.
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
outperformed buy-and-hold after the tested transaction costs and
evaluation procedures.

The machine-learning experiments also failed to demonstrate
reliable predictive superiority over simple baselines.

The main contribution of this project is its research framework:
strategies are tested under explicit assumptions, evaluated against
benchmarks, examined for statistical support and rejected when
the evidence is insufficient.

The results should be interpreted as historical research findings,
not investment advice or evidence of a deployable trading edge.
