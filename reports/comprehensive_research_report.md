
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

The experiment registry currently contains **6 experiments**,
of which **6 are rejected**.

The multi-stock mean-reversion experiment evaluated
**10 stocks** over the 2024–2025 out-of-sample period.
None of the ten individual stocks outperformed their respective
benchmarks on total return.

The equal-weighted portfolio also underperformed its benchmark.
Its mean daily excess return was significantly negative under
HAC inference, with a p-value of approximately 0.0086.

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

| Metric                               |   Value |
|:-------------------------------------|--------:|
| Accuracy                             |  0.477  |
| Balanced Accuracy                    |  0.4623 |
| Precision                            |  0.5166 |
| Recall                               |  0.6264 |
| ROC AUC                              |  0.4662 |
| Brier Score                          |  0.2587 |
| Training Majority Baseline Accuracy  |  0.5449 |
| Always-Up Baseline Accuracy          |  0.5449 |
| Historical-Rate Baseline Brier Score |  0.2484 |

### Strategy performance

| Metric                | ML Strategy   | Buy and Hold   |
|:----------------------|:--------------|:---------------|
| Total Return          | 3.24%         | 47.83%         |
| CAGR                  | 1.62%         | 21.73%         |
| Annualized Volatility | 23.44%        | 27.89%         |
| Sharpe Ratio          | 0.1838        | 0.8429         |
| Max Drawdown          | -31.18%       | -33.36%        |

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
| Precision                            |  0.5356 |
| Recall                               |  0.7985 |
| ROC AUC                              |  0.4469 |
| Brier Score                          |  0.2542 |
| Expanding Majority Baseline Accuracy |  0.5449 |
| Historical-Rate Baseline Brier Score |  0.2485 |

### Strategy performance

| Metric                | Walk-Forward ML   | Buy and Hold   |
|:----------------------|:------------------|:---------------|
| Total Return          | 13.00%            | 47.83%         |
| CAGR                  | 6.34%             | 21.73%         |
| Annualized Volatility | 25.38%            | 27.89%         |
| Sharpe Ratio          | 0.3676            | 0.8429         |
| Max Drawdown          | -31.73%           | -33.36%        |

The walk-forward model did not establish predictive performance
above its simple baselines. Its trading strategy also underperformed
buy-and-hold.

Walk-forward evaluation improves the testing design, but does not
by itself establish that a model is free from data leakage or
implementation bias. Those properties depend on the complete
feature-generation and execution pipeline.



## 8. Strategy Diagnostics and Multi-Stock Research

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

| ticker   | company             | sector     |   window |   threshold | transaction_cost   | test_start   | test_end   |   observations | Strategy_Total Return   | Strategy_CAGR   | Strategy_Annualized Volatility   |   Strategy_Sharpe Ratio (rf=0) | Strategy_Maximum Drawdown   | BuyHold_Total Return   | BuyHold_CAGR   | BuyHold_Annualized Volatility   |   BuyHold_Sharpe Ratio (rf=0) | BuyHold_Maximum Drawdown   | Excess_Total_Return   | Beats_Buy_Hold   |
|:---------|:--------------------|:-----------|---------:|------------:|:-------------------|:-------------|:-----------|---------------:|:------------------------|:----------------|:---------------------------------|-------------------------------:|:----------------------------|:-----------------------|:---------------|:--------------------------------|------------------------------:|:---------------------------|:----------------------|:-----------------|
| AAPL     | Apple               | Technology |       20 |       -0.05 | 0.10%              | 2024-01-02   | 2025-12-30 |            501 | 11.50%                  | 5.63%           | 17.21%                           |                         0.4018 | -13.44%                     | 47.28%                 | 21.50%         | 29.90%                          |                        0.7998 | -33.33%                    | -35.78%               | False            |
| MSFT     | Microsoft           | Technology |       20 |       -0.05 | 0.10%              | 2024-01-02   | 2025-12-30 |            501 | 8.39%                   | 4.13%           | 10.12%                           |                         0.45   | -8.46%                      | 32.43%                 | 15.18%         | 24.31%                          |                        0.7019 | -24.43%                    | -24.05%               | False            |
| NVDA     | NVIDIA              | Technology |       20 |       -0.05 | 0.10%              | 2024-01-02   | 2025-12-30 |            501 | 119.12%                 | 48.38%          | 28.40%                           |                         1.528  | -12.94%                     | 285.18%                | 97.05%         | 53.50%                          |                        1.5378 | -42.84%                    | -166.06%              | False            |
| GOOGL    | Alphabet            | Technology |       20 |       -0.05 | 0.10%              | 2024-01-02   | 2025-12-30 |            501 | 12.98%                  | 6.33%           | 11.58%                           |                         0.587  | -9.43%                      | 127.51%                | 51.21%         | 32.85%                          |                        1.4213 | -30.32%                    | -114.53%              | False            |
| AMZN     | Amazon              | Consumer   |       20 |       -0.05 | 0.10%              | 2024-01-02   | 2025-12-30 |            501 | 20.31%                  | 9.75%           | 17.89%                           |                         0.6066 | -14.48%                     | 53.70%                 | 24.13%         | 33.83%                          |                        0.8075 | -32.22%                    | -33.39%               | False            |
| WMT      | Walmart             | Consumer   |       20 |       -0.05 | 0.10%              | 2024-01-02   | 2025-12-30 |            501 | -0.92%                  | -0.46%          | 4.78%                            |                        -0.073  | -5.92%                      | 117.78%                | 47.92%         | 21.86%                          |                        1.9004 | -23.59%                    | -118.69%              | False            |
| JPM      | JPMorgan Chase      | Financials |       20 |       -0.05 | 0.10%              | 2024-01-02   | 2025-12-30 |            501 | 10.03%                  | 4.92%           | 10.47%                           |                         0.5098 | -9.71%                      | 100.55%                | 41.91%         | 24.09%                          |                        1.5741 | -25.71%                    | -90.53%               | False            |
| GS       | Goldman Sachs       | Financials |       20 |       -0.05 | 0.10%              | 2024-01-02   | 2025-12-30 |            501 | 8.80%                   | 4.33%           | 14.81%                           |                         0.3592 | -13.16%                     | 141.25%                | 55.73%         | 29.30%                          |                        1.659  | -32.88%                    | -132.45%              | False            |
| XOM      | Exxon Mobil         | Energy     |       20 |       -0.05 | 0.10%              | 2024-01-02   | 2025-12-30 |            501 | 2.53%                   | 1.26%           | 8.01%                            |                         0.1965 | -5.77%                      | 28.73%                 | 13.54%         | 23.06%                          |                        0.6669 | -19.30%                    | -26.20%               | False            |
| JNJ      | Johnson and Johnson | Healthcare |       20 |       -0.05 | 0.10%              | 2024-01-02   | 2025-12-30 |            501 | 4.45%                   | 2.21%           | 7.22%                            |                         0.3399 | -7.42%                      | 40.19%                 | 18.52%         | 17.88%                          |                        1.0401 | -15.40%                    | -35.74%               | False            |

All ten stocks underperformed their respective benchmarks
on total return at the reference transaction cost of 0.10%.

#### Cross-sectional results

| Metric                        | Mean    | Median   | Standard Deviation   | Minimum   | 25th Percentile   | 75th Percentile   | Maximum   |
|:------------------------------|:--------|:---------|:---------------------|:----------|:------------------|:------------------|:----------|
| Strategy Total Return         | 19.72%  | 9.41%    | 35.42%               | -0.92%    | 5.43%             | 12.61%            | 119.12%   |
| Buy-and-Hold Total Return     | 97.46%  | 77.12%   | 78.28%               | 28.73%    | 41.96%            | 125.08%           | 285.18%   |
| Excess Total Return           | -77.74% | -63.15%  | 52.72%               | -166.06%  | -117.65%          | -33.97%           | -24.05%   |
| Strategy Sharpe Ratio         | 0.4906  | 0.4259   | 0.4154               | -0.0730   | 0.3447            | 0.5677            | 1.5280    |
| Buy-and-Hold Sharpe Ratio     | 1.2109  | 1.2307   | 0.4565               | 0.6669    | 0.8017            | 1.5650            | 1.9004    |
| Strategy Maximum Drawdown     | -10.07% | -9.57%   | 3.24%                | -14.48%   | -13.11%           | -7.68%            | -5.77%    |
| Buy-and-Hold Maximum Drawdown | -28.00% | -28.02%  | 7.94%                | -42.84%   | -32.71%           | -23.80%           | -15.40%   |

#### Sector-level descriptive results

| sector     |   Stocks | Mean_Strategy_Return   | Median_Strategy_Return   | Mean_BuyHold_Return   | Median_BuyHold_Return   | Mean_Excess_Return   | Median_Excess_Return   |   Mean_Strategy_Sharpe |   Mean_BuyHold_Sharpe |   Stocks_Beating_BuyHold |   Percent_Beating_BuyHold |
|:-----------|---------:|:-----------------------|:-------------------------|:----------------------|:------------------------|:---------------------|:-----------------------|-----------------------:|----------------------:|-------------------------:|--------------------------:|
| Consumer   |        2 | 9.70%                  | 9.70%                    | 85.74%                | 85.74%                  | -76.04%              | -76.04%                |                 0.2668 |                1.3539 |                        0 |                         0 |
| Energy     |        1 | 2.53%                  | 2.53%                    | 28.73%                | 28.73%                  | -26.20%              | -26.20%                |                 0.1965 |                0.6669 |                        0 |                         0 |
| Financials |        2 | 9.41%                  | 9.41%                    | 120.90%               | 120.90%                 | -111.49%             | -111.49%               |                 0.4345 |                1.6165 |                        0 |                         0 |
| Healthcare |        1 | 4.45%                  | 4.45%                    | 40.19%                | 40.19%                  | -35.74%              | -35.74%                |                 0.3399 |                1.0401 |                        0 |                         0 |
| Technology |        4 | 38.00%                 | 12.24%                   | 123.10%               | 87.40%                  | -85.10%              | -75.16%                |                 0.7417 |                1.1152 |                        0 |                         0 |

Sector groups are small and unevenly represented. These
descriptive results should not be interpreted as reliable
sector-wide conclusions.

#### Transaction-cost sensitivity

The experiment was reconstructed at several transaction-cost
assumptions. The 0.10% reconstruction was checked against the
original strategy results.

| transaction_cost   |   stocks | Mean_Strategy_Return   | Median_Strategy_Return   | Mean_Excess_Return   | Median_Excess_Return   |   Mean_Strategy_Sharpe |   Median_Strategy_Sharpe | Mean_Strategy_Max_Drawdown   |   Stocks_Beating_BuyHold |   Percent_Beating_BuyHold |
|:-------------------|---------:|:-----------------------|:-------------------------|:---------------------|:-----------------------|-----------------------:|-------------------------:|:-----------------------------|-------------------------:|--------------------------:|
| 0.00%              |       10 | 22.02%                 | 10.95%                   | -75.44%              | -62.06%                |                 0.5557 |                   0.4898 | -9.78%                       |                        0 |                         0 |
| 0.05%              |       10 | 20.86%                 | 10.18%                   | -76.60%              | -62.59%                |                 0.5232 |                   0.4579 | -9.93%                       |                        0 |                         0 |
| 0.10%              |       10 | 19.72%                 | 9.41%                    | -77.74%              | -63.15%                |                 0.4906 |                   0.4259 | -10.07%                      |                        0 |                         0 |
| 0.25%              |       10 | 16.37%                 | 7.15%                    | -81.08%              | -65.79%                |                 0.3926 |                   0.3297 | -10.64%                      |                        0 |                         0 |
| 0.50%              |       10 | 11.06%                 | 2.27%                    | -86.40%              | -70.04%                |                 0.2297 |                   0.1539 | -11.74%                      |                        0 |                         0 |

Even with zero transaction costs, none of the ten stocks
outperformed its benchmark. Transaction costs worsened the
results but did not explain the underlying underperformance.

#### Individual-stock HAC significance tests

The individual-stock tests evaluate whether each stock's
mean daily strategy excess return differs from zero.

Holm correction is applied across the ten individual tests
to control the family-wise error rate.

| ticker   |   Observations | Mean_Daily_Excess_Return   | Annualized_Mean_Excess_Return   | HAC_Standard_Error   |   HAC_T_Statistic |   HAC_P_Value | CI_95_Lower   | CI_95_Upper   |   Holm_Adjusted_P_Value | Significant_After_Holm   | Raw_Conclusion   | Holm_Conclusion                       |
|:---------|---------------:|:---------------------------|:--------------------------------|:---------------------|------------------:|--------------:|:--------------|:--------------|------------------------:|:-------------------------|:-----------------|:--------------------------------------|
| WMT      |            501 | -0.17%                     | -41.89%                         | 0.06%                |           -2.8778 |        0.004  | -0.28%        | -0.05%        |                  0.0401 | True                     | Significant      | Significant after Holm correction     |
| GS       |            501 | -0.17%                     | -43.28%                         | 0.07%                |           -2.349  |        0.0188 | -0.32%        | -0.03%        |                  0.1694 | False                    | Significant      | Not significant after Holm correction |
| JPM      |            501 | -0.13%                     | -32.58%                         | 0.06%                |           -2.0373 |        0.0416 | -0.25%        | -0.00%        |                  0.3329 | False                    | Significant      | Not significant after Holm correction |
| GOOGL    |            501 | -0.16%                     | -39.89%                         | 0.08%                |           -1.9877 |        0.0469 | -0.31%        | -0.00%        |                  0.3329 | False                    | Significant      | Not significant after Holm correction |
| JNJ      |            501 | -0.06%                     | -16.14%                         | 0.05%                |           -1.3655 |        0.1721 | -0.16%        | 0.03%         |                  1      | False                    | Not significant  | Not significant after Holm correction |
| NVDA     |            501 | -0.15%                     | -38.88%                         | 0.13%                |           -1.183  |        0.2368 | -0.41%        | 0.10%         |                  1      | False                    | Not significant  | Not significant after Holm correction |
| AAPL     |            501 | -0.07%                     | -17.00%                         | 0.07%                |           -0.9542 |        0.34   | -0.21%        | 0.07%         |                  1      | False                    | Not significant  | Not significant after Holm correction |
| XOM      |            501 | -0.05%                     | -13.80%                         | 0.06%                |           -0.8953 |        0.3706 | -0.17%        | 0.07%         |                  1      | False                    | Not significant  | Not significant after Holm correction |
| MSFT     |            501 | -0.05%                     | -12.51%                         | 0.06%                |           -0.8554 |        0.3924 | -0.16%        | 0.06%         |                  1      | False                    | Not significant  | Not significant after Holm correction |
| AMZN     |            501 | -0.07%                     | -16.47%                         | 0.08%                |           -0.7801 |        0.4353 | -0.23%        | 0.10%         |                  1      | False                    | Not significant  | Not significant after Holm correction |

Four stocks had unadjusted p-values below 0.05. After Holm
correction, only WMT remained significant at the 5% level.
Its significant result represents negative mean daily excess
return, not evidence of a profitable strategy.

#### Equal-weighted portfolio HAC test

The portfolio is constructed by averaging the ten constituent
strategy returns each day. The benchmark is constructed by
averaging the ten constituent asset returns each day. Both
are therefore equal-weighted and daily rebalanced.

|   Observations | Mean_Daily_Excess_Return   | Annualized_Mean_Excess_Return   | HAC_Standard_Error   |   HAC_T_Statistic |   HAC_P_Value | CI_95_Lower   | CI_95_Upper   |   Stocks | Portfolio_Construction          | Conclusion   |
|---------------:|:---------------------------|:--------------------------------|:---------------------|------------------:|--------------:|:--------------|:--------------|---------:|:--------------------------------|:-------------|
|            501 | -0.11%                     | -27.24%                         | 0.04%                |           -2.6259 |        0.0086 | -0.19%        | -0.03%        |       10 | Equal-weighted daily rebalanced | Significant  |

The portfolio contained 501 daily observations. Its mean
daily excess return was approximately -0.1081%, with a HAC
p-value of 0.008642. The 95% confidence interval was entirely
negative.

#### HAC lag sensitivity

|   HAC_Max_Lags |   Observations | Mean_Daily_Excess_Return   | Annualized_Mean_Excess_Return   | HAC_Standard_Error   |   HAC_T_Statistic |   HAC_P_Value | CI_95_Lower   | CI_95_Upper   | Significant_At_5_Percent   |
|---------------:|---------------:|:---------------------------|:--------------------------------|:---------------------|------------------:|--------------:|:--------------|:--------------|:---------------------------|
|              1 |            501 | -0.11%                     | -27.24%                         | 0.04%                |           -2.6981 |        0.007  | -0.19%        | -0.03%        | True                       |
|              5 |            501 | -0.11%                     | -27.24%                         | 0.04%                |           -2.6259 |        0.0086 | -0.19%        | -0.03%        | True                       |
|             10 |            501 | -0.11%                     | -27.24%                         | 0.04%                |           -2.541  |        0.0111 | -0.19%        | -0.02%        | True                       |
|             20 |            501 | -0.11%                     | -27.24%                         | 0.04%                |           -2.5138 |        0.0119 | -0.19%        | -0.02%        | True                       |

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



## 9. Experiment Registry

The registry records formal strategy experiments and their decisions.

| Experiment_ID                        | Strategy                         | Evaluation_Period   | Strategy_Total_Return   | Buy_Hold_Total_Return   |   Strategy_Sharpe |   Buy_Hold_Sharpe | HAC_P_Value   | Decision   |
|:-------------------------------------|:---------------------------------|:--------------------|:------------------------|:------------------------|------------------:|------------------:|:--------------|:-----------|
| AAPL_MR_OOS_2024_2025                | Mean Reversion                   | 2024-2025           | 8.72%                   | 42.54%                  |            0.3495 |            0.7742 | 0.3546        | REJECT     |
| AAPL_SMA_BASELINE                    | Moving Average                   | 2021-2025           | 33.15%                  | 115.81%                 |            0.4113 |            0.6932 | Not available | REJECT     |
| AAPL_MR_OPEN_OOS_2024_2025           | Mean Reversion (Open Execution)  | 2024-2025           | 11.50%                  | 47.28%                  |            0.4018 |            0.7998 | 0.3404        | REJECT     |
| US10_MR_OPEN_PORTFOLIO_OOS_2024_2025 | Mean Reversion (Open Execution)  | 2024-2025           | 17.69%                  | 97.34%                  |            0.8649 |            1.9342 | 0.0086        | REJECT     |
| AAPL_STATIC_ML_OOS_2024_2025         | Static Logistic Regression       | 2024-2025           | 3.24%                   | 47.83%                  |            0.1838 |            0.8429 | Not available | REJECT     |
| AAPL_WALK_FORWARD_ML_OOS_2024_2025   | Walk-Forward Logistic Regression | 2024-2025           | 13.00%                  | 47.83%                  |            0.3676 |            0.8429 | Not available | REJECT     |

The registry currently contains 6 experiments,
of which 6 are rejected.

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
