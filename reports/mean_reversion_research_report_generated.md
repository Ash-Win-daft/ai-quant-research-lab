# AI Quant Research Lab
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

**Selected window:** 20 trading sessions  
**Selected entry threshold:** -5.0%  
**Research-period Sharpe ratio:** 0.9083

### Research-period parameter results

| Window | Entry threshold | Total return | Sharpe ratio |
|---:|---:|---:|---:|
| 10 | -2.0% | 16.04% | 0.4012 |
| 10 | -3.0% | 24.52% | 0.6162 |
| 20 | -2.0% | 0.80% | 0.1052 |
| 20 | -3.0% | 16.86% | 0.3921 |
| 20 | -5.0% | 38.14% | 0.9083 |
| 50 | -3.0% | 36.55% | 0.6773 |
| 50 | -5.0% | 26.56% | 0.5971 |

These results describe the training period. They are not out-of-sample
evidence of future performance.

## 3. Out-of-sample evaluation

The selected parameters were applied to the separate 2024–2025 test
period. The test period was not used to select the configuration.

### Performance comparison

| Metric | Mean reversion | Buy and hold |
|---|---:|---:|
| Total return | 8.72% | 42.54% |
| CAGR | 4.29% | 19.47% |
| Annualized volatility | 15.18% | 27.99% |
| Sharpe ratio (risk-free rate = 0) | 0.3495 | 0.7742 |
| Maximum drawdown | -18.96% | -33.36% |

## 4. Interpretation

The mean-reversion strategy returned 8.72%
during the out-of-sample period, compared with
42.54% for buy and hold.

Its Sharpe ratio was 0.3495, compared with
0.7742 for buy and hold.

The strategy's maximum drawdown was -18.96%,
while the benchmark's was -33.36%.

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
