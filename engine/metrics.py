
import numpy as np
import pandas as pd


def calculate_performance_metrics(returns, periods_per_year=252):
    """
    Calculate standardized performance metrics from daily returns.

    Returns must be decimal returns, e.g. 0.01 means 1%.
    """

    if not isinstance(returns, pd.Series):
        raise TypeError("Returns must be a pandas Series.")

    if returns.empty:
        raise ValueError("Returns cannot be empty.")

    if returns.isna().any():
        raise ValueError("Returns contain missing values.")

    if not np.isfinite(returns).all():
        raise ValueError("Returns contain non-finite values.")

    if periods_per_year <= 0:
        raise ValueError("Periods per year must be positive.")

    # Growth of an initial investment of 1 unit
    equity = (1 + returns).cumprod()

    total_return = equity.iloc[-1] - 1

    # Annualized growth rate, based on the number of observations
    years = len(returns) / periods_per_year

    if (1 + total_return) > 0 and years > 0:
        cagr = (1 + total_return) ** (1 / years) - 1
    else:
        cagr = np.nan

    annualized_volatility = (
        returns.std(ddof=1) * np.sqrt(periods_per_year)
        if len(returns) > 1
        else np.nan
    )

    daily_std = returns.std(ddof=1)

    sharpe_ratio = (
        returns.mean() / daily_std * np.sqrt(periods_per_year)
        if pd.notna(daily_std) and daily_std > 0
        else np.nan
    )

    # Include the initial capital value when measuring drawdown
    equity_with_initial = pd.concat([
        pd.Series([1.0]),
        equity.reset_index(drop=True)
    ], ignore_index=True)

    running_peak = equity_with_initial.cummax()
    drawdowns = equity_with_initial / running_peak - 1
    max_drawdown = drawdowns.min()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio (rf=0)": sharpe_ratio,
        "Maximum Drawdown": max_drawdown,
    }
