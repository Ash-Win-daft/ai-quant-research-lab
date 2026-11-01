from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

DAILY_PATH = (
    REPORTS_DIR / "multi_stock_mean_reversion_open_daily.csv"
)

INDIVIDUAL_OUTPUT = (
    REPORTS_DIR / "multi_stock_hac_significance.csv"
)

PORTFOLIO_OUTPUT = (
    REPORTS_DIR / "multi_stock_portfolio_hac_significance.csv"
)

TRANSACTION_COST = 0.001
HAC_MAXLAGS = 5
SIGNIFICANCE_LEVEL = 0.05


# --------------------------------------------------
# LOAD AND VALIDATE DAILY RETURNS
# --------------------------------------------------

def load_daily_returns():

    if not DAILY_PATH.exists():
        raise FileNotFoundError(DAILY_PATH)

    daily = pd.read_csv(
        DAILY_PATH,
        parse_dates=["Date"],
    )

    required_columns = {
        "Date",
        "ticker",
        "Net_Return",
        "Asset_Return",
        "Gross_Return",
        "Turnover",
    }

    missing = required_columns - set(daily.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if daily.empty:
        raise ValueError("Daily return dataset is empty.")

    if daily.duplicated(["Date", "ticker"]).any():
        raise ValueError("Duplicate ticker/date observations.")

    numeric_columns = [
        "Net_Return",
        "Asset_Return",
        "Gross_Return",
        "Turnover",
    ]

    for column in numeric_columns:

        daily[column] = pd.to_numeric(
            daily[column],
            errors="coerce",
        )

        if daily[column].isna().any():
            raise ValueError(
                f"Missing or invalid values in {column}."
            )

        if not np.isfinite(daily[column]).all():
            raise ValueError(
                f"Non-finite values in {column}."
            )

    # Verify that the original transaction-cost convention
    # is still represented in the daily return data.
    reconstructed = (
        daily["Gross_Return"]
        - daily["Turnover"] * TRANSACTION_COST
    )

    if not np.allclose(
        reconstructed,
        daily["Net_Return"],
        atol=1e-10,
    ):
        raise ValueError(
            "Net returns do not match the original "
            "0.10% transaction-cost assumption."
        )

    return daily


# --------------------------------------------------
# NEWey-WEST HAC MEAN TEST
# --------------------------------------------------

def hac_mean_test(excess_returns):

    if not isinstance(excess_returns, pd.Series):
        excess_returns = pd.Series(excess_returns)

    excess_returns = excess_returns.dropna()

    if len(excess_returns) <= HAC_MAXLAGS:
        raise ValueError(
            "Insufficient observations for the requested HAC test."
        )

    if not np.isfinite(excess_returns).all():
        raise ValueError("Non-finite excess returns detected.")

    # Test whether the mean daily excess return is zero.
    model = sm.OLS(
        excess_returns.to_numpy(),
        np.ones((len(excess_returns), 1)),
    )

    result = model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_MAXLAGS},
    )

    confidence_interval = result.conf_int(alpha=0.05)[0]

    return {
        "Observations": len(excess_returns),
        "Mean_Daily_Excess_Return": excess_returns.mean(),
        "Annualized_Mean_Excess_Return": (
            excess_returns.mean() * 252
        ),
        "HAC_Standard_Error": result.bse[0],
        "HAC_T_Statistic": result.tvalues[0],
        "HAC_P_Value": result.pvalues[0],
        "CI_95_Lower": confidence_interval[0],
        "CI_95_Upper": confidence_interval[1],
    }


# --------------------------------------------------
# INDIVIDUAL STOCK TESTS
# --------------------------------------------------

def calculate_individual_tests(daily):

    records = []

    for ticker, stock_data in daily.groupby("ticker"):

        stock_data = stock_data.sort_values("Date")

        excess_returns = (
            stock_data["Net_Return"]
            - stock_data["Asset_Return"]
        )

        test = hac_mean_test(excess_returns)

        records.append({
            "ticker": ticker,
            **test,
        })

    results = pd.DataFrame(records)

    if len(results) != daily["ticker"].nunique():
        raise ValueError(
            "Not all stocks received a significance test."
        )

    # Correct for testing ten stocks simultaneously.
    reject, adjusted_pvalues, _, _ = multipletests(
        results["HAC_P_Value"],
        alpha=SIGNIFICANCE_LEVEL,
        method="holm",
    )

    results["Holm_Adjusted_P_Value"] = adjusted_pvalues
    results["Significant_After_Holm"] = reject

    results["Raw_Conclusion"] = np.where(
        results["HAC_P_Value"] < SIGNIFICANCE_LEVEL,
        "Significant",
        "Not significant",
    )

    results["Holm_Conclusion"] = np.where(
        results["Significant_After_Holm"],
        "Significant after Holm correction",
        "Not significant after Holm correction",
    )

    return results.sort_values("HAC_P_Value")


# --------------------------------------------------
# EQUAL-WEIGHTED PORTFOLIO TEST
# --------------------------------------------------

def calculate_portfolio_test(daily):

    # Each date must contain exactly one observation
    # for every stock in the research universe.
    counts = daily.groupby("Date")["ticker"].nunique()

    expected_stocks = daily["ticker"].nunique()

    if not (counts == expected_stocks).all():
        raise ValueError(
            "Portfolio dates do not contain all stocks."
        )

    # Equal-weighted daily rebalanced portfolio.
    portfolio = (
        daily.groupby("Date")
        .agg(
            Strategy_Return=("Net_Return", "mean"),
            Benchmark_Return=("Asset_Return", "mean"),
        )
        .sort_index()
    )

    portfolio_excess = (
        portfolio["Strategy_Return"]
        - portfolio["Benchmark_Return"]
    )

    test = hac_mean_test(portfolio_excess)

    test["Stocks"] = expected_stocks
    test["Portfolio_Construction"] = (
        "Equal-weighted daily rebalanced"
    )

    test["Conclusion"] = (
        "Significant"
        if test["HAC_P_Value"] < SIGNIFICANCE_LEVEL
        else "Not significant"
    )

    return pd.DataFrame([test])


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 75)
    print("MULTI-STOCK NEWey-WEST HAC SIGNIFICANCE ANALYSIS")
    print("=" * 75)

    daily = load_daily_returns()

    print("\nStocks:", daily["ticker"].nunique())
    print("Daily observations:", len(daily))
    print("Transaction cost:", TRANSACTION_COST)
    print("HAC maximum lags:", HAC_MAXLAGS)
    print("Significance level:", SIGNIFICANCE_LEVEL)

    individual = calculate_individual_tests(daily)
    portfolio = calculate_portfolio_test(daily)

    individual.to_csv(
        INDIVIDUAL_OUTPUT,
        index=False,
    )

    portfolio.to_csv(
        PORTFOLIO_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 100)
    print("INDIVIDUAL STOCK HAC TESTS")
    print("=" * 100)

    display_columns = [
        "ticker",
        "Observations",
        "Mean_Daily_Excess_Return",
        "Annualized_Mean_Excess_Return",
        "HAC_T_Statistic",
        "HAC_P_Value",
        "Holm_Adjusted_P_Value",
        "Holm_Conclusion",
    ]

    print(
        individual[display_columns]
        .round(6)
        .to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("EQUAL-WEIGHTED PORTFOLIO HAC TEST")
    print("=" * 100)

    print(portfolio.round(6).to_string(index=False))

    print("\nSUMMARY")
    print("-" * 75)

    print(
        "Stocks significant before correction:",
        int(
            (
                individual["HAC_P_Value"]
                < SIGNIFICANCE_LEVEL
            ).sum()
        ),
    )

    print(
        "Stocks significant after Holm correction:",
        int(individual["Significant_After_Holm"].sum()),
    )

    print(
        "Portfolio HAC p-value:",
        f"{portfolio.iloc[0]['HAC_P_Value']:.6f}",
    )

    print("\nREPORTS SAVED")
    print(INDIVIDUAL_OUTPUT)
    print(PORTFOLIO_OUTPUT)

    print("\nHAC SIGNIFICANCE ANALYSIS COMPLETED.")


if __name__ == "__main__":
    main()