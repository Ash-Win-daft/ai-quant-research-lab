
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "multi_stock_mean_reversion_open_daily.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "multi_stock_hac_lag_sensitivity.csv"
)

TRANSACTION_COST = 0.001
HAC_LAGS = [1, 5, 10, 20]
ALPHA = 0.05


def load_and_validate_data():
    required_columns = [
        "Date",
        "ticker",
        "Net_Return",
        "Asset_Return",
        "Gross_Return",
        "Turnover",
    ]

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["Date"] = pd.to_datetime(df["Date"])

    numeric_columns = [
        "Net_Return",
        "Asset_Return",
        "Gross_Return",
        "Turnover",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    if df[required_columns].isna().any().any():
        raise ValueError("Missing values detected.")

    if not np.isfinite(
        df[numeric_columns].to_numpy()
    ).all():
        raise ValueError("Non-finite numerical values detected.")

    if df.duplicated(["Date", "ticker"]).any():
        raise ValueError("Duplicate ticker/date observations.")

    reconstructed = (
        df["Gross_Return"]
        - df["Turnover"] * TRANSACTION_COST
    )

    if not np.allclose(
        df["Net_Return"],
        reconstructed,
        atol=1e-10,
    ):
        raise ValueError(
            "Transaction-cost reconstruction failed."
        )

    return df


def build_equal_weighted_portfolio(df):
    stock_counts = df.groupby("Date")["ticker"].nunique()

    expected_stocks = df["ticker"].nunique()

    if not (stock_counts == expected_stocks).all():
        raise ValueError(
            "Not every date contains the full stock universe."
        )

    portfolio = (
        df.groupby("Date")
        .agg(
            Strategy_Return=("Net_Return", "mean"),
            Benchmark_Return=("Asset_Return", "mean"),
        )
        .sort_index()
    )

    portfolio["Excess_Return"] = (
        portfolio["Strategy_Return"]
        - portfolio["Benchmark_Return"]
    )

    return portfolio


def run_hac_test(excess_returns, maxlags):
    observations = len(excess_returns)

    if observations <= maxlags:
        raise ValueError(
            f"Insufficient observations for {maxlags} lags."
        )

    model = sm.OLS(
        excess_returns.to_numpy(),
        np.ones((observations, 1)),
    )

    result = model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags},
    )

    mean_excess = float(result.params[0])
    standard_error = float(result.bse[0])
    t_statistic = float(result.tvalues[0])
    p_value = float(result.pvalues[0])

    confidence_interval = result.conf_int(alpha=ALPHA)[0]

    return {
        "HAC_Max_Lags": maxlags,
        "Observations": observations,
        "Mean_Daily_Excess_Return": mean_excess,
        "Annualized_Mean_Excess_Return": mean_excess * 252,
        "HAC_Standard_Error": standard_error,
        "HAC_T_Statistic": t_statistic,
        "HAC_P_Value": p_value,
        "CI_95_Lower": float(confidence_interval[0]),
        "CI_95_Upper": float(confidence_interval[1]),
        "Significant_At_5_Percent": p_value < ALPHA,
    }


def main():
    print("=" * 75)
    print("PORTFOLIO HAC LAG SENSITIVITY ANALYSIS")
    print("=" * 75)

    df = load_and_validate_data()

    portfolio = build_equal_weighted_portfolio(df)

    excess_returns = portfolio["Excess_Return"]

    print(f"\nStocks: {df['ticker'].nunique()}")
    print(f"Daily observations: {len(portfolio)}")
    print(f"Transaction cost: {TRANSACTION_COST}")
    print(f"HAC lags tested: {HAC_LAGS}")
    print(f"Significance level: {ALPHA}")

    results = []

    for lag in HAC_LAGS:
        results.append(
            run_hac_test(excess_returns, lag)
        )

    results_df = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 95)
    print("HAC LAG SENSITIVITY RESULTS")
    print("=" * 95)

    display_columns = [
        "HAC_Max_Lags",
        "Observations",
        "Mean_Daily_Excess_Return",
        "HAC_Standard_Error",
        "HAC_T_Statistic",
        "HAC_P_Value",
        "CI_95_Lower",
        "CI_95_Upper",
        "Significant_At_5_Percent",
    ]

    print(
        results_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    significant_count = results_df[
        "Significant_At_5_Percent"
    ].sum()

    print("\nSUMMARY")
    print("-" * 75)
    print(
        f"Specifications tested: {len(HAC_LAGS)}"
    )
    print(
        f"Significant negative-excess tests: "
        f"{significant_count}/{len(HAC_LAGS)}"
    )

    if significant_count == len(HAC_LAGS):
        print(
            "Conclusion: Negative portfolio excess returns "
            "remain statistically significant across all "
            "tested HAC lag specifications."
        )
    elif significant_count == 0:
        print(
            "Conclusion: The portfolio result is not "
            "statistically significant under any tested "
            "HAC lag specification."
        )
    else:
        print(
            "Conclusion: Statistical significance depends "
            "on the HAC lag specification."
        )

    print("\nREPORT SAVED")
    print(OUTPUT_FILE)

    print("\nHAC LAG SENSITIVITY ANALYSIS COMPLETED.")


if __name__ == "__main__":
    main()
