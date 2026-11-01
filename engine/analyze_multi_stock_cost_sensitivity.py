from pathlib import Path

import numpy as np
import pandas as pd

from engine.metrics import calculate_performance_metrics


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

DAILY_PATH = (
    REPORTS_DIR / "multi_stock_mean_reversion_open_daily.csv"
)

SUMMARY_PATH = (
    REPORTS_DIR / "multi_stock_mean_reversion_open_summary.csv"
)

DETAILED_OUTPUT = (
    REPORTS_DIR / "multi_stock_cost_sensitivity_detailed.csv"
)

SUMMARY_OUTPUT = (
    REPORTS_DIR / "multi_stock_cost_sensitivity_summary.csv"
)

TRANSACTION_COSTS = [
    0.0000,
    0.0005,
    0.0010,
    0.0025,
    0.0050,
]

ORIGINAL_COST = 0.001


# --------------------------------------------------
# LOAD AND VALIDATE DAILY RESULTS
# --------------------------------------------------

def load_daily_results():

    if not DAILY_PATH.exists():
        raise FileNotFoundError(DAILY_PATH)

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(SUMMARY_PATH)

    daily = pd.read_csv(
        DAILY_PATH,
        parse_dates=["Date"],
    )

    original_summary = pd.read_csv(SUMMARY_PATH)

    required_columns = {
        "Date",
        "ticker",
        "sector",
        "Gross_Return",
        "Asset_Return",
        "Turnover",
        "Net_Return",
    }

    missing = required_columns - set(daily.columns)

    if missing:
        raise ValueError(
            f"Missing daily columns: {sorted(missing)}"
        )

    if daily.empty:
        raise ValueError("Daily results are empty.")

    numeric_columns = [
        "Gross_Return",
        "Asset_Return",
        "Turnover",
        "Net_Return",
    ]

    for column in numeric_columns:

        daily[column] = pd.to_numeric(
            daily[column],
            errors="coerce",
        )

        if daily[column].isna().any():
            raise ValueError(
                f"Invalid values found in {column}."
            )

        if not np.isfinite(daily[column]).all():
            raise ValueError(
                f"Non-finite values found in {column}."
            )

    if (daily["Turnover"] < 0).any():
        raise ValueError("Negative turnover detected.")

    if daily.duplicated(["ticker", "Date"]).any():
        raise ValueError(
            "Duplicate ticker/date observations detected."
        )

    # Confirm that the original 0.10% transaction-cost
    # results can be reconstructed from gross returns.
    reconstructed_original = (
        daily["Gross_Return"]
        - daily["Turnover"] * ORIGINAL_COST
    )

    if not np.allclose(
        reconstructed_original,
        daily["Net_Return"],
        atol=1e-10,
    ):
        raise ValueError(
            "Daily returns do not match the original "
            "transaction-cost calculation."
        )

    return daily, original_summary


# --------------------------------------------------
# RECONSTRUCT STRATEGY RETURNS AT DIFFERENT COSTS
# --------------------------------------------------

def calculate_cost_sensitivity(daily, original_summary):

    detailed_records = []

    original_returns = (
        original_summary
        .set_index("ticker")["Strategy_Total Return"]
    )

    benchmark_returns = {}

    for ticker, stock_data in daily.groupby("ticker"):

        stock_data = stock_data.sort_values("Date")

        if stock_data.empty:
            raise ValueError(
                f"No observations found for {ticker}."
            )

        # Benchmark remains unchanged across cost scenarios.
        benchmark = calculate_performance_metrics(
            stock_data["Asset_Return"]
        )

        benchmark_returns[ticker] = benchmark

        for cost in TRANSACTION_COSTS:

            net_returns = (
                stock_data["Gross_Return"]
                - stock_data["Turnover"] * cost
            )

            strategy = calculate_performance_metrics(
                net_returns
            )

            excess_return = (
                strategy["Total Return"]
                - benchmark["Total Return"]
            )

            detailed_records.append({
                "ticker": ticker,
                "sector": stock_data["sector"].iloc[0],
                "transaction_cost": cost,
                "observations": len(stock_data),

                "Strategy_Total_Return":
                    strategy["Total Return"],

                "BuyHold_Total_Return":
                    benchmark["Total Return"],

                "Excess_Total_Return":
                    excess_return,

                "Strategy_Sharpe":
                    strategy["Sharpe Ratio (rf=0)"],

                "BuyHold_Sharpe":
                    benchmark["Sharpe Ratio (rf=0)"],

                "Strategy_Maximum_Drawdown":
                    strategy["Maximum Drawdown"],

                "BuyHold_Maximum_Drawdown":
                    benchmark["Maximum Drawdown"],

                "Beats_Buy_Hold":
                    strategy["Total Return"]
                    > benchmark["Total Return"],
            })

    detailed = pd.DataFrame(detailed_records)

    # Check that the reconstructed 0.10% scenario
    # matches the original experiment's summary.
    baseline = detailed.loc[
        np.isclose(
            detailed["transaction_cost"],
            ORIGINAL_COST,
        )
    ].set_index("ticker")

    for ticker, original_return in original_returns.items():

        if ticker not in baseline.index:
            raise ValueError(
                f"Missing baseline result for {ticker}."
            )

        reconstructed_return = baseline.loc[
            ticker, "Strategy_Total_Return"
        ]

        if not np.isclose(
            reconstructed_return,
            original_return,
            atol=1e-8,
        ):
            raise ValueError(
                f"{ticker}: Reconstructed baseline return "
                "does not match the original experiment."
            )

    return detailed


# --------------------------------------------------
# AGGREGATE RESULTS ACROSS STOCKS
# --------------------------------------------------

def calculate_aggregate_statistics(detailed):

    records = []

    for cost, group in detailed.groupby(
        "transaction_cost",
        sort=True,
    ):

        records.append({
            "transaction_cost": cost,
            "stocks": len(group),

            "Mean_Strategy_Return":
                group["Strategy_Total_Return"].mean(),

            "Median_Strategy_Return":
                group["Strategy_Total_Return"].median(),

            "Mean_Excess_Return":
                group["Excess_Total_Return"].mean(),

            "Median_Excess_Return":
                group["Excess_Total_Return"].median(),

            "Mean_Strategy_Sharpe":
                group["Strategy_Sharpe"].mean(),

            "Median_Strategy_Sharpe":
                group["Strategy_Sharpe"].median(),

            "Mean_Strategy_Max_Drawdown":
                group["Strategy_Maximum_Drawdown"].mean(),

            "Stocks_Beating_BuyHold":
                int(group["Beats_Buy_Hold"].sum()),

            "Percent_Beating_BuyHold":
                group["Beats_Buy_Hold"].mean() * 100,
        })

    return pd.DataFrame(records)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 75)
    print("MULTI-STOCK TRANSACTION-COST SENSITIVITY ANALYSIS")
    print("=" * 75)

    daily, original_summary = load_daily_results()

    print("\nLoaded daily observations:", len(daily))
    print("Stocks:", daily["ticker"].nunique())

    print("\nTesting transaction costs:")

    for cost in TRANSACTION_COSTS:
        print(f"{cost:.2%}")

    detailed = calculate_cost_sensitivity(
        daily,
        original_summary,
    )

    aggregate = calculate_aggregate_statistics(
        detailed
    )

    detailed.to_csv(DETAILED_OUTPUT, index=False)
    aggregate.to_csv(SUMMARY_OUTPUT, index=False)

    print("\n" + "=" * 100)
    print("AGGREGATE CROSS-STOCK COST SENSITIVITY")
    print("=" * 100)

    print(
        aggregate.round(4).to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("PER-STOCK RESULTS")
    print("=" * 100)

    display_columns = [
        "ticker",
        "transaction_cost",
        "Strategy_Total_Return",
        "BuyHold_Total_Return",
        "Excess_Total_Return",
        "Strategy_Sharpe",
        "Beats_Buy_Hold",
    ]

    print(
        detailed[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nREPORTS SAVED")
    print("-" * 75)
    print(DETAILED_OUTPUT)
    print(SUMMARY_OUTPUT)

    print(
        "\nTRANSACTION-COST SENSITIVITY "
        "ANALYSIS COMPLETED SUCCESSFULLY."
    )


if __name__ == "__main__":
    main()