
from pathlib import Path

import pandas as pd

from data.validate_market_data import validate_market_data
from engine.backtester_open import run_open_backtest
from engine.metrics import calculate_performance_metrics


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UNIVERSE_PATH = PROJECT_ROOT / "config" / "stock_universe.csv"
DATA_DIR = PROJECT_ROOT / "data" / "multi_stock"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRANSACTION_COST = 0.001

# Parameters selected using AAPL's 2021-2023 training data.
# These are deliberately fixed across all ten stocks.
WINDOW = 20
THRESHOLD = -0.05

TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

SUMMARY_PATH = (
    REPORTS_DIR / "multi_stock_mean_reversion_open_summary.csv"
)

DAILY_PATH = (
    REPORTS_DIR / "multi_stock_mean_reversion_open_daily.csv"
)


# --------------------------------------------------
# LOAD STOCK UNIVERSE
# --------------------------------------------------

def load_universe():
    universe = pd.read_csv(UNIVERSE_PATH)

    required_columns = {"ticker", "company", "sector"}

    if not required_columns.issubset(universe.columns):
        raise ValueError(
            f"Universe must contain {required_columns}."
        )

    universe["ticker"] = (
        universe["ticker"].astype(str).str.strip().str.upper()
    )

    if universe["ticker"].duplicated().any():
        raise ValueError("Duplicate tickers in stock universe.")

    return universe


# --------------------------------------------------
# LOAD AND VALIDATE STOCK DATA
# --------------------------------------------------

def load_stock_data(ticker):
    file_path = DATA_DIR / f"{ticker}_daily.csv"

    history = validate_market_data(file_path)

    if history.empty:
        raise ValueError(f"{ticker} dataset is empty.")

    opens = history["Open"].copy()
    closes = history["Close"].copy()

    if not opens.index.equals(closes.index):
        raise ValueError(
            f"{ticker}: Open and Close dates do not match."
        )

    return opens, closes


# --------------------------------------------------
# GENERATE MEAN-REVERSION SIGNALS
# --------------------------------------------------

def generate_signals(closes):
    """
    Generate signals using closing prices.

    Signal = 1 when price is at least 5% below
    its 20-day rolling average.

    Signal = 0 otherwise.
    """

    rolling_average = closes.rolling(
        window=WINDOW,
        min_periods=WINDOW,
    ).mean()

    deviation = closes / rolling_average - 1

    signal = (deviation < THRESHOLD).astype(int)

    return signal


# --------------------------------------------------
# RUN ONE STOCK'S OUT-OF-SAMPLE EXPERIMENT
# --------------------------------------------------

def run_stock_experiment(ticker, company, sector, opens, closes):

    signals = generate_signals(closes)

    all_results = run_open_backtest(
        opens=opens,
        closes=closes,
        signal=signals,
        transaction_cost=TRANSACTION_COST,
    )

    # Confirm that the first test position uses the
    # signal from the preceding trading session.
    first_test_date = all_results.index[
        all_results.index >= pd.Timestamp(TEST_START)
    ][0]

    previous_date = signals.index[
        signals.index < first_test_date
    ][-1]

    expected_position = signals.loc[previous_date]
    actual_position = all_results.loc[
        first_test_date, "Position"
    ]

    if expected_position != actual_position:
        raise ValueError(
            f"{ticker}: First test-period position "
            "does not match the previous close signal."
        )

    # Find the final opening price within the test period.
    test_opens = opens.loc[:TEST_END]

    if test_opens.empty:
        raise ValueError(
            f"{ticker}: No opening prices in test period."
        )

    last_test_open = test_opens.index[-1]

    # Keep intervals starting in 2024-2025 and ending
    # no later than the final 2025 opening price.
    test_results = all_results.loc[
        (all_results.index >= pd.Timestamp(TEST_START))
        & (all_results.index < last_test_open)
    ].copy()

    if test_results.empty:
        raise ValueError(
            f"{ticker}: No out-of-sample returns."
        )

    # Reset equity curves to 1.0 at the start of OOS.
    test_results["Strategy_Equity"] = (
        1 + test_results["Net_Return"]
    ).cumprod()

    test_results["Buy_Hold_Equity"] = (
        1 + test_results["Asset_Return"]
    ).cumprod()

    strategy_metrics = calculate_performance_metrics(
        test_results["Net_Return"]
    )

    benchmark_metrics = calculate_performance_metrics(
        test_results["Asset_Return"]
    )

    # Build one consolidated summary record.
    summary = {
        "ticker": ticker,
        "company": company,
        "sector": sector,
        "window": WINDOW,
        "threshold": THRESHOLD,
        "transaction_cost": TRANSACTION_COST,
        "test_start": test_results.index.min().date(),
        "test_end": test_results.index.max().date(),
        "observations": len(test_results),
    }

    for metric, value in strategy_metrics.items():
        summary[f"Strategy_{metric}"] = value

    for metric, value in benchmark_metrics.items():
        summary[f"BuyHold_{metric}"] = value

    summary["Excess_Total_Return"] = (
        strategy_metrics["Total Return"]
        - benchmark_metrics["Total Return"]
    )

    summary["Beats_Buy_Hold"] = (
        strategy_metrics["Total Return"]
        > benchmark_metrics["Total Return"]
    )

    # Prepare daily observations for the consolidated file.
    daily = test_results.copy()

    daily.insert(0, "ticker", ticker)
    daily.insert(1, "company", company)
    daily.insert(2, "sector", sector)

    daily.index.name = "Date"
    daily = daily.reset_index()

    return summary, daily


# --------------------------------------------------
# RUN THE MULTI-STOCK EXPERIMENT
# --------------------------------------------------

def main():

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    universe = load_universe()

    print("=" * 75)
    print("MULTI-STOCK MEAN-REVERSION RESEARCH EXPERIMENT")
    print("=" * 75)

    print(f"Stocks: {len(universe)}")
    print(f"Window: {WINDOW}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Transaction cost: {TRANSACTION_COST}")
    print(f"Test period: {TEST_START} to {TEST_END}")
    print("Execution: Previous-close signal, open-to-open returns")
    print("Parameter source: AAPL training period (2021-2023)")

    summary_records = []
    daily_records = []
    reference_dates = None

    for row in universe.itertuples(index=False):

        ticker = row.ticker
        company = row.company
        sector = row.sector

        print("\n" + "-" * 75)
        print(f"Running experiment: {ticker} ({company})")
        print("-" * 75)

        try:
            opens, closes = load_stock_data(ticker)

            # Ensure all stocks share identical trading dates.
            if reference_dates is None:
                reference_dates = opens.index
            elif not opens.index.equals(reference_dates):
                raise ValueError(
                    f"{ticker}: Trading dates differ from "
                    "the reference stock."
                )

            summary, daily = run_stock_experiment(
                ticker=ticker,
                company=company,
                sector=sector,
                opens=opens,
                closes=closes,
            )

            summary_records.append(summary)
            daily_records.append(daily)

            print(
                "Strategy total return: "
                f"{summary['Strategy_Total Return']:.4f}"
            )

            print(
                "Buy-and-hold total return: "
                f"{summary['BuyHold_Total Return']:.4f}"
            )

            print(
                "Strategy Sharpe ratio: "
                f"{summary['Strategy_Sharpe Ratio (rf=0)']:.4f}"
            )

        except Exception as error:
            print(f"FAILED: {ticker}: {error}")
            raise

    # --------------------------------------------------
    # SAVE CONSOLIDATED RESULTS
    # --------------------------------------------------

    summary_df = pd.DataFrame(summary_records)
    daily_df = pd.concat(daily_records, ignore_index=True)

    if len(summary_df) != len(universe):
        raise ValueError(
            "Not all configured stocks produced results."
        )

    if summary_df["ticker"].duplicated().any():
        raise ValueError("Duplicate ticker results detected.")

    summary_df.to_csv(SUMMARY_PATH, index=False)
    daily_df.to_csv(DAILY_PATH, index=False)

    # --------------------------------------------------
    # DISPLAY CROSS-STOCK COMPARISON
    # --------------------------------------------------

    display_columns = [
        "ticker",
        "Strategy_Total Return",
        "BuyHold_Total Return",
        "Strategy_Sharpe Ratio (rf=0)",
        "BuyHold_Sharpe Ratio (rf=0)",
        "Strategy_Maximum Drawdown",
        "BuyHold_Maximum Drawdown",
        "Excess_Total_Return",
        "Beats_Buy_Hold",
    ]

    print("\n" + "=" * 110)
    print("CROSS-STOCK OUT-OF-SAMPLE COMPARISON")
    print("=" * 110)

    print(
        summary_df[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT SUMMARY")
    print("=" * 75)

    print("Stocks tested:", len(summary_df))
    print(
        "Stocks outperforming buy-and-hold:",
        int(summary_df["Beats_Buy_Hold"].sum()),
    )
    print(
        "Stocks underperforming buy-and-hold:",
        int((~summary_df["Beats_Buy_Hold"]).sum()),
    )

    print("\nReports saved:")
    print(SUMMARY_PATH)
    print(DAILY_PATH)

    print("\nMULTI-STOCK EXPERIMENT COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
