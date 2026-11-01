
from pathlib import Path
import argparse

import pandas as pd
import yfinance as yf

from data.validate_market_data import validate_market_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent

UNIVERSE_PATH = PROJECT_ROOT / "config" / "stock_universe.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "multi_stock"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "multi_stock_download_summary.csv"

START_DATE = "2021-01-01"
END_DATE = "2026-01-10"  # Exclusive: includes data through Jan 9.

REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def load_stock_universe():
    """Load and validate the configured stock universe."""

    if not UNIVERSE_PATH.is_file():
        raise FileNotFoundError(
            f"Stock universe not found: {UNIVERSE_PATH}"
        )

    universe = pd.read_csv(UNIVERSE_PATH)

    required = {"ticker", "company", "sector"}

    if not required.issubset(universe.columns):
        raise ValueError(
            f"Stock universe must contain: {sorted(required)}"
        )

    universe["ticker"] = (
        universe["ticker"].astype(str).str.strip().str.upper()
    )

    if universe.empty:
        raise ValueError("Stock universe is empty.")

    if universe["ticker"].eq("").any():
        raise ValueError("Empty ticker found.")

    if universe["ticker"].duplicated().any():
        raise ValueError("Duplicate tickers found.")

    return universe


def normalize_download(raw):
    """Normalize yfinance output into adjusted OHLCV columns."""

    if raw.empty:
        raise ValueError("No market data returned.")

    if isinstance(raw.columns, pd.MultiIndex):
        # Find the column level containing Open, High, Low, etc.
        for level in range(raw.columns.nlevels):
            values = set(raw.columns.get_level_values(level))

            if set(REQUIRED_COLUMNS).issubset(values):
                raw.columns = raw.columns.get_level_values(level)
                break
        else:
            raise ValueError(
                "Could not identify OHLCV columns in downloaded data."
            )

    raw = raw.loc[:, ~raw.columns.duplicated()]

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in raw.columns
    ]

    if missing:
        raise ValueError(f"Missing downloaded columns: {missing}")

    output = raw[REQUIRED_COLUMNS].copy()

    output.index.name = "Date"
    output = output.reset_index()

    output["Date"] = pd.to_datetime(
        output["Date"],
        errors="raise"
    ).dt.tz_localize(None)

    output = output.sort_values("Date").reset_index(drop=True)

    return output


def download_stock(ticker, overwrite=False):
    """Download, validate and save one stock's adjusted OHLCV data."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"{ticker}_daily.csv"

    if output_path.exists() and not overwrite:
        print(f"\n{ticker}: Existing dataset found. Validating it.")

        df = validate_market_data(output_path)

        return {
            "ticker": ticker,
            "status": "EXISTING_VALIDATED",
            "observations": len(df),
            "first_date": df.index.min().date(),
            "last_date": df.index.max().date(),
            "file": str(output_path.relative_to(PROJECT_ROOT)),
            "error": "",
        }

    print(f"\nDownloading {ticker}...")

    raw = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
        timeout=30,
    )

    output = normalize_download(raw)

    # Validate a temporary file before replacing the final dataset.
    temporary_path = OUTPUT_DIR / f"{ticker}_daily.tmp.csv"

    try:
        output.to_csv(temporary_path, index=False)

        validated = validate_market_data(temporary_path)

        temporary_path.replace(output_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        "ticker": ticker,
        "status": "DOWNLOADED",
        "observations": len(validated),
        "first_date": validated.index.min().date(),
        "last_date": validated.index.max().date(),
        "file": str(output_path.relative_to(PROJECT_ROOT)),
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download historical data for the configured stock universe."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing stock datasets.",
    )

    args = parser.parse_args()

    universe = load_stock_universe()

    print("=" * 65)
    print("MULTI-STOCK HISTORICAL DATA PIPELINE")
    print("=" * 65)

    print(f"Stocks configured: {len(universe)}")
    print(f"Period: {START_DATE} to {END_DATE} (exclusive)")
    print("Price adjustment: auto_adjust=True")

    results = []

    for ticker in universe["ticker"]:
        try:
            result = download_stock(
                ticker,
                overwrite=args.overwrite
            )

            results.append(result)

        except Exception as error:
            print(f"\nERROR downloading {ticker}: {error}")

            results.append({
                "ticker": ticker,
                "status": "FAILED",
                "observations": 0,
                "first_date": None,
                "last_date": None,
                "file": "",
                "error": str(error),
            })

    summary = pd.DataFrame(results)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n" + "=" * 65)
    print("DOWNLOAD SUMMARY")
    print("=" * 65)

    print(summary.to_string(index=False))

    successful = summary["status"].ne("FAILED").sum()
    failed = summary["status"].eq("FAILED").sum()

    print(f"\nSuccessful: {successful}")
    print(f"Failed: {failed}")
    print(f"Summary saved: {SUMMARY_PATH}")

    if failed:
        raise SystemExit(
            "\nSome downloads failed. Check the errors and rerun."
        )

    print("\nMULTI-STOCK DATA PIPELINE COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
