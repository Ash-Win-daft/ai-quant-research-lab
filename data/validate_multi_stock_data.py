
from pathlib import Path

import pandas as pd

from data.validate_market_data import validate_market_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent

UNIVERSE_PATH = PROJECT_ROOT / "config" / "stock_universe.csv"
DATA_DIR = PROJECT_ROOT / "data" / "multi_stock"

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "multi_stock_validation_summary.csv"
)


def main():
    print("=" * 70)
    print("MULTI-STOCK DATA INTEGRITY VALIDATION")
    print("=" * 70)

    universe = pd.read_csv(UNIVERSE_PATH)

    reference_dates = None
    reference_ticker = None

    results = []

    for ticker in universe["ticker"].astype(str).str.upper():

        file_path = DATA_DIR / f"{ticker}_daily.csv"

        print(f"\nValidating {ticker}...")

        try:
            df = validate_market_data(file_path)

            dates_aligned = True

            if reference_dates is None:
                reference_dates = df.index
                reference_ticker = ticker
            else:
                dates_aligned = df.index.equals(reference_dates)

            if not dates_aligned:
                raise ValueError(
                    f"Trading dates do not match {reference_ticker}."
                )

            results.append({
                "ticker": ticker,
                "status": "PASSED",
                "observations": len(df),
                "first_date": df.index.min().date(),
                "last_date": df.index.max().date(),
                "dates_aligned": dates_aligned,
                "error": "",
            })

        except Exception as error:

            print(f"VALIDATION FAILED: {error}")

            results.append({
                "ticker": ticker,
                "status": "FAILED",
                "observations": 0,
                "first_date": None,
                "last_date": None,
                "dates_aligned": False,
                "error": str(error),
            })

    summary = pd.DataFrame(results)

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(SUMMARY_PATH, index=False)

    successful = (summary["status"] == "PASSED").sum()
    failed = (summary["status"] == "FAILED").sum()

    all_dates_aligned = (
        failed == 0
        and summary["dates_aligned"].all()
    )

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(summary.to_string(index=False))

    print("\nStocks passed:", successful)
    print("Stocks failed:", failed)
    print("Trading dates aligned:", all_dates_aligned)

    if reference_dates is not None:
        print("Common observations:", len(reference_dates))
        print("Common first date:", reference_dates.min().date())
        print("Common last date:", reference_dates.max().date())

    print(f"\nReport saved: {SUMMARY_PATH}")

    if failed > 0 or not all_dates_aligned:
        raise SystemExit(
            "MULTI-STOCK VALIDATION FAILED."
        )

    print("\nMULTI-STOCK VALIDATION PASSED.")


if __name__ == "__main__":
    main()
