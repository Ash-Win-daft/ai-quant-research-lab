
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]

PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


def validate_market_data(file_path):
    """Validate a local OHLCV market-data CSV."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="raise"
    )

    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    if not df["Date"].is_monotonic_increasing:
        raise ValueError("Dates are not in chronological order.")

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Missing values detected.")

    if (df[PRICE_COLUMNS] <= 0).any().any():
        raise ValueError("Prices must be positive.")

    if (df["Volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")

    invalid_ohlc = (
        (df["High"] < df[["Open", "Close", "Low"]].max(axis=1))
        | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
    )

    if invalid_ohlc.any():
        raise ValueError("Invalid OHLC relationships detected.")

    df = df.set_index("Date")

    print(f"Validation passed: {path.name}")
    print(f"Observations: {len(df)}")
    print(f"Date range: {df.index.min().date()} to "
          f"{df.index.max().date()}")

    return df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    validate_market_data(
        project_root / "data" / "aapl_daily.csv"
    )

    validate_market_data(
        project_root / "data" / "aapl_daily_extended.csv"
    )
