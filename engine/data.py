
import pandas as pd


def validate_market_data(data):
    """
    Validate the basic structure and quality of market data.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Market data must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("Market data is empty.")

    if "Close" not in data.columns:
        raise ValueError("Market data must contain a 'Close' column.")

    if data["Close"].isna().any():
        raise ValueError("Close prices contain missing values.")

    if (data["Close"] <= 0).any():
        raise ValueError("Close prices must be positive.")

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Market data must use dates as its index.")

    if data.index.has_duplicates:
        raise ValueError("Market data contains duplicate dates.")

    if not data.index.is_monotonic_increasing:
        raise ValueError("Market data dates must be in ascending order.")

    return True
