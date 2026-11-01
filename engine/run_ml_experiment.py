
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aapl_adjusted_close.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

TEST_START = pd.Timestamp("2024-01-01")
TRANSACTION_COST = 0.001


FEATURES = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "volatility_10d",
    "volatility_20d",
    "price_vs_sma20",
    "sma20_vs_sma50",
]


# ============================================================
# DATA PREPARATION
# ============================================================

def load_and_prepare_data():
    """Load adjusted prices and construct features and targets."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Adjusted-price snapshot not found: {DATA_PATH}"
        )

    prices_df = pd.read_csv(
        DATA_PATH,
        parse_dates=["Date"],
        index_col="Date",
    )

    if "Adj_Close" not in prices_df.columns:
        raise ValueError("Adj_Close column is missing.")

    prices = prices_df["Adj_Close"].sort_index()

    if prices.index.has_duplicates:
        raise ValueError("Duplicate dates detected.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Dates must be chronological.")

    if prices.isna().any() or (prices <= 0).any():
        raise ValueError("Invalid adjusted prices detected.")

    df = pd.DataFrame(index=prices.index)

    daily_return = prices.pct_change(fill_method=None)

    # Historical features available at the current close.
    df["return_1d"] = daily_return
    df["return_5d"] = prices.pct_change(5, fill_method=None)
    df["return_10d"] = prices.pct_change(10, fill_method=None)
    df["return_20d"] = prices.pct_change(20, fill_method=None)

    df["volatility_10d"] = daily_return.rolling(10).std()
    df["volatility_20d"] = daily_return.rolling(20).std()

    sma_20 = prices.rolling(20).mean()
    sma_50 = prices.rolling(50).mean()

    df["price_vs_sma20"] = prices / sma_20 - 1
    df["sma20_vs_sma50"] = sma_20 / sma_50 - 1

    # Target is the following trading session's return.
    df["next_return"] = daily_return.shift(-1)

    # Record the date on which each target becomes observable.
    df["target_date"] = pd.Series(
        prices.index,
        index=prices.index,
    ).shift(-1)

    df["target"] = (df["next_return"] > 0).astype(float)
    df.loc[df["next_return"].isna(), "target"] = np.nan

    df = df.dropna(
        subset=FEATURES + [
            "next_return",
            "target",
            "target_date",
        ]
    ).copy()

    df["target"] = df["target"].astype(int)

    return prices, df


# ============================================================
# MODEL
# ============================================================

def create_model():
    """Create the logistic regression pipeline."""

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000)),
    ])


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_performance(returns):
    """Calculate portfolio performance from daily returns."""

    returns = returns.dropna()

    if returns.empty:
        raise ValueError("Cannot calculate performance on empty returns.")

    equity = (1 + returns).cumprod()

    total_return = equity.iloc[-1] - 1
    years = len(returns) / 252

    cagr = (
        equity.iloc[-1] ** (1 / years) - 1
        if years > 0 else np.nan
    )

    annual_volatility = (
        returns.std(ddof=1) * np.sqrt(252)
    )

    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(252)
        if returns.std(ddof=1) > 0 else np.nan
    )

    drawdown = equity / equity.cummax() - 1

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": annual_volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": drawdown.min(),
    }


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment():
    print("Loading adjusted-price data...", flush=True)

    prices, df = load_and_prepare_data()

    # Test observations are selected by feature date.
    test_df = df.loc[df.index >= TEST_START].copy()

    # Training labels must have become observable before
    # the test period begins. This excludes the final
    # pre-test row whose next-session return is in 2024.
    train_df = df.loc[
        (df.index < TEST_START)
        & (df["target_date"] < TEST_START)
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Training or test dataset is empty.")

    if train_df["target_date"].max() >= TEST_START:
        raise ValueError("Training labels cross the test boundary.")

    X_train = train_df[FEATURES]
    y_train = train_df["target"]

    X_test = test_df[FEATURES]
    y_test = test_df["target"]

    print(f"Training observations: {len(train_df)}")
    print(f"Test observations: {len(test_df)}")
    print(
        "Training target dates end:",
        train_df["target_date"].max().date(),
    )
    print(
        "Test feature dates:",
        test_df.index.min().date(),
        "to",
        test_df.index.max().date(),
    )

    model = create_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test).astype(int)
    probabilities = model.predict_proba(X_test)[:, 1]

    # ========================================================
    # PREDICTION EVALUATION
    # ========================================================

    train_up_rate = y_train.mean()
    majority_prediction = int(train_up_rate >= 0.5)

    actual_up_rate = y_test.mean()

    evaluation = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Balanced Accuracy",
            "Precision",
            "Recall",
            "ROC AUC",
            "Brier Score",
            "Training Majority Baseline Accuracy",
            "Always-Up Baseline Accuracy",
            "Historical-Rate Baseline Brier Score",
        ],
        "Value": [
            accuracy_score(y_test, predictions),
            balanced_accuracy_score(y_test, predictions),
            precision_score(
                y_test, predictions, zero_division=0
            ),
            recall_score(
                y_test, predictions, zero_division=0
            ),
            roc_auc_score(y_test, probabilities),
            brier_score_loss(y_test, probabilities),
            accuracy_score(
                y_test,
                np.full(len(y_test), majority_prediction),
            ),
            accuracy_score(
                y_test,
                np.ones(len(y_test), dtype=int),
            ),
            brier_score_loss(
                y_test,
                np.full(len(y_test), train_up_rate),
            ),
        ],
    })

    print("\nPREDICTION EVALUATION")
    print(evaluation.to_string(index=False))

    # ========================================================
    # STRATEGY PERFORMANCE
    # ========================================================

    strategy = pd.DataFrame(index=test_df.index)

    strategy["Prediction"] = predictions
    strategy["Probability_Up"] = probabilities
    strategy["Actual_Target"] = y_test
    strategy["Actual_Return"] = test_df["next_return"]

    strategy["Position"] = strategy["Prediction"].astype(float)

    strategy["Turnover"] = (
        strategy["Position"].diff().abs()
    )

    # Charge for entering the initial position.
    strategy.iloc[
        0, strategy.columns.get_loc("Turnover")
    ] = abs(strategy["Position"].iloc[0])

    strategy["Strategy_Return_Net"] = (
        strategy["Position"] * strategy["Actual_Return"]
        - strategy["Turnover"] * TRANSACTION_COST
    )

    strategy["Buy_Hold_Return"] = strategy["Actual_Return"]

    performance = pd.DataFrame({
        "ML Strategy": calculate_performance(
            strategy["Strategy_Return_Net"]
        ),
        "Buy and Hold": calculate_performance(
            strategy["Buy_Hold_Return"]
        ),
    })

    print("\nSTRATEGY PERFORMANCE")
    print(performance.round(4))

    # ========================================================
    # COEFFICIENTS
    # ========================================================

    coefficients = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": model.named_steps["model"].coef_[0],
    })

    coefficients["Absolute_Coefficient"] = (
        coefficients["Coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        "Absolute_Coefficient",
        ascending=False,
    )

    # ========================================================
    # YEARLY PERFORMANCE
    # ========================================================

    yearly_rows = []

    for year, group in strategy.groupby(strategy.index.year):
        ml_metrics = calculate_performance(
            group["Strategy_Return_Net"]
        )

        bh_metrics = calculate_performance(
            group["Buy_Hold_Return"]
        )

        yearly_rows.append({
            "Year": year,
            "ML_Total_Return": ml_metrics["Total Return"],
            "Buy_Hold_Total_Return": bh_metrics["Total Return"],
            "ML_Sharpe": ml_metrics["Sharpe Ratio"],
            "Buy_Hold_Sharpe": bh_metrics["Sharpe Ratio"],
        })

    yearly_performance = pd.DataFrame(yearly_rows)

    # ========================================================
    # EQUITY CURVES
    # ========================================================

    strategy["ML_Equity"] = (
        1 + strategy["Strategy_Return_Net"]
    ).cumprod()

    strategy["Buy_Hold_Equity"] = (
        1 + strategy["Buy_Hold_Return"]
    ).cumprod()

    # ========================================================
    # EXPORT RESULTS
    # ========================================================

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    strategy.to_csv(
        REPORTS_DIR / "ml_daily_strategy_returns.csv"
    )

    strategy[
        ["ML_Equity", "Buy_Hold_Equity"]
    ].to_csv(
        REPORTS_DIR / "ml_equity_curves.csv"
    )

    evaluation.to_csv(
        REPORTS_DIR / "ml_evaluation.csv",
        index=False,
    )

    # The existing pipeline expects this filename.
    evaluation.to_csv(
        REPORTS_DIR / "ml_prediction_metrics.csv",
        index=False,
    )

    performance.to_csv(
        REPORTS_DIR / "ml_performance.csv"
    )

    coefficients.to_csv(
        REPORTS_DIR / "ml_coefficients.csv",
        index=False,
    )

    yearly_performance.to_csv(
        REPORTS_DIR / "ml_yearly_performance.csv",
        index=False,
    )

    print("\nML EXPERIMENT COMPLETED")
    print("Results saved in:", REPORTS_DIR)


if __name__ == "__main__":
    run_experiment()
