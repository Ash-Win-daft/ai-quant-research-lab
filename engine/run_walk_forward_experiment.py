# ========================================================
# AI QUANT RESEARCH LAB
# Walk-Forward Machine-Learning Experiment
# ========================================================

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


# ========================================================
# CONFIGURATION
# ========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "aapl_adjusted_close.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

TEST_START = "2024-01-01"

TRANSACTION_COST = 0.001  # 0.10% per position change

ANNUALIZATION_FACTOR = 252

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

# ========================================================
# 1. LOAD DATA AND PREPARE FEATURES
# ========================================================

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

    # Historical momentum features
    df["return_1d"] = daily_return
    df["return_5d"] = prices.pct_change(5, fill_method=None)
    df["return_10d"] = prices.pct_change(10, fill_method=None)
    df["return_20d"] = prices.pct_change(20, fill_method=None)

    # Historical volatility features
    df["volatility_10d"] = daily_return.rolling(10).std()
    df["volatility_20d"] = daily_return.rolling(20).std()

    # Moving-average features
    sma_20 = prices.rolling(20).mean()
    sma_50 = prices.rolling(50).mean()

    df["price_vs_sma20"] = prices / sma_20 - 1
    df["sma20_vs_sma50"] = sma_20 / sma_50 - 1

    # Next-session target
    df["next_return"] = daily_return.shift(-1)

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

# ========================================================
# 2. CREATE MACHINE-LEARNING MODEL
# ========================================================

def create_model():
    """Create the logistic regression pipeline."""

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),

        ("scaler", StandardScaler()),

        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight=None,
        )),
    ])

# ========================================================
# 3. WALK-FORWARD TRAINING AND PREDICTION
# ========================================================

def run_walk_forward(prices, df):
    """Generate predictions using expanding-window training."""

    test_dates = df.index[df.index >= TEST_START]

    predictions = []
    probabilities = []
    actual_returns = []
    actual_targets = []
    training_sizes = []

    for i, test_date in enumerate(test_dates):

        # Historical observations whose labels are already known.
        train_df = df.loc[
            (df.index < test_date)
            & (df["target_date"] <= test_date)
        ].copy()

        if train_df.empty:
            raise ValueError(
                f"No training observations available for {test_date}."
            )

        X_train = train_df[FEATURES]
        y_train = train_df["target"]

        X_test = df.loc[[test_date], FEATURES]

        model = create_model()
        model.fit(X_train, y_train)

        prediction = int(model.predict(X_test)[0])

        probability = float(
            model.predict_proba(X_test)[0, 1]
        )

        predictions.append(prediction)
        probabilities.append(probability)

        actual_returns.append(
            df.loc[test_date, "next_return"]
        )

        actual_targets.append(
            df.loc[test_date, "target"]
        )

        training_sizes.append(len(train_df))

        if (i + 1) % 50 == 0:
            print(
                f"Processed {i + 1}/{len(test_dates)} "
                f"test observations.",
                flush=True,
            )

    results = pd.DataFrame(
        {
            "Prediction": predictions,
            "Probability_Up": probabilities,
            "Actual_Target": actual_targets,
            "Actual_Return": actual_returns,
            "Training_Size": training_sizes,
        },
        index=test_dates,
    )

    return results

# ========================================================
# 4. EVALUATE WALK-FORWARD PREDICTIONS
# ========================================================

def evaluate_predictions(results, df):
    """Evaluate predictions against historical baselines."""

    actual = results["Actual_Target"]
    predicted = results["Prediction"]
    probabilities = results["Probability_Up"]

    # Expanding historical majority-class baseline.
    majority_predictions = []

    for test_date in results.index:

        train_df = df.loc[
            (df.index < test_date)
            & (df["target_date"] <= test_date)
        ]

        majority_predictions.append(
            int(train_df["target"].mean() >= 0.5)
        )

    majority_predictions = np.array(majority_predictions)

    # Historical up-rate probability baseline.
    historical_up_rates = []

    for test_date in results.index:

        train_df = df.loc[
            (df.index < test_date)
            & (df["target_date"] <= test_date)
        ]

        historical_up_rates.append(
            train_df["target"].mean()
        )

    historical_up_rates = np.array(historical_up_rates)

    evaluation = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Balanced Accuracy",
            "Precision",
            "Recall",
            "ROC AUC",
            "Brier Score",
            "Expanding Majority Baseline Accuracy",
            "Historical-Rate Baseline Brier Score",
        ],
        "Value": [
            accuracy_score(actual, predicted),
            balanced_accuracy_score(actual, predicted),
            precision_score(actual, predicted, zero_division=0),
            recall_score(actual, predicted, zero_division=0),
            roc_auc_score(actual, probabilities),
            brier_score_loss(actual, probabilities),
            accuracy_score(actual, majority_predictions),
            brier_score_loss(actual, historical_up_rates),
        ],
    })

    return evaluation

# ========================================================
# 5. EVALUATE WALK-FORWARD TRADING STRATEGY
# ========================================================

def evaluate_strategy(results):
    """Calculate net strategy returns and benchmark performance."""

    strategy = results.copy()

    # Each prediction determines the position for the next session.
    strategy["Position"] = strategy["Prediction"].astype(float)

    # Calculate entries and exits.
    strategy["Turnover"] = (
        strategy["Position"].diff().abs()
    )

    # First test-period position starts from cash.
    strategy.iloc[
        0, strategy.columns.get_loc("Turnover")
    ] = abs(strategy["Position"].iloc[0])

    # Deduct transaction costs.
    strategy["Strategy_Return_Net"] = (
        strategy["Position"] * strategy["Actual_Return"]
        - strategy["Turnover"] * TRANSACTION_COST
    )

    # Buy-and-hold benchmark.
    strategy["Buy_Hold_Return"] = strategy["Actual_Return"]

    return strategy


# ========================================================
# 6. CALCULATE PERFORMANCE METRICS
# ========================================================

def calculate_performance(returns):
    """Calculate standard trading performance metrics."""

    returns = returns.dropna()

    equity = (1 + returns).cumprod()

    total_return = equity.iloc[-1] - 1

    years = len(returns) / ANNUALIZATION_FACTOR

    cagr = (
        equity.iloc[-1] ** (1 / years) - 1
        if years > 0 and equity.iloc[-1] > 0
        else np.nan
    )

    volatility = (
        returns.std(ddof=1)
        * np.sqrt(ANNUALIZATION_FACTOR)
    )

    sharpe = (
        returns.mean()
        / returns.std(ddof=1)
        * np.sqrt(ANNUALIZATION_FACTOR)
        if returns.std(ddof=1) > 0
        else np.nan
    )

    max_drawdown = (
        equity / equity.cummax() - 1
    ).min()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
    }


def summarize_performance(strategy):
    """Compare the ML strategy against buy-and-hold."""

    performance = pd.DataFrame({
        "Walk-Forward ML": calculate_performance(
            strategy["Strategy_Return_Net"]
        ),
        "Buy and Hold": calculate_performance(
            strategy["Buy_Hold_Return"]
        ),
    })

    return performance

# ========================================================
# 7. MAIN EXPERIMENT
# ========================================================

def main():

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading historical data...")

    prices, df = load_and_prepare_data()

    print(f"Total observations: {len(df)}")

    print("\nRunning walk-forward experiment...")

    results = run_walk_forward(prices, df)

    print("\nEvaluating predictions...")

    evaluation = evaluate_predictions(results, df)

    print("\nEvaluating trading strategy...")

    strategy = evaluate_strategy(results)

    performance = summarize_performance(strategy)

    # ----------------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------------

    results.to_csv(
        REPORTS_DIR / "walk_forward_predictions.csv"
    )

    evaluation.to_csv(
        REPORTS_DIR / "walk_forward_metrics.csv",
        index=False
    )

    performance.to_csv(
        REPORTS_DIR / "walk_forward_performance.csv"
    )

    strategy.to_csv(
        REPORTS_DIR / "walk_forward_daily_strategy_returns.csv"
    )

    equity_curves = pd.DataFrame({
        "Walk_Forward_ML": (
            1 + strategy["Strategy_Return_Net"]
        ).cumprod(),
        "Buy_and_Hold": (
            1 + strategy["Buy_Hold_Return"]
        ).cumprod(),
    })

    equity_curves.to_csv(
        REPORTS_DIR / "walk_forward_equity_curves.csv"
    )

    # ----------------------------------------------------
    # DISPLAY RESULTS
    # ----------------------------------------------------

    print("\nPREDICTION EVALUATION")
    print(evaluation.to_string(index=False))

    print("\nSTRATEGY PERFORMANCE")
    print(performance)

    print("\nRESULTS SAVED TO:")
    print(REPORTS_DIR)

    print("\nWalk-forward experiment completed.")


# ========================================================
# 8. SCRIPT ENTRY POINT
# ========================================================

if __name__ == "__main__":
    main()