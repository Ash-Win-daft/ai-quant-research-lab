
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Locate the project and results files
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

FILES = {
    "oos_comparison": (
        REPORTS_DIR / "aapl_mean_reversion_oos_comparison.csv"
    ),
    "cost_stress": (
        REPORTS_DIR / "aapl_mean_reversion_transaction_cost_stress.csv"
    ),
    "hac_significance": (
        REPORTS_DIR / "aapl_mean_reversion_hac_significance.csv"
    ),
}


# --------------------------------------------------
# 2. Validate that the required results exist
# --------------------------------------------------

for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {name} results: {path}"
        )


# --------------------------------------------------
# 3. Load the results
# --------------------------------------------------

oos = pd.read_csv(FILES["oos_comparison"], index_col=0)
costs = pd.read_csv(FILES["cost_stress"])
hac = pd.read_csv(FILES["hac_significance"])

required_metrics = [
    "Total Return",
    "Sharpe Ratio (rf=0)",
    "Maximum Drawdown",
]

for metric in required_metrics:
    if metric not in oos.index:
        raise ValueError(f"Missing out-of-sample metric: {metric}")

required_cost_columns = [
    "Transaction_Cost_Per_Turnover",
    "Strategy_Total_Return",
    "Buy_Hold_Total_Return",
    "Strategy_Sharpe",
    "Buy_Hold_Sharpe",
]

for column in required_cost_columns:
    if column not in costs.columns:
        raise ValueError(f"Missing cost-test column: {column}")

if "HAC_P_Value" not in hac.columns:
    raise ValueError("HAC results do not contain HAC_P_Value.")


# --------------------------------------------------
# 4. Extract the main out-of-sample results
# --------------------------------------------------

strategy_return = float(oos.loc["Total Return", "Mean Reversion"])
benchmark_return = float(oos.loc["Total Return", "Buy and Hold"])

strategy_sharpe = float(
    oos.loc["Sharpe Ratio (rf=0)", "Mean Reversion"]
)
benchmark_sharpe = float(
    oos.loc["Sharpe Ratio (rf=0)", "Buy and Hold"]
)

strategy_drawdown = float(
    oos.loc["Maximum Drawdown", "Mean Reversion"]
)
benchmark_drawdown = float(
    oos.loc["Maximum Drawdown", "Buy and Hold"]
)

hac_p_value = float(hac.iloc[0]["HAC_P_Value"])


# --------------------------------------------------
# 5. Evaluate the transaction-cost sensitivity
# --------------------------------------------------

# Use the 0.10% scenario as the reference cost.
reference_cost = 0.001

reference_rows = costs[
    (costs["Transaction_Cost_Per_Turnover"] - reference_cost).abs()
    < 1e-10
]

if reference_rows.empty:
    raise ValueError(
        "The cost stress test has no 0.10% reference-cost scenario."
    )

reference = reference_rows.iloc[0]

reference_return = float(reference["Strategy_Total_Return"])
zero_cost_rows = costs[
    costs["Transaction_Cost_Per_Turnover"].abs() < 1e-10
]

if zero_cost_rows.empty:
    raise ValueError("The cost stress test has no zero-cost scenario.")

zero_cost_return = float(
    zero_cost_rows.iloc[0]["Strategy_Total_Return"]
)

high_cost_rows = costs[
    (costs["Transaction_Cost_Per_Turnover"] - 0.005).abs() < 1e-10
]

if high_cost_rows.empty:
    raise ValueError("The cost stress test has no 0.50% scenario.")

high_cost_return = float(
    high_cost_rows.iloc[0]["Strategy_Total_Return"]
)

cost_sensitive = (
    zero_cost_return > 0
    and high_cost_return < zero_cost_return * 0.25
)


# --------------------------------------------------
# 6. Apply transparent decision rules
# --------------------------------------------------

findings = []

# Rule A: compare total return.
if strategy_return < benchmark_return:
    findings.append(
        "REJECT: out-of-sample total return was below buy and hold."
    )
else:
    findings.append(
        "PASS: out-of-sample total return was at least as high as buy and hold."
    )

# Rule B: compare Sharpe ratio.
if strategy_sharpe < benchmark_sharpe:
    findings.append(
        "REVIEW: out-of-sample Sharpe ratio was below buy and hold."
    )
else:
    findings.append(
        "PASS: out-of-sample Sharpe ratio was at least as high as buy and hold."
    )

# Rule C: check statistical evidence.
if hac_p_value >= 0.05:
    findings.append(
        "REVIEW: HAC test did not find statistically significant "
        "evidence of a nonzero mean daily excess return."
    )
else:
    findings.append(
        "PASS: HAC test found a statistically significant "
        "mean daily excess return at the 5% level; further validation is needed."
    )

# Rule D: check transaction-cost sensitivity.
if cost_sensitive:
    findings.append(
        "REVIEW: returns deteriorated sharply under higher transaction costs."
    )
else:
    findings.append(
        "PASS: the defined severe cost-sensitivity flag was not triggered."
    )


# --------------------------------------------------
# 7. Assign the overall research status
# --------------------------------------------------

# This is a screening rule, not a universal trading standard.
if strategy_return < benchmark_return and strategy_sharpe < benchmark_sharpe:
    status = "REJECT"
    explanation = (
        "The candidate underperformed the benchmark on both total return "
        "and Sharpe ratio in this test. Do not advance it as a superior "
        "replacement for buy and hold without a new, justified hypothesis."
    )
elif hac_p_value >= 0.05 or cost_sensitive:
    status = "REVIEW"
    explanation = (
        "The current evidence is inconclusive or sensitive to costs. "
        "Keep the experiment in the research archive, but do not claim "
        "a demonstrated advantage."
    )
else:
    status = "PASS"
    explanation = (
        "The candidate passed these preliminary screening rules. "
        "This is not proof of a robust or deployable trading edge."
    )


# --------------------------------------------------
# 8. Print the decision
# --------------------------------------------------

print("\nAI QUANT RESEARCH LAB")
print("STRATEGY EVALUATION")
print("===================")

print("\nCandidate: AAPL mean reversion")
print("Evaluation period: 2024-2025")
print("Reference transaction cost: 0.10% per unit of turnover")

print("\nPERFORMANCE")
print("-----------")
print(f"Strategy total return: {strategy_return:.2%}")
print(f"Buy-and-hold total return: {benchmark_return:.2%}")
print(f"Strategy Sharpe ratio: {strategy_sharpe:.3f}")
print(f"Buy-and-hold Sharpe ratio: {benchmark_sharpe:.3f}")
print(f"Strategy maximum drawdown: {strategy_drawdown:.2%}")
print(f"Buy-and-hold maximum drawdown: {benchmark_drawdown:.2%}")

print("\nROBUSTNESS")
print("----------")
print(f"Return at zero cost: {zero_cost_return:.2%}")
print(f"Return at 0.10% cost: {reference_return:.2%}")
print(f"Return at 0.50% cost: {high_cost_return:.2%}")
print(f"HAC p-value: {hac_p_value:.4f}")

print("\nFINDINGS")
print("--------")
for finding in findings:
    print("-", finding)

print("\nOVERALL STATUS:", status)
print(explanation)


# --------------------------------------------------
# 9. Save the evaluation record
# --------------------------------------------------

evaluation = pd.DataFrame([{
    "Strategy": "AAPL Mean Reversion",
    "Evaluation_Period": "2024-2025",
    "Reference_Cost": reference_cost,
    "Strategy_Total_Return": strategy_return,
    "Buy_Hold_Total_Return": benchmark_return,
    "Strategy_Sharpe": strategy_sharpe,
    "Buy_Hold_Sharpe": benchmark_sharpe,
    "Strategy_Max_Drawdown": strategy_drawdown,
    "Buy_Hold_Max_Drawdown": benchmark_drawdown,
    "HAC_P_Value": hac_p_value,
    "Zero_Cost_Return": zero_cost_return,
    "Reference_Cost_Return": reference_return,
    "High_Cost_Return": high_cost_return,
    "Cost_Sensitive_Flag": cost_sensitive,
    "Overall_Status": status,
    "Decision_Explanation": explanation,
}])

output_path = REPORTS_DIR / "aapl_mean_reversion_evaluation.csv"
evaluation.to_csv(output_path, index=False)

print("\nEvaluation saved to:", output_path)
