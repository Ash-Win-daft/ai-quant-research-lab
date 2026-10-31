
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t


# --------------------------------------------------
# 1. Locate and load the saved daily returns
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_FILE = (
    REPORTS_DIR / "aapl_mean_reversion_significance_daily.csv"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Daily returns file not found: {INPUT_FILE}"
    )

returns = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

required_columns = ["Strategy", "Buy_Hold"]

for column in required_columns:
    if column not in returns.columns:
        raise ValueError(f"Missing required column: {column}")

returns = returns[required_columns].dropna()

if len(returns) < 30:
    raise ValueError("Too few observations for this analysis.")


# --------------------------------------------------
# 2. Calculate daily excess returns
# --------------------------------------------------

returns["Excess_Return"] = (
    returns["Strategy"] - returns["Buy_Hold"]
)

excess = returns["Excess_Return"]

n = len(excess)
mean_excess = excess.mean()


# --------------------------------------------------
# 3. Ordinary OLS benchmark
# --------------------------------------------------

# Testing whether the average excess return is zero
# is equivalent to testing the intercept in this model.
X = pd.DataFrame(
    {"Intercept": np.ones(n)},
    index=excess.index,
)

ols_model = sm.OLS(excess, X).fit()

ordinary_t = float(ols_model.tvalues["Intercept"])
ordinary_p = float(ols_model.pvalues["Intercept"])


# --------------------------------------------------
# 4. Newey-West HAC adjustment
# --------------------------------------------------

# Five lags is a simple initial choice for daily data.
# This is a modelling choice, not a universally optimal lag.
hac_lags = 5

hac_model = ols_model.get_robustcov_results(
    cov_type="HAC",
    maxlags=hac_lags,
    use_correction=True,
)

hac_t = float(hac_model.tvalues[0])
hac_p = float(hac_model.pvalues[0])
hac_se = float(hac_model.bse[0])

# Use the Student t critical value with n - 1 degrees of freedom
# for an approximate 95% interval.
critical_value = student_t.ppf(0.975, df=n - 1)

ci_low = mean_excess - critical_value * hac_se
ci_high = mean_excess + critical_value * hac_se


# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

print("\nNEWEY-WEST HAC SIGNIFICANCE TEST")
print("--------------------------------")
print("Observations:", n)
print("HAC lags:", hac_lags)
print("Mean daily excess return:", f"{mean_excess:.6%}")

print("\nOrdinary t-test")
print("---------------")
print("t-statistic:", round(ordinary_t, 4))
print("p-value:", round(ordinary_p, 6))

print("\nNewey-West HAC test")
print("-------------------")
print("HAC standard error:", f"{hac_se:.6%}")
print("HAC t-statistic:", round(hac_t, 4))
print("HAC p-value:", round(hac_p, 6))
print("Approximate 95% confidence interval:",
      f"[{ci_low:.6%}, {ci_high:.6%}]")

if hac_p < 0.05:
    print("\nResult: reject the zero-mean-excess-return null at 5%.")
else:
    print("\nResult: insufficient evidence to reject the null at 5%.")

print(
    "\nInterpretation: this test adjusts uncertainty for "
    "heteroskedasticity and autocorrelation up to the chosen lag structure. "
    "It does not eliminate parameter-selection bias or prove "
    "that the strategy will perform similarly in the future."
)


# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

summary = pd.DataFrame([{
    "Observations": n,
    "Mean_Daily_Excess_Return": mean_excess,
    "Ordinary_T_Statistic": ordinary_t,
    "Ordinary_P_Value": ordinary_p,
    "HAC_Lags": hac_lags,
    "HAC_Standard_Error": hac_se,
    "HAC_T_Statistic": hac_t,
    "HAC_P_Value": hac_p,
    "HAC_CI_95_Lower": ci_low,
    "HAC_CI_95_Upper": ci_high,
}])

output_file = (
    REPORTS_DIR / "aapl_mean_reversion_hac_significance.csv"
)

summary.to_csv(output_file, index=False)

print("\nSaved:", output_file)
