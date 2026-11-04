
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "multi_stock_hac_lag_sensitivity.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_FILE = OUTPUT_DIR / "hac_robustness.png"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# LOAD AND VALIDATE DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "HAC_Max_Lags",
    "HAC_P_Value",
    "Annualized_Mean_Excess_Return",
    "Significant_At_5_Percent",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.sort_values("HAC_Max_Lags").reset_index(drop=True)

if df[required_columns].isna().any().any():
    raise ValueError("Missing values detected.")

if len(df) != 4:
    raise ValueError(f"Expected 4 HAC lag specifications, found {len(df)}")

if not np.all(df["HAC_P_Value"].between(0, 1)):
    raise ValueError("Invalid p-values detected.")

if not np.all(df["Significant_At_5_Percent"].astype(bool)):
    raise ValueError("At least one result is not significant at 5%.")

if not np.all(df["Annualized_Mean_Excess_Return"] < 0):
    raise ValueError("Expected negative excess returns across all lags.")


# ------------------------------------------------------------
# CREATE VISUAL
# ------------------------------------------------------------

lags = df["HAC_Max_Lags"].astype(str).tolist()
p_values = df["HAC_P_Value"].to_numpy()

fig, ax = plt.subplots(figsize=(11, 7))

bars = ax.bar(
    lags,
    p_values,
    width=0.55,
    label="Newey-West HAC p-value",
)

# 5% significance threshold
ax.axhline(
    0.05,
    linestyle="--",
    linewidth=1.5,
    label="5% significance threshold",
)

# Label each bar
for bar, value in zip(bars, p_values):
    ax.annotate(
        f"{value:.4f}",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=10,
    )

ax.set_title(
    "Statistical Robustness Across HAC Lag Specifications",
    fontsize=15,
    fontweight="bold",
    pad=18,
)

ax.set_xlabel("Newey-West Maximum Lag", fontsize=11)
ax.set_ylabel("HAC p-value", fontsize=11)

ax.set_ylim(0, 0.06)

ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

fig.text(
    0.5,
    0.015,
    "All specifications indicate statistically significant "
    "negative mean excess returns | 10-stock equal-weighted portfolio",
    ha="center",
    fontsize=9,
    color="dimgray",
)

plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("=" * 60)
print("HAC ROBUSTNESS VISUAL GENERATED")
print("=" * 60)

print(f"HAC specifications: {len(df)}")
print(f"All significant at 5%: {bool(df['Significant_At_5_Percent'].all())}")
print(
    "Annualized mean excess return:",
    f"{df['Annualized_Mean_Excess_Return'].iloc[0]:.4%}",
)

print(f"Saved to: {OUTPUT_FILE}")
