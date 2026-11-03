# ==========================================
# STAGE 3.1: STATISTICAL ANALYSIS
# ==========================================

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

print("\n===== STATISTICAL ANALYSIS =====")

# Load cleaned dataset

df = pd.read_csv(
    "outputs/results/cleaned_household_data.csv"
)

# STEP 1: Descriptive statistics

print("\n--- DESCRIPTIVE STATISTICS ---")

print(
    df[
        ["TOT_INCOME", "M_EXP_FOOD", "M_EXP_TRANSPORT"]
    ].describe()
)

# STEP 2: Correlation matrix

print("\n--- CORRELATION MATRIX ---")

correlation = df[
    ["TOT_INCOME", "M_EXP_FOOD", "M_EXP_TRANSPORT"]
].corr()

print(correlation)

correlation.to_csv(
    "outputs/results/correlation_matrix.csv"
)

# STEP 3: Pearson correlation

print("\n--- PEARSON CORRELATION ---")

for target in ["M_EXP_FOOD", "M_EXP_TRANSPORT"]:

    r, p = pearsonr(
        df["TOT_INCOME"],
        df[target]
    )

    print("\nIncome vs", target)
    print("Pearson Correlation:", round(r, 4))
    print("P-value:", p)

# STEP 4: Spearman correlation

print("\n--- SPEARMAN CORRELATION ---")

for target in ["M_EXP_FOOD", "M_EXP_TRANSPORT"]:

    r, p = spearmanr(
        df["TOT_INCOME"],
        df[target]
    )

    print("\nIncome vs", target)
    print("Spearman Correlation:", round(r, 4))
    print("P-value:", p)

# STEP 5: Skewness

print("\n--- SKEWNESS ---")

print(
    df[
        ["TOT_INCOME", "M_EXP_FOOD", "M_EXP_TRANSPORT"]
    ].skew()
)

print("\n===== ANALYSIS COMPLETED =====")