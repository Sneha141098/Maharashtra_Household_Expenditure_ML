# ==========================================
# STAGE 2: DATA CLEANING AND PREPROCESSING
# ==========================================

import pandas as pd
import numpy as np
import os

print("\n===== STAGE 2: DATA CLEANING =====")

# STEP 1: Load merged dataset

df = pd.read_csv(
    "outputs/results/merged_household_data.csv"
)

print("\nOriginal Dataset Shape:", df.shape)

# STEP 2: Inspect data types

print("\n--- DATA TYPES ---")
print(df.dtypes)

# STEP 3: Check missing values

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# STEP 4: Remove duplicate household records

duplicate_count = df["HH_ID"].duplicated().sum()

print("\nDuplicate Household IDs:", duplicate_count)

df = df.drop_duplicates(
    subset="HH_ID",
    keep="first"
)

# STEP 5: Clean categorical variables

categorical_columns = [
    "STATE",
    "DISTRICT",
    "REGION_TYPE",
    "AGE_GROUP",
    "OCCUPATION_GROUP",
    "EDU_GROUP",
    "GENDER_GROUP",
    "SIZE_GROUP"
]

for col in categorical_columns:
    df[col] = df[col].astype("string").str.strip()

print("\nCategorical Variables Cleaned.")

# STEP 6: Convert numerical columns

numerical_columns = [
    "TOT_INCOME",
    "M_EXP_FOOD",
    "M_EXP_TRANSPORT"
]

for col in numerical_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

print("\nNumerical Variables Converted.")

# STEP 7: Check missing values again

print("\n--- MISSING VALUES AFTER CONVERSION ---")
print(df.isnull().sum())

# STEP 8: Check invalid negative values

print("\n--- NEGATIVE VALUES CHECK ---")

for col in numerical_columns:
    print(
        col,
        "Negative values:",
        (df[col] < 0).sum()
    )

# STEP 9: Check zero values

print("\n--- ZERO VALUE ANALYSIS ---")

for col in numerical_columns:
    print(
        col,
        "Zero values:",
        (df[col] == 0).sum()
    )

# STEP 10: Detect outliers using IQR

print("\n--- OUTLIER DETECTION ---")

for col in numerical_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ]

    print("\nVariable:", col)
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)
    print("Outliers:", len(outliers))

# STEP 11: Final dataset information

print("\n--- FINAL DATASET INFORMATION ---")

print("Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate IDs:", df["HH_ID"].duplicated().sum())

# STEP 12: Save cleaned dataset

os.makedirs("outputs/results", exist_ok=True)

df.to_csv(
    "outputs/results/cleaned_household_data.csv",
    index=False
)
# STEP 13: Generate data quality report

quality_report = pd.DataFrame({
    "Variable": numerical_columns,
    "Mean": [df[col].mean() for col in numerical_columns],
    "Median": [df[col].median() for col in numerical_columns],
    "Standard Deviation": [
        df[col].std() for col in numerical_columns
    ],
    "Minimum": [df[col].min() for col in numerical_columns],
    "Maximum": [df[col].max() for col in numerical_columns]
})

quality_report.to_csv(
    "outputs/results/data_quality_report.csv",
    index=False
)

print("\nData Quality Report Generated!")
print(quality_report)
print("\nCleaned dataset saved successfully!")

print("\n===== STAGE 2 COMPLETED =====")