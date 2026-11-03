# ==========================================
# PROJECT: HOUSEHOLD EXPENDITURE ANALYSIS
# STAGE 1: DATA LOADING AND INTEGRATION
# ==========================================

import pandas as pd
import numpy as np
import os

print("\n===== HOUSEHOLD EXPENDITURE ANALYSIS =====")

# STEP 1: Load datasets

income = pd.read_excel("data/Household Income.xlsx")

expenditure = pd.read_excel(
    "data/Household Expenditure_Maharastra.xlsx"
)

print("\n1. DATASETS LOADED SUCCESSFULLY")

# STEP 2: Check dimensions

print("\nIncome Dataset Shape:", income.shape)
print("Expenditure Dataset Shape:", expenditure.shape)

# STEP 3: Merge datasets using HH_ID

df = pd.merge(
    expenditure,
    income[["HH_ID", "TOT_INCOME"]],
    on="HH_ID",
    how="inner"
)

print("\n2. DATASETS MERGED SUCCESSFULLY")

# STEP 4: Display dataset information

print("\nMerged Dataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

# STEP 5: Display first five rows

print("\nFirst Five Observations:")
print(df.head())

# STEP 6: Check missing values

print("\nMissing Values:")
print(df.isnull().sum())

# STEP 7: Check duplicate household IDs

print("\nDuplicate Household IDs:")
print(df["HH_ID"].duplicated().sum())

# STEP 8: Statistical summary

print("\nStatistical Summary:")
print(df.describe())

# STEP 9: Save merged dataset

os.makedirs("outputs/results", exist_ok=True)

df.to_csv(
    "outputs/results/merged_household_data.csv",
    index=False
)

print("\n3. MERGED DATASET SAVED SUCCESSFULLY")

print("\n===== STAGE 1 COMPLETED =====")