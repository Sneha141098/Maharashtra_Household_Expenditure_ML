# ==========================================
# STAGE 4: FEATURE ENGINEERING
# AND ML DATA PREPARATION
# ==========================================

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

print("\n===== STAGE 4: ML PREPARATION =====")

# STEP 1: Load cleaned dataset

df = pd.read_csv(
    "outputs/results/cleaned_household_data.csv"
)

# STEP 2: Select independent variables

features = [
    "TOT_INCOME",
    "DISTRICT",
    "AGE_GROUP",
    "OCCUPATION_GROUP",
    "EDU_GROUP",
    "GENDER_GROUP",
    "SIZE_GROUP"
]

X = df[features]

# STEP 3: Define target variables

y_transport = df["M_EXP_TRANSPORT"]

y_food = df["M_EXP_FOOD"]

print("\nFEATURE VARIABLES:")
print(X.columns.tolist())

print("\nTRANSPORT TARGET:")
print(y_transport.name)

print("\nFOOD TARGET:")
print(y_food.name)

# STEP 4: Identify numerical and categorical features

numerical_features = [
    "TOT_INCOME"
]

categorical_features = [
    "DISTRICT",
    "AGE_GROUP",
    "OCCUPATION_GROUP",
    "EDU_GROUP",
    "GENDER_GROUP",
    "SIZE_GROUP"
]

print("\nNumerical Features:", numerical_features)

print("\nCategorical Features:", categorical_features)

# STEP 5: Split transportation dataset

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X,
    y_transport,
    test_size=0.20,
    random_state=42
)

# STEP 6: Split food dataset

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X,
    y_food,
    test_size=0.20,
    random_state=42
)

# STEP 7: Display dataset sizes

print("\n--- TRANSPORT DATASET ---")

print("Training observations:", len(X_train_t))
print("Testing observations:", len(X_test_t))

print("\n--- FOOD DATASET ---")

print("Training observations:", len(X_train_f))
print("Testing observations:", len(X_test_f))

# STEP 8: Save feature information

os.makedirs("outputs/results", exist_ok=True)

feature_report = pd.DataFrame({
    "Feature": features,
    "Type": [
        "Numerical",
        "Categorical",
        "Categorical",
        "Categorical",
        "Categorical",
        "Categorical",
        "Categorical"
    ]
})

feature_report.to_csv(
    "outputs/results/feature_report.csv",
    index=False
)

print("\nFeature report saved successfully!")

print("\n===== STAGE 4 COMPLETED =====")