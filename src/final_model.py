import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ==========================================
# 1. LOAD DATA
# ==========================================

DATA_PATH = "outputs/results/cleaned_household_data.csv"

data = pd.read_csv(DATA_PATH)

print("\nDATA LOADED SUCCESSFULLY")
print("Dataset shape:", data.shape)

# ==========================================
# 2. DEFINE FEATURES AND TARGETS
# ==========================================

features = [
    "TOT_INCOME",
    "DISTRICT",
    "AGE_GROUP",
    "OCCUPATION_GROUP",
    "EDU_GROUP",
    "GENDER_GROUP",
    "SIZE_GROUP"
]

targets = [
    "M_EXP_TRANSPORT",
    "M_EXP_FOOD"
]

X = data[features]

numeric_features = ["TOT_INCOME"]

categorical_features = [
    "DISTRICT",
    "AGE_GROUP",
    "OCCUPATION_GROUP",
    "EDU_GROUP",
    "GENDER_GROUP",
    "SIZE_GROUP"
]

# ==========================================
# 3. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# ==========================================
# 4. TRAINING FUNCTION
# ==========================================

os.makedirs("outputs/models", exist_ok=True)

def train_final_model(target):

    print("\n" + "=" * 55)
    print("TRAINING MODEL:", target)
    print("=" * 55)

    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Select model based on Stage 7 results
    if target == "M_EXP_TRANSPORT":

        model = LinearRegression()

        model_name = "Linear Regression"

    else:

        try:
            model = BaggingRegressor(
                estimator=DecisionTreeRegressor(
                    random_state=42
                ),
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

        except TypeError:
            model = BaggingRegressor(
                base_estimator=DecisionTreeRegressor(
                    random_state=42
                ),
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

        model_name = "Bagging Regressor"

    # Complete pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Evaluate
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    print("\nModel:", model_name)
    print("R2 Score:", round(r2, 4))
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))

    # Save model
    filename = f"outputs/models/{target}_model.joblib"

    joblib.dump(pipeline, filename)

    print("\nMODEL SAVED:", filename)

    return {
        "Target": target,
        "Model": model_name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    }


# ==========================================
# 5. TRAIN BOTH MODELS
# ==========================================

results = []

for target in targets:

    result = train_final_model(target)

    results.append(result)


# ==========================================
# 6. SAVE PERFORMANCE REPORT
# ==========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/results/final_model_performance.csv",
    index=False
)

print("\nFINAL MODEL PERFORMANCE")
print(results_df)

print("\n" + "=" * 55)
print("FINAL MODEL TRAINING COMPLETED")
print("=" * 55)