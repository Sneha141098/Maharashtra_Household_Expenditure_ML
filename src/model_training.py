import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    BaggingRegressor
)
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("outputs/results/cleaned_household_data.csv")

print("\n===== STAGE 5: MODEL TRAINING =====")

features = [
    "TOT_INCOME",
    "DISTRICT",
    "AGE_GROUP",
    "OCCUPATION_GROUP",
    "EDU_GROUP",
    "GENDER_GROUP",
    "SIZE_GROUP"
]

numerical_features = ["TOT_INCOME"]

categorical_features = [
    "DISTRICT",
    "AGE_GROUP",
    "OCCUPATION_GROUP",
    "EDU_GROUP",
    "GENDER_GROUP",
    "SIZE_GROUP"
]

X = df[features]

# Same train-test split for both targets
X_train, X_test = train_test_split(
    X,
    test_size=0.20,
    random_state=42
)

# Recover matching target rows using the split indices
y_transport = df.loc[X_train.index, "M_EXP_TRANSPORT"]
y_transport_test = df.loc[X_test.index, "M_EXP_TRANSPORT"]

y_food = df.loc[X_train.index, "M_EXP_FOOD"]
y_food_test = df.loc[X_test.index, "M_EXP_FOOD"]


# --------------------------------------------------
# 2. DATA PREPROCESSING
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# --------------------------------------------------
# 3. DEFINE SIX ML MODELS
# --------------------------------------------------

models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    # Bootstrap Aggregation
    "Bagging Regressor": BaggingRegressor(
        estimator=DecisionTreeRegressor(random_state=42),
        n_estimators=100,
        max_samples=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    ),

    # Support Vector Machine Regression
    "Support Vector Regression": SVR(
        kernel="rbf",
        C=10,
        epsilon=0.1
    )
}


# --------------------------------------------------
# 4. MODEL TRAINING AND EVALUATION
# --------------------------------------------------

def train_and_evaluate(X_train, X_test, y_train, y_test, target_name):

    results = []

    print(f"\n\n========== {target_name} PREDICTION ==========")

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        mse = mean_squared_error(y_test, predictions)

        rmse = np.sqrt(mse)

        r2 = r2_score(y_test, predictions)

        results.append({
            "Target": target_name,
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2
        })

        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.4f}")

    return results


# --------------------------------------------------
# 5. TRAIN FOR TRANSPORT EXPENDITURE
# --------------------------------------------------

transport_results = train_and_evaluate(
    X_train,
    X_test,
    y_transport,
    y_transport_test,
    "Transportation"
)


# --------------------------------------------------
# 6. TRAIN FOR FOOD EXPENDITURE
# --------------------------------------------------

food_results = train_and_evaluate(
    X_train,
    X_test,
    y_food,
    y_food_test,
    "Food"
)


# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

all_results = transport_results + food_results

results_df = pd.DataFrame(all_results)

os.makedirs("outputs/results", exist_ok=True)

results_df.to_csv(
    "outputs/results/model_comparison.csv",
    index=False
)

print("\n\n===== MODEL COMPARISON =====")

print(
    results_df.sort_values(
        by=["Target", "R2_Score"],
        ascending=[True, False]
    ).to_string(index=False)
)

print("\nModel comparison saved successfully!")
