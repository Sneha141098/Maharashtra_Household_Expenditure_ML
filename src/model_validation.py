# =====================================================
# STAGE 9: MODEL VALIDATION AND HYPERPARAMETER TUNING
# =====================================================

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
    RandomizedSearchCV
)

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
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

print("\n===== STAGE 9: MODEL VALIDATION =====")

os.makedirs("outputs/results", exist_ok=True)


# =====================================================
# 1. LOAD DATA
# =====================================================

df = pd.read_csv(
    "outputs/results/cleaned_household_data.csv"
)

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

targets = {
    "Transportation": "M_EXP_TRANSPORT",
    "Food": "M_EXP_FOOD"
}

X_train, X_test = train_test_split(
    X,
    test_size=0.20,
    random_state=42
)


# =====================================================
# 2. PREPROCESSING
# =====================================================

def create_preprocessor(cat_features):

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),

            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cat_features
            )
        ]
    )


preprocessor = create_preprocessor(
    categorical_features
)


# =====================================================
# 3. DEFINE MODELS
# =====================================================

models = {

    "Linear Regression": (
        LinearRegression(),
        {}
    ),

    "Decision Tree": (
        DecisionTreeRegressor(random_state=42),
        {
            "model__max_depth": [5, 10, 15, 20, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 5]
        }
    ),

    "Random Forest": (
        RandomForestRegressor(
            random_state=42,
            n_jobs=1
        ),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 20, None],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2]
        }
    ),

    "Gradient Boosting": (
        GradientBoostingRegressor(
            random_state=42
        ),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 5]
        }
    ),

    "Bagging Regressor": (
        BaggingRegressor(
            estimator=DecisionTreeRegressor(
                random_state=42
            ),
            random_state=42,
            n_jobs=1
        ),
        {
            "model__n_estimators": [50, 100],
            "model__max_samples": [0.7, 0.8, 1.0],
            "model__estimator__max_depth": [5, 10, None]
        }
    ),

    "Support Vector Regression": (
        SVR(kernel="rbf"),
        {
            "model__C": [1, 10, 50],
            "model__epsilon": [0.1, 0.2, 0.5],
            "model__gamma": ["scale", "auto"]
        }
    )
}


# =====================================================
# 4. CROSS VALIDATION
# =====================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

validation_results = []

best_estimators = {}

for target_name, target_column in targets.items():

    print("\n====================================")
    print("TARGET:", target_name)
    print("====================================")

    y = df.loc[X_train.index, target_column]

    for model_name, (model, parameters) in models.items():

        print("\nModel:", model_name)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        scoring = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error"
        }

        # 5-fold cross validation

        cv_results = cross_validate(
            pipeline,
            X_train,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        mean_r2 = np.mean(cv_results["test_r2"])

        mean_mae = -np.mean(cv_results["test_mae"])

        mean_rmse = -np.mean(cv_results["test_rmse"])

        print("CV R2:", round(mean_r2, 4))
        print("CV MAE:", round(mean_mae, 2))
        print("CV RMSE:", round(mean_rmse, 2))

        validation_results.append({

            "Target": target_name,
            "Model": model_name,
            "CV_R2": mean_r2,
            "CV_MAE": mean_mae,
            "CV_RMSE": mean_rmse

        })

        # =================================================
        # 5. HYPERPARAMETER TUNING
        # =================================================

        if parameters:

            print("Starting hyperparameter tuning...")

            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=parameters,
                n_iter=5,
                scoring="r2",
                cv=cv,
                random_state=42,
                n_jobs=-1
            )

            search.fit(X_train, y)

            best_model = search.best_estimator_

            print("Best Parameters:")
            print(search.best_params_)

            print(
                "Best CV R2:",
                round(search.best_score_, 4)
            )

        else:

            best_model = pipeline.fit(
                X_train,
                y
            )

        best_estimators[(target_name, model_name)] = best_model


# =====================================================
# 6. SAVE CROSS VALIDATION RESULTS
# =====================================================

validation_df = pd.DataFrame(
    validation_results
)

validation_df.to_csv(
    "outputs/results/cross_validation_results.csv",
    index=False
)

print("\n===== CROSS VALIDATION SUMMARY =====")

print(validation_df.to_string(index=False))


# =====================================================
# 7. TEST SET EVALUATION AFTER TUNING
# =====================================================

tuned_results = []

for target_name, target_column in targets.items():

    y_train = df.loc[X_train.index, target_column]

    y_test = df.loc[X_test.index, target_column]

    for model_name in models:

        model = best_estimators[
            (target_name, model_name)
        ]

        predictions = model.predict(X_test)

        tuned_results.append({

            "Target": target_name,
            "Model": model_name,

            "R2": r2_score(
                y_test,
                predictions
            ),

            "MAE": mean_absolute_error(
                y_test,
                predictions
            ),

            "RMSE": np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions
                )
            )

        })


tuned_df = pd.DataFrame(tuned_results)

tuned_df.to_csv(
    "outputs/results/tuned_model_results.csv",
    index=False
)

print("\n===== TUNED MODEL RESULTS =====")

print(tuned_df.to_string(index=False))


# =====================================================
# 8. DISTRICT ABLATION ANALYSIS
# =====================================================

print("\n===== DISTRICT IMPORTANCE INVESTIGATION =====")

features_without_district = [
    feature for feature in features
    if feature != "DISTRICT"
]

X_without_district = df[features_without_district]

X_train_no = X_without_district.loc[X_train.index]

X_test_no = X_without_district.loc[X_test.index]

no_district_categorical = [
    feature for feature in categorical_features
    if feature != "DISTRICT"
]

no_district_preprocessor = create_preprocessor(
    no_district_categorical
)

ablation_results = []

for target_name, target_column in targets.items():

    y_train = df.loc[X_train.index, target_column]

    y_test = df.loc[X_test.index, target_column]

    # Identify the best model based on cross-validation R2

    target_cv = validation_df[
        validation_df["Target"] == target_name
    ]

    best_model_name = target_cv.loc[
        target_cv["CV_R2"].idxmax(),
        "Model"
    ]

    print("\nTarget:", target_name)

    print("Selected model:", best_model_name)

    # Model with district

    model_with = best_estimators[
        (target_name, best_model_name)
    ]

    prediction_with = model_with.predict(X_test)

    r2_with = r2_score(
        y_test,
        prediction_with
    )

    # Model without district

    original_model = models[best_model_name][0]

    if best_model_name == "Bagging Regressor":

        # Reuse tuned parameters except preprocessing
        params = best_estimators[
            (target_name, best_model_name)
        ].named_steps["model"].get_params()

        model_without = original_model.set_params(
            **{
                key: value
                for key, value in params.items()
                if key != "estimator"
            }
        )

    else:

        model_without = original_model

    pipeline_without = Pipeline([

        ("preprocessor", no_district_preprocessor),

        ("model", model_without)

    ])

    pipeline_without.fit(
        X_train_no,
        y_train
    )

    prediction_without = pipeline_without.predict(
        X_test_no
    )

    r2_without = r2_score(
        y_test,
        prediction_without
    )

    difference = r2_with - r2_without

    print("R2 with district:", round(r2_with, 4))

    print("R2 without district:", round(r2_without, 4))

    print("R2 difference:", round(difference, 4))

    ablation_results.append({

        "Target": target_name,
        "Selected_Model": best_model_name,
        "R2_With_District": r2_with,
        "R2_Without_District": r2_without,
        "R2_Difference": difference

    })


ablation_df = pd.DataFrame(
    ablation_results
)

ablation_df.to_csv(
    "outputs/results/district_ablation_results.csv",
    index=False
)


# =====================================================
# 9. FINAL OUTPUT
# =====================================================

print("\n===== DISTRICT ABLATION SUMMARY =====")

print(ablation_df.to_string(index=False))

print("\n===== STAGE 9 COMPLETED =====")

print("\nFiles saved successfully:")
print("1. cross_validation_results.csv")
print("2. tuned_model_results.csv")
print("3. district_ablation_results.csv")