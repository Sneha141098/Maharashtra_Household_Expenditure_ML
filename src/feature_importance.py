import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

# ------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------

print("\n===== STAGE 7: FEATURE IMPORTANCE ANALYSIS =====")

df = pd.read_csv("outputs/results/cleaned_household_data.csv")

results_df = pd.read_csv(
    "outputs/results/model_comparison.csv"
)

os.makedirs("outputs/graphs", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

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

X_train, X_test = train_test_split(
    X,
    test_size=0.20,
    random_state=42
)

y_targets = {
    "Transportation": "M_EXP_TRANSPORT",
    "Food": "M_EXP_FOOD"
}


# ------------------------------------------------
# 2. PREPROCESSING
# ------------------------------------------------

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


# ------------------------------------------------
# 3. MODEL DEFINITIONS
# ------------------------------------------------

def get_model(model_name):

    model_dict = {

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

        "Bagging Regressor": BaggingRegressor(
            estimator=DecisionTreeRegressor(
                random_state=42
            ),
            n_estimators=100,
            max_samples=0.8,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        ),

        "Support Vector Regression": SVR(
            kernel="rbf",
            C=10,
            epsilon=0.1
        )
    }

    return model_dict[model_name]


# ------------------------------------------------
# 4. IDENTIFY BEST MODEL FOR EACH TARGET
# ------------------------------------------------

all_importance = []

for target_name, target_column in y_targets.items():

    print(f"\n========== {target_name} ==========")

    target_results = results_df[
        results_df["Target"] == target_name
    ]

    best_model_name = target_results.loc[
        target_results["R2_Score"].idxmax(),
        "Model"
    ]

    print("Best model:", best_model_name)

    y = df[target_column]

    y_train = y.loc[X_train.index]
    y_test = y.loc[X_test.index]

    model = get_model(best_model_name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train the selected model
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    print(
        "Test R2 Score:",
        round(r2_score(y_test, predictions), 4)
    )

    # ------------------------------------------------
    # 5. PERMUTATION IMPORTANCE
    # ------------------------------------------------

    importance = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="r2",
        n_jobs=-1
    )

    importance_df = pd.DataFrame({

        "Feature": features,

        "Importance_Mean": importance.importances_mean,

        "Importance_Std": importance.importances_std,

        "Target": target_name,

        "Best_Model": best_model_name

    })

    importance_df = importance_df.sort_values(
        by="Importance_Mean",
        ascending=False
    )

    all_importance.append(importance_df)

    print("\nFeature Importance Ranking:")

    print(
        importance_df[
            ["Feature", "Importance_Mean", "Importance_Std"]
        ].to_string(index=False)
    )

    # ------------------------------------------------
    # 6. FEATURE IMPORTANCE GRAPH
    # ------------------------------------------------

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=importance_df,
        x="Importance_Mean",
        y="Feature"
    )

    plt.title(
        f"Feature Importance - {target_name}\n"
        f"Best Model: {best_model_name}"
    )

    plt.xlabel("Decrease in R2 after permutation")
    plt.ylabel("Household Characteristics")

    plt.tight_layout()

    filename = (
        f"outputs/graphs/"
        f"{target_name.lower()}_feature_importance.png"
    )

    plt.savefig(filename, dpi=300)

    plt.close()


# ------------------------------------------------
# 7. SAVE COMBINED RESULTS
# ------------------------------------------------

final_importance = pd.concat(
    all_importance,
    ignore_index=True
)

final_importance.to_csv(
    "outputs/results/feature_importance.csv",
    index=False
)

print("\n===== FINAL FEATURE IMPORTANCE =====")

print(
    final_importance.to_string(index=False)
)

print("\nFeature importance analysis completed!")


# ------------------------------------------------
# 8. GENERATE PDF REPORT
# ------------------------------------------------

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)

pdf_path = "outputs/Stage_7_Feature_Importance_Report.pdf"

doc = SimpleDocTemplate(
    pdf_path,
    pagesize=A4,
    rightMargin=40,
    leftMargin=40,
    topMargin=40,
    bottomMargin=40
)

styles = getSampleStyleSheet()

story = []

story.append(Paragraph(
    "Feature Importance Analysis",
    styles["Title"]
))

story.append(Spacer(1, 12))

story.append(Paragraph(
    "Machine Learning-Based Analysis of Household "
    "Expenditure in Urban Maharashtra",
    styles["Heading2"]
))

story.append(Spacer(1, 15))

story.append(Paragraph(
    "1. Methodology",
    styles["Heading1"]
))

story.append(Paragraph(
    "Permutation importance was used to evaluate the predictive "
    "contribution of household characteristics. The method measures "
    "the reduction in model R2 score after randomly shuffling each "
    "feature. The analysis uses the best-performing model selected "
    "separately for each expenditure target.",
    styles["BodyText"]
))

story.append(Spacer(1, 15))

story.append(Paragraph(
    "2. Feature Importance Results",
    styles["Heading1"]
))

for target_name in y_targets:

    target_df = final_importance[
        final_importance["Target"] == target_name
    ].sort_values(
        "Importance_Mean",
        ascending=False
    )

    story.append(Paragraph(
        f"{target_name} Expenditure",
        styles["Heading2"]
    ))

    best_model = target_df["Best_Model"].iloc[0]

    story.append(Paragraph(
        f"Selected model: {best_model}",
        styles["BodyText"]
    ))

    table_data = [
        ["Feature", "Importance", "Std. Dev."]
    ]

    for _, row in target_df.iterrows():

        table_data.append([
            row["Feature"],
            f'{row["Importance_Mean"]:.5f}',
            f'{row["Importance_Std"]:.5f}'
        ])

    table = Table(
        table_data,
        colWidths=[180, 130, 130]
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))

    story.append(table)
    story.append(Spacer(1, 15))

    graph_path = (
        f"outputs/graphs/"
        f"{target_name.lower()}_feature_importance.png"
    )

    story.append(Image(
        graph_path,
        width=6.2 * inch,
        height=3.8 * inch
    ))

    story.append(Spacer(1, 15))

story.append(PageBreak())

story.append(Paragraph(
    "3. Interpretation and Limitations",
    styles["Heading1"]
))

story.append(Paragraph(
    "Features with larger positive permutation importance values "
    "contribute more to the selected model's predictive performance. "
    "Near-zero values indicate limited additional predictive "
    "contribution under this evaluation. Negative values can occur "
    "because of sampling variability or model behaviour.",
    styles["BodyText"]
))

story.append(Spacer(1, 10))

story.append(Paragraph(
    "Feature importance indicates predictive relevance and does not "
    "establish a causal relationship. Correlated household "
    "characteristics can also share or obscure predictive importance.",
    styles["BodyText"]
))

doc.build(story)

print("\nPDF generated successfully!")
print("Location:", pdf_path)