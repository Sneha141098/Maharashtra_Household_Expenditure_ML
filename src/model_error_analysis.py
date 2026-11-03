import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak
)

from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet


# =====================================================
# STAGE 15: MODEL ERROR ANALYSIS
# =====================================================

print("\n==========================================")
print("STAGE 15: MODEL ERROR ANALYSIS")
print("==========================================\n")


# =====================================================
# 1. PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "outputs/results/cleaned_household_data.csv"

MODEL_DIR = BASE_DIR / "outputs/models"

GRAPH_DIR = BASE_DIR / "outputs/graphs/error_analysis"

RESULT_DIR = BASE_DIR / "outputs/results"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. LOAD DATA
# =====================================================

print("Loading dataset...")

data = pd.read_csv(DATA_PATH)

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
    "M_EXP_FOOD",
    "M_EXP_TRANSPORT"
]

X = data[features].copy()

print("Dataset shape:", data.shape)


# =====================================================
# 3. TRAIN-TEST SPLIT
# =====================================================

X_train, X_test = train_test_split(
    X,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =====================================================
# 4. MODEL CONFIGURATION
# =====================================================

models = {

    "Food": {
        "file": MODEL_DIR / "M_EXP_FOOD_model.joblib",
        "target": "M_EXP_FOOD"
    },

    "Transportation": {
        "file": MODEL_DIR / "M_EXP_TRANSPORT_model.joblib",
        "target": "M_EXP_TRANSPORT"
    }

}


# =====================================================
# 5. STORAGE
# =====================================================

performance_results = []

district_results = []

generated_images = []


# =====================================================
# 6. MODEL ERROR ANALYSIS
# =====================================================

def analyse_model(model_name, model_path, target):

    print("\n==========================================")
    print(f"ANALYSING {model_name.upper()} MODEL")
    print("==========================================")

    # Load model

    model = joblib.load(model_path)

    # Actual values

    y_actual = data.loc[X_test.index, target]

    # Predictions

    y_predicted = model.predict(X_test)

    # Residuals

    residuals = y_actual.values - y_predicted

    # =================================================
    # 7. PERFORMANCE METRICS
    # =================================================

    mae = mean_absolute_error(
        y_actual,
        y_predicted
    )

    mse = mean_squared_error(
        y_actual,
        y_predicted
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_actual,
        y_predicted
    )

    performance_results.append({

        "Model": model_name,

        "Target": target,

        "MAE": mae,

        "MSE": mse,

        "RMSE": rmse,

        "R2_Score": r2

    })

    print("\nModel Performance:")

    print("MAE:", round(mae, 3))

    print("MSE:", round(mse, 3))

    print("RMSE:", round(rmse, 3))

    print("R2 Score:", round(r2, 4))


    # =================================================
    # 8. ACTUAL VS PREDICTED
    # =================================================

    print("\nGenerating Actual vs Predicted plot...")

    plt.figure(figsize=(9, 6))

    plt.scatter(
        y_actual,
        y_predicted,
        alpha=0.5
    )

    min_value = min(
        y_actual.min(),
        y_predicted.min()
    )

    max_value = max(
        y_actual.max(),
        y_predicted.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    plt.xlabel("Actual Expenditure")

    plt.ylabel("Predicted Expenditure")

    plt.title(
        f"Actual vs Predicted - {model_name}"
    )

    plt.tight_layout()

    actual_path = GRAPH_DIR / f"{target}_Actual_vs_Predicted.png"

    plt.savefig(
        actual_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    generated_images.append(
        (f"{model_name}: Actual vs Predicted", actual_path)
    )


    # =================================================
    # 9. RESIDUAL DISTRIBUTION
    # =================================================

    print("Generating residual distribution...")

    plt.figure(figsize=(9, 6))

    sns.histplot(
        residuals,
        kde=True
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.xlabel("Prediction Error (Actual - Predicted)")

    plt.ylabel("Frequency")

    plt.title(
        f"Residual Distribution - {model_name}"
    )

    plt.tight_layout()

    residual_path = GRAPH_DIR / f"{target}_Residual_Distribution.png"

    plt.savefig(
        residual_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    generated_images.append(
        (f"{model_name}: Residual Distribution", residual_path)
    )


    # =================================================
    # 10. RESIDUAL VS PREDICTED
    # =================================================

    print("Generating residual vs predicted plot...")

    plt.figure(figsize=(9, 6))

    plt.scatter(
        y_predicted,
        residuals,
        alpha=0.5
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Predicted Expenditure")

    plt.ylabel("Residuals")

    plt.title(
        f"Residuals vs Predicted - {model_name}"
    )

    plt.tight_layout()

    residual_pred_path = GRAPH_DIR / f"{target}_Residual_vs_Predicted.png"

    plt.savefig(
        residual_pred_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    generated_images.append(
        (f"{model_name}: Residual vs Predicted", residual_pred_path)
    )


    # =================================================
    # 11. DISTRICT-WISE ERROR ANALYSIS
    # =================================================

    print("Calculating district-wise errors...")

    district_data = pd.DataFrame({

        "DISTRICT": data.loc[X_test.index, "DISTRICT"].values,

        "Actual": y_actual.values,

        "Predicted": y_predicted

    })

    district_data["Absolute_Error"] = abs(
        district_data["Actual"] -
        district_data["Predicted"]
    )

    district_summary = district_data.groupby(
        "DISTRICT"
    ).agg(

        MAE=("Absolute_Error", "mean"),

        Mean_Actual=("Actual", "mean"),

        Mean_Predicted=("Predicted", "mean"),

        Household_Count=("Actual", "count")

    ).reset_index()

    district_summary["Target"] = target

    district_results.append(district_summary)

    print("\nDistrict-wise MAE:")

    print(
        district_summary.sort_values(
            "MAE",
            ascending=False
        ).head(10).to_string(index=False)
    )

    print(f"\n{model_name} analysis completed.")


# =====================================================
# 12. RUN BOTH MODELS
# =====================================================

for model_name, config in models.items():

    model_path = config["file"]

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    analyse_model(
        model_name,
        model_path,
        config["target"]
    )


# =====================================================
# 13. SAVE PERFORMANCE RESULTS
# =====================================================

performance_df = pd.DataFrame(
    performance_results
)

performance_path = RESULT_DIR / "Model_Error_Performance.csv"

performance_df.to_csv(
    performance_path,
    index=False
)

print("\nPerformance results saved.")


# =====================================================
# 14. SAVE DISTRICT-WISE ERRORS
# =====================================================

district_df = pd.concat(
    district_results,
    ignore_index=True
)

district_path = RESULT_DIR / "District_Wise_Model_Error.csv"

district_df.to_csv(
    district_path,
    index=False
)

print("District-wise results saved.")


# =====================================================
# 15. GENERATE PDF REPORT
# =====================================================

print("\nGenerating PDF report...")

pdf_path = BASE_DIR / "outputs/Stage_15_Model_Error_Report.pdf"

page_width, page_height = landscape(A4)

pdf = SimpleDocTemplate(
    str(pdf_path),
    pagesize=landscape(A4),
    rightMargin=30,
    leftMargin=30,
    topMargin=30,
    bottomMargin=30
)

styles = getSampleStyleSheet()

elements = []

elements.append(
    Paragraph(
        "STAGE 15: MODEL ERROR ANALYSIS REPORT",
        styles["Title"]
    )
)

elements.append(Spacer(1, 20))

elements.append(
    Paragraph(
        "Evaluation of food and transportation expenditure "
        "prediction models using regression performance metrics "
        "and residual diagnostics.",
        styles["BodyText"]
    )
)

elements.append(Spacer(1, 20))

elements.append(
    Paragraph(
        "Model Performance Summary",
        styles["Heading2"]
    )
)

for _, row in performance_df.iterrows():

    text = (
        f"{row['Model']} Model | "
        f"MAE: {row['MAE']:.2f} | "
        f"RMSE: {row['RMSE']:.2f} | "
        f"R2: {row['R2_Score']:.4f}"
    )

    elements.append(
        Paragraph(text, styles["BodyText"])
    )

elements.append(PageBreak())

available_width = page_width - 60
available_height = page_height - 90

for index, (title, image_path) in enumerate(generated_images):

    elements.append(
        Paragraph(title, styles["Heading2"])
    )

    elements.append(Spacer(1, 15))

    from PIL import Image as PILImage

    with PILImage.open(image_path) as img:
        width, height = img.size

    scale = min(
        available_width / width,
        available_height / height
    ) * 0.90

    graph = Image(
        str(image_path),
        width=width * scale,
        height=height * scale
    )

    elements.append(graph)

    if index < len(generated_images) - 1:
        elements.append(PageBreak())

pdf.build(elements)


# =====================================================
# 16. FINAL OUTPUT
# =====================================================

print("\n==========================================")
print("STAGE 15 COMPLETED SUCCESSFULLY")
print("==========================================")

print("\nModel Performance:")
print(performance_df.to_string(index=False))

print("\nGenerated Graphs:", len(generated_images))

print("\nPDF Report:")
print(pdf_path)

print("\nPerformance CSV:")
print(performance_path)

print("\nDistrict Error CSV:")
print(district_path)