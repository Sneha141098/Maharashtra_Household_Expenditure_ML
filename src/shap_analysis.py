import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image as PILImage

from reportlab.platypus import (
    SimpleDocTemplate,
    Image,
    Paragraph,
    Spacer,
    PageBreak
)

from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet


# ==========================================================
# STAGE 14: SHAP MODEL EXPLAINABILITY
# ==========================================================

print("\n==============================================")
print("STAGE 14: SHAP MODEL EXPLAINABILITY")
print("==============================================\n")


# ==========================================================
# 1. DEFINE PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "outputs/results/cleaned_household_data.csv"

MODEL_DIR = BASE_DIR / "outputs/models"

GRAPH_DIR = BASE_DIR / "outputs/graphs/shap"

RESULT_DIR = BASE_DIR / "outputs/results"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. LOAD DATASET
# ==========================================================

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

X = data[features].copy()

print("Dataset shape:", X.shape)


# ==========================================================
# 3. PREPARE BACKGROUND AND EXPLANATION DATA
# ==========================================================

X_background = X.sample(
    n=min(20, len(X)),
    random_state=42
)

X_explain = X.sample(
    n=min(30, len(X)),
    random_state=7
)

print("Background samples:", len(X_background))
print("Explanation samples:", len(X_explain))


# ==========================================================
# 4. PREPARE NUMERIC DATA FOR SHAP VISUALIZATIONS
# ==========================================================

# SHAP calculations use the original categorical values.
# Numeric category codes are used only for plotting.

categorical_features = [
    "DISTRICT",
    "AGE_GROUP",
    "OCCUPATION_GROUP",
    "EDU_GROUP",
    "GENDER_GROUP",
    "SIZE_GROUP"
]

plot_data = X_explain.copy()

category_mappings = {}

for col in categorical_features:

    values = data[col].astype(str)

    unique_values = sorted(values.unique())

    mapping = {
        value: index
        for index, value in enumerate(unique_values)
    }

    category_mappings[col] = mapping

    plot_data[col] = (
        X_explain[col]
        .astype(str)
        .map(mapping)
        .astype(float)
    )

plot_matrix = plot_data.to_numpy(dtype=float)

# Save category mapping for interpretation

mapping_rows = []

for col, mapping in category_mappings.items():

    for category, code in mapping.items():

        mapping_rows.append({
            "Feature": col,
            "Category": category,
            "Numeric_Code": code
        })

pd.DataFrame(mapping_rows).to_csv(
    RESULT_DIR / "SHAP_Category_Mapping.csv",
    index=False
)


# ==========================================================
# 5. DEFINE TRAINED MODELS
# ==========================================================

models = {

    "Food": (
        MODEL_DIR / "M_EXP_FOOD_model.joblib",
        "M_EXP_FOOD"
    ),

    "Transportation": (
        MODEL_DIR / "M_EXP_TRANSPORT_model.joblib",
        "M_EXP_TRANSPORT"
    )

}


# ==========================================================
# 6. STORAGE
# ==========================================================

all_importance = []

generated_images = []


# ==========================================================
# 7. SHAP ANALYSIS FUNCTION
# ==========================================================

def perform_shap_analysis(model_name, model_path, target):

    print("\n==============================================")
    print(f"ANALYSING {model_name.upper()} EXPENDITURE")
    print("==============================================")

    # ------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------

    model = joblib.load(model_path)

    print("Model loaded:", type(model).__name__)

    # ------------------------------------------------------
    # PREDICTION FUNCTION
    # ------------------------------------------------------

    original_dtypes = X.dtypes.to_dict()

    def predict_function(input_data):

        input_data = np.atleast_2d(input_data)

        input_df = pd.DataFrame(
            input_data,
            columns=features
        )

        # Restore original column data types
        for col in features:

            input_df[col] = input_df[col].astype(
                original_dtypes[col]
            )

        # Ensure income is numeric
        input_df["TOT_INCOME"] = pd.to_numeric(
            input_df["TOT_INCOME"]
        )

        return model.predict(input_df)

    # ------------------------------------------------------
    # CREATE KERNEL EXPLAINER
    # ------------------------------------------------------

    print("\nPreparing SHAP background data...")

    background = X_background.to_numpy(dtype=object)

    explain_data = X_explain.to_numpy(dtype=object)

    explainer = shap.KernelExplainer(
        predict_function,
        background
    )

    # ------------------------------------------------------
    # CALCULATE SHAP VALUES
    # ------------------------------------------------------

    print("Calculating SHAP values...")

    raw_shap_values = explainer.shap_values(
        explain_data,
        nsamples=100
    )

    # Handle different SHAP output formats

    if isinstance(raw_shap_values, list):

        raw_shap_values = raw_shap_values[0]

    raw_shap_values = np.asarray(raw_shap_values)

    if raw_shap_values.ndim == 3:

        if raw_shap_values.shape[0] == 1:

            raw_shap_values = raw_shap_values[0]

        elif raw_shap_values.shape[-1] == 1:

            raw_shap_values = raw_shap_values[:, :, 0]

        else:

            raise ValueError(
                "Unexpected SHAP output dimensions."
            )

    print("SHAP calculation completed.")

    # ------------------------------------------------------
    # CREATE SHAP EXPLANATION OBJECT
    # ------------------------------------------------------

    expected_value = float(
        np.asarray(explainer.expected_value).reshape(-1)[0]
    )

    shap_values = shap.Explanation(

        values=raw_shap_values,

        base_values=np.full(
            len(X_explain),
            expected_value
        ),

        data=plot_matrix,

        feature_names=features

    )

    # ------------------------------------------------------
    # 8. FEATURE IMPORTANCE
    # ------------------------------------------------------

    print("\nCalculating feature importance...")

    importance = np.abs(raw_shap_values).mean(axis=0)

    importance_df = pd.DataFrame({

        "Feature": features,

        "Mean_Absolute_SHAP": importance

    })

    importance_df = importance_df.sort_values(

        by="Mean_Absolute_SHAP",

        ascending=False

    )

    importance_df["Target"] = target

    all_importance.append(importance_df)

    print("\nSHAP Feature Importance:")

    print(
        importance_df.to_string(index=False)
    )

    # ------------------------------------------------------
    # 9. SHAP SUMMARY PLOT
    # ------------------------------------------------------

    print("\nGenerating SHAP summary plot...")

    plt.figure(figsize=(10, 7))

    shap.summary_plot(

        raw_shap_values,

        plot_matrix,

        feature_names=features,

        show=False,

        plot_type="dot"

    )

    plt.title(
        f"SHAP Summary Plot - {model_name}"
    )

    plt.tight_layout()

    summary_path = GRAPH_DIR / f"{target}_SHAP_Summary.png"

    plt.savefig(

        summary_path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close("all")

    generated_images.append({

        "title": f"{model_name} - SHAP Summary Plot",

        "path": summary_path

    })

    # ------------------------------------------------------
    # 10. SHAP BAR PLOT
    # ------------------------------------------------------

    print("Generating SHAP bar plot...")

    plt.figure(figsize=(10, 6))

    shap.plots.bar(

        shap_values,

        max_display=7,

        show=False

    )

    plt.title(
        f"SHAP Feature Importance - {model_name}"
    )

    plt.tight_layout()

    bar_path = GRAPH_DIR / f"{target}_SHAP_Bar.png"

    plt.savefig(

        bar_path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close("all")

    generated_images.append({

        "title": f"{model_name} - SHAP Feature Importance",

        "path": bar_path

    })

    # ------------------------------------------------------
    # 11. SHAP WATERFALL PLOT
    # ------------------------------------------------------

    print("Generating SHAP waterfall plot...")

    plt.figure(figsize=(10, 7))

    shap.plots.waterfall(

        shap_values[0],

        max_display=7,

        show=False

    )

    plt.title(
        f"Individual Prediction Explanation - {model_name}"
    )

    plt.tight_layout()

    waterfall_path = GRAPH_DIR / f"{target}_SHAP_Waterfall.png"

    plt.savefig(

        waterfall_path,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close("all")

    generated_images.append({

        "title": f"{model_name} - Individual Prediction Explanation",

        "path": waterfall_path

    })

    print(f"\n{model_name} analysis completed successfully.")


# ==========================================================
# 12. RUN SHAP FOR BOTH MODELS
# ==========================================================

for model_name, (model_path, target) in models.items():

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    perform_shap_analysis(

        model_name,

        model_path,

        target

    )


# ==========================================================
# 13. SAVE FEATURE IMPORTANCE CSV
# ==========================================================

print("\nSaving SHAP feature importance...")

final_importance = pd.concat(

    all_importance,

    ignore_index=True

)

importance_path = RESULT_DIR / "SHAP_Feature_Importance.csv"

final_importance.to_csv(

    importance_path,

    index=False

)

print("Feature importance saved:", importance_path)


# ==========================================================
# 14. GENERATE PDF REPORT
# ==========================================================

print("\nGenerating PDF report...")

pdf_path = BASE_DIR / "outputs/SHAP_Explainability_Report.pdf"

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

# Report heading

elements.append(

    Paragraph(

        "STAGE 14: SHAP MODEL EXPLAINABILITY REPORT",

        styles["Title"]

    )

)

elements.append(Spacer(1, 20))

elements.append(

    Paragraph(

        "Machine Learning-Based Analysis of Household "
        "Food and Transportation Expenditure Patterns "
        "in Maharashtra.",

        styles["BodyText"]

    )

)

elements.append(Spacer(1, 20))

# Add graphs

available_width = page_width - 60

available_height = page_height - 100

for index, item in enumerate(generated_images):

    image_path = item["path"]

    elements.append(

        Paragraph(

            item["title"],

            styles["Heading2"]

        )

    )

    elements.append(Spacer(1, 15))

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


# ==========================================================
# 15. FINAL OUTPUT
# ==========================================================

print("\n==============================================")
print("STAGE 14 COMPLETED SUCCESSFULLY")
print("==============================================")

print("\nGenerated graphs:", len(generated_images))

print("\nPDF Report:")
print(pdf_path)

print("\nFeature Importance:")
print(importance_path)

print("\nCategory Mapping:")
print(RESULT_DIR / "SHAP_Category_Mapping.csv")

print("\nGraph Directory:")
print(GRAPH_DIR)

print("\nAll operations completed!")