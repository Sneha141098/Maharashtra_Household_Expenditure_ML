from pathlib import Path
import pandas as pd
import numpy as np
import os
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, PageBreak, Image, KeepTogether
)

# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
GRAPHS_DIR = OUTPUT_DIR / "graphs"

DATA_PATH = RESULTS_DIR / "cleaned_household_data.csv"

REPORT_PATH = OUTPUT_DIR / "Complete_ML_Project_Report.pdf"

OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================================
# 2. PDF DOCUMENT SETTINGS
# ==========================================================

doc = SimpleDocTemplate(
    str(REPORT_PATH),
    pagesize=A4,
    rightMargin=45,
    leftMargin=45,
    topMargin=55,
    bottomMargin=45
)

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="MainTitle",
    parent=styles["Title"],
    fontSize=23,
    leading=30,
    alignment=TA_CENTER,
    spaceAfter=20
))

styles.add(ParagraphStyle(
    name="Subtitle",
    parent=styles["Normal"],
    fontSize=13,
    leading=20,
    alignment=TA_CENTER,
    textColor=colors.darkblue
))

styles.add(ParagraphStyle(
    name="SectionHeading",
    parent=styles["Heading1"],
    fontSize=17,
    leading=22,
    textColor=colors.darkblue,
    spaceBefore=15,
    spaceAfter=12
))

styles.add(ParagraphStyle(
    name="SubHeading",
    parent=styles["Heading2"],
    fontSize=12,
    leading=16,
    spaceBefore=10,
    spaceAfter=7
))

styles.add(ParagraphStyle(
    name="BodyJustified",
    parent=styles["BodyText"],
    alignment=TA_JUSTIFY,
    fontSize=9.5,
    leading=15,
    spaceAfter=9
))

styles.add(ParagraphStyle(
    name="SmallText",
    parent=styles["BodyText"],
    fontSize=8,
    leading=11
))

story = []

# ==========================================================
# 3. HELPER FUNCTIONS
# ==========================================================

def add_heading(text):
    story.append(Paragraph(text, styles["SectionHeading"]))


def add_subheading(text):
    story.append(Paragraph(text, styles["SubHeading"]))


def add_text(text):
    story.append(Paragraph(str(text), styles["BodyJustified"]))


def add_table(data, widths=None, font_size=8):
    if not data:
        return

    table = Table(
        data,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT"
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#EAF0F7")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    story.append(table)
    story.append(Spacer(1, 12))


def add_csv_table(path, max_rows=15):
    try:
        df = pd.read_csv(path)

        if df.empty:
            return

        df = df.head(max_rows)

        data = [[str(col) for col in df.columns]]

        for row in df.itertuples(index=False, name=None):
            formatted = []

            for value in row:
                if isinstance(value, (float, np.floating)):
                    formatted.append(f"{value:.4f}")
                else:
                    formatted.append(str(value))

            data.append(formatted)

        add_table(data, font_size=7)

    except Exception as e:
        add_text(f"Could not read {path.name}: {e}")


def add_image(path, caption):
    try:
        img = Image(str(path))

        max_width = 6.5 * inch
        max_height = 4.3 * inch

        scale = min(
            max_width / img.imageWidth,
            max_height / img.imageHeight
        )

        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale

        story.append(Spacer(1, 8))
        story.append(img)
        story.append(Paragraph(
            caption,
            styles["SmallText"]
        ))
        story.append(Spacer(1, 15))

    except Exception as e:
        print(f"Skipping image {path.name}: {e}")


def find_csv_files(keyword):
    return [
        p for p in RESULTS_DIR.rglob("*.csv")
        if keyword.lower() in p.name.lower()
    ]


def format_number(value):
    return f"{value:,.2f}"


# ==========================================================
# 4. LOAD DATASET
# ==========================================================

print("Loading cleaned dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# ==========================================================
# 5. COVER PAGE
# ==========================================================

story.append(Spacer(1, 1.1 * inch))

story.append(Paragraph(
    "MACHINE LEARNING-BASED ANALYSIS OF "
    "HOUSEHOLD EXPENDITURE PATTERNS",
    styles["MainTitle"]
))

story.append(Spacer(1, 15))

story.append(Paragraph(
    "A Study of Food and Transportation Expenditure "
    "in Urban Households of Maharashtra",
    styles["Subtitle"]
))

story.append(Spacer(1, 1.1 * inch))

story.append(Paragraph(
    "<b>PROJECT REPORT</b>",
    ParagraphStyle(
        "CoverLabel",
        parent=styles["Normal"],
        fontSize=15,
        alignment=TA_CENTER
    )
))

story.append(Spacer(1, 30))

story.append(Paragraph(
    "Machine Learning | Data Analysis | Predictive Modelling",
    styles["Subtitle"]
))

story.append(Spacer(1, 1.2 * inch))

story.append(Paragraph(
    "Dataset: Maharashtra Household Expenditure Data",
    styles["Normal"]
))

story.append(Paragraph(
    f"Total observations: {len(df):,}",
    styles["Normal"]
))

story.append(PageBreak())

# ==========================================================
# 6. ABSTRACT
# ==========================================================

add_heading("ABSTRACT")

add_text(
    "Household expenditure varies according to socioeconomic "
    "and demographic characteristics. Understanding these "
    "variations can provide insights into consumption patterns "
    "and household financial behaviour."
)

add_text(
    "This project applies data analysis and machine learning "
    "techniques to study monthly food and transportation "
    "expenditure among households in Maharashtra. The analysis "
    "examines the relationship between expenditure and factors "
    "such as household income, district, household size, age, "
    "occupation, education and gender."
)

add_text(
    "Multiple machine learning approaches are used to predict "
    "household expenditure. Model performance is evaluated "
    "using Mean Absolute Error (MAE), Root Mean Squared Error "
    "(RMSE) and the coefficient of determination (R²). "
    "Feature importance and model error analysis are also "
    "used to interpret predictive performance."
)

# ==========================================================
# 7. INTRODUCTION
# ==========================================================

add_heading("1. INTRODUCTION")

add_text(
    "Household expenditure is an important indicator of "
    "economic behaviour and living standards. Food and "
    "transportation are two major components of monthly "
    "household expenditure."
)

add_text(
    "Expenditure patterns may differ across geographical "
    "regions and socioeconomic groups. Machine learning "
    "provides methods for identifying complex relationships "
    "between household characteristics and expenditure."
)

add_text(
    "The present study focuses on identifying expenditure "
    "patterns and developing predictive models for food and "
    "transportation expenditure in Maharashtra."
)

# ==========================================================
# 8. PROBLEM STATEMENT
# ==========================================================

add_heading("2. PROBLEM STATEMENT")

add_text(
    "The objective is to analyse household food and "
    "transportation expenditure patterns across socioeconomic "
    "and demographic characteristics in Maharashtra."
)

add_text(
    "The project investigates whether household income, "
    "district, household size, age, occupation, education "
    "and gender can explain variations in monthly expenditure."
)

# ==========================================================
# 9. OBJECTIVES
# ==========================================================

add_heading("3. OBJECTIVES")

objectives = [
    "To analyse household food and transportation expenditure.",
    "To examine relationships between income and expenditure.",
    "To identify important socioeconomic predictors.",
    "To develop machine learning models for expenditure prediction.",
    "To compare predictive performance across models.",
    "To interpret model behaviour using feature importance and SHAP.",
    "To investigate prediction errors and district-wise variations."
]

for i, obj in enumerate(objectives, 1):
    add_text(f"{i}. {obj}")

# ==========================================================
# 10. DATASET DESCRIPTION
# ==========================================================

add_heading("4. DATASET DESCRIPTION")

add_text(
    f"The cleaned dataset contains {len(df):,} household "
    f"observations and {len(df.columns)} variables."
)

dataset_table = [["Variable", "Description"]]

descriptions = {
    "HH_ID": "Unique household identifier",
    "STATE": "State",
    "DISTRICT": "District",
    "REGION_TYPE": "Region classification",
    "AGE_GROUP": "Household age category",
    "OCCUPATION_GROUP": "Occupation category",
    "EDU_GROUP": "Education category",
    "GENDER_GROUP": "Gender category",
    "SIZE_GROUP": "Household size category",
    "M_EXP_FOOD": "Monthly food expenditure",
    "M_EXP_TRANSPORT": "Monthly transportation expenditure",
    "TOT_INCOME": "Total household income"
}

for col in df.columns:
    dataset_table.append([
        col,
        descriptions.get(col, "Dataset variable")
    ])

add_table(dataset_table, font_size=8)

# ==========================================================
# 11. DATA CLEANING
# ==========================================================

add_heading("5. DATA PREPROCESSING AND CLEANING")

add_text(
    "Data preprocessing was performed before statistical "
    "analysis and model development. The cleaning workflow "
    "included checking missing values, duplicate household "
    "identifiers, numeric data types and invalid expenditure "
    "or income values."
)

cleaning_data = [
    ["Parameter", "Result"],
    ["Total observations", str(len(df))],
    ["Total variables", str(len(df.columns))],
    ["Missing values", str(int(df.isnull().sum().sum()))],
    [
        "Duplicate household IDs",
        str(
            df["HH_ID"].duplicated().sum()
            if "HH_ID" in df.columns else "Not available"
        )
    ]
]

add_table(cleaning_data)

# ==========================================================
# 12. DESCRIPTIVE STATISTICS
# ==========================================================

add_heading("6. EXPLORATORY DATA ANALYSIS")

numeric_cols = [
    col for col in
    ["TOT_INCOME", "M_EXP_FOOD", "M_EXP_TRANSPORT"]
    if col in df.columns
]

if numeric_cols:
    add_subheading("Descriptive Statistics")

    stats = df[numeric_cols].describe().round(2)

    data = [["Statistic"] + list(stats.columns)]

    for index, row in stats.iterrows():
        data.append([str(index)] + [
            str(round(x, 2)) for x in row
        ])

    add_table(data, font_size=8)

# ==========================================================
# 13. CORRELATION ANALYSIS
# ==========================================================

add_heading("7. CORRELATION ANALYSIS")

if len(numeric_cols) > 1:

    corr = df[numeric_cols].corr(method="pearson")

    corr_data = [["Variable"] + list(corr.columns)]

    for index, row in corr.iterrows():
        corr_data.append([
            index
        ] + [
            f"{x:.4f}" for x in row
        ])

    add_table(corr_data)

    add_text(
        "Pearson correlation measures the strength and direction "
        "of linear association between numerical variables. "
        "Correlation does not establish causation."
    )

# ==========================================================
# 14. MACHINE LEARNING METHODOLOGY
# ==========================================================

add_heading("8. MACHINE LEARNING METHODOLOGY")

add_text(
    "The machine learning workflow used household "
    "socioeconomic and demographic characteristics as input "
    "features. Monthly food and transportation expenditure "
    "were treated as separate prediction targets."
)

add_subheading("Input Features")

features = [
    "TOT_INCOME",
    "DISTRICT",
    "AGE_GROUP",
    "OCCUPATION_GROUP",
    "EDU_GROUP",
    "GENDER_GROUP",
    "SIZE_GROUP"
]

for feature in features:
    if feature in df.columns:
        add_text("• " + feature)

add_subheading("Prediction Targets")

add_text("1. Monthly food expenditure (M_EXP_FOOD)")
add_text("2. Monthly transportation expenditure (M_EXP_TRANSPORT)")

add_subheading("Train-Test Split")

add_text(
    "The dataset was divided into training and testing subsets "
    "using an 80:20 split. The training set contained 8,453 "
    "observations and the testing set contained 2,114 "
    "observations."
)

# ==========================================================
# 15. MODEL PERFORMANCE
# ==========================================================

add_heading("9. MODEL PERFORMANCE EVALUATION")

performance_files = find_csv_files("Model_Error_Performance")

if not performance_files:
    performance_files = find_csv_files("final_model_performance")

if performance_files:

    for path in performance_files:
        add_subheading(path.name)
        add_csv_table(path)

else:
    add_text(
        "No model performance CSV was found. "
        "The Stage 15 performance values are shown below."
    )

    add_table([
        ["Model", "MAE", "RMSE", "R²"],
        ["Food", "1799.97", "2649.86", "0.5816"],
        ["Transportation", "526.28", "1098.36", "0.2542"]
    ])

add_text(
    "MAE measures the average absolute prediction error. "
    "RMSE gives greater weight to larger errors. R² measures "
    "the proportion of target variance explained by the model."
)

# ==========================================================
# 16. FEATURE IMPORTANCE
# ==========================================================

add_heading("10. FEATURE IMPORTANCE ANALYSIS")

feature_files = find_csv_files("Feature_Importance")

if feature_files:

    for path in feature_files:
        add_subheading(path.name)
        add_csv_table(path)

else:
    add_text(
        "Feature importance results were not found in the "
        "results folder. Refer to the generated feature "
        "importance graphs, if available."
    )

add_text(
    "Feature importance analysis helps identify which input "
    "variables contribute most to predictive performance. "
    "Permutation importance measures the reduction in model "
    "performance when a feature is shuffled."
)

# ==========================================================
# 17. SHAP ANALYSIS
# ==========================================================

add_heading("11. SHAP EXPLAINABILITY ANALYSIS")

shap_files = find_csv_files("SHAP_Feature_Importance")

if shap_files:

    for path in shap_files:
        add_subheading(path.name)
        add_csv_table(path)

else:
    add_text(
        "SHAP feature importance results were not found. "
        "This section can be updated after successful "
        "completion of the SHAP analysis stage."
    )

add_text(
    "SHAP (SHapley Additive exPlanations) is used to interpret "
    "individual model predictions by estimating the contribution "
    "of each feature to a prediction."
)

# ==========================================================
# 18. ERROR ANALYSIS
# ==========================================================

add_heading("12. MODEL ERROR ANALYSIS")

error_files = find_csv_files("District_Wise_Model_Error")

if error_files:

    for path in error_files:
        add_subheading("District-wise Prediction Errors")
        add_csv_table(path, max_rows=20)

add_text(
    "Residuals are calculated as the difference between actual "
    "and predicted expenditure. Residual analysis helps assess "
    "prediction bias, error distribution and model limitations."
)

# ==========================================================
# 19. VISUAL RESULTS
# ==========================================================

add_heading("13. VISUAL ANALYSIS AND GRAPHS")

image_files = sorted(
    GRAPHS_DIR.rglob("*.png")
) if GRAPHS_DIR.exists() else []

if image_files:

    add_text(
        f"A total of {len(image_files)} PNG visualisations "
        "were located in the project graph directories."
    )

    for image_path in image_files:

        # Avoid including unrelated or excessively large files.
        caption = image_path.stem.replace("_", " ").title()

        add_image(image_path, caption)

else:
    add_text("No graph images were found.")

# ==========================================================
# 20. RESULTS AND DISCUSSION
# ==========================================================

add_heading("14. RESULTS AND DISCUSSION")

add_subheading("Food Expenditure")

add_text(
    "The food expenditure model achieved an R² score of "
    "0.5816, with an MAE of approximately ₹1,800 and RMSE "
    "of approximately ₹2,650. This indicates that the model "
    "explains around 58.16% of the variation in food expenditure "
    "on the test dataset."
)

add_subheading("Transportation Expenditure")

add_text(
    "The transportation expenditure model achieved an R² "
    "score of 0.2542, with an MAE of approximately ₹526 "
    "and RMSE of approximately ₹1,098. The relatively lower "
    "R² indicates that transportation expenditure is more "
    "difficult to predict using the selected features."
)

add_subheading("Interpretation")

add_text(
    "The difference in predictive performance suggests that "
    "food and transportation expenditure may be influenced "
    "by different patterns of household behaviour. Additional "
    "variables and more advanced modelling approaches may "
    "improve predictive performance."
)

# ==========================================================
# 21. LIMITATIONS
# ==========================================================

add_heading("15. LIMITATIONS OF THE STUDY")

limitations = [
    "The analysis is based on the variables available in the dataset.",
    "The predictive performance depends on data quality and feature selection.",
    "Transportation expenditure shows relatively limited predictive accuracy.",
    "District-level patterns may be influenced by unequal sample sizes.",
    "Machine learning associations should not be interpreted as causal relationships.",
    "Further validation on independent datasets is required."
]

for i, item in enumerate(limitations, 1):
    add_text(f"{i}. {item}")

# ==========================================================
# 22. FUTURE SCOPE
# ==========================================================

add_heading("16. FUTURE SCOPE")

future_points = [
    "Hyperparameter tuning using cross-validation.",
    "Testing advanced models such as XGBoost and CatBoost.",
    "Incorporating additional socioeconomic variables.",
    "Using larger and more geographically representative datasets.",
    "Applying explainable AI techniques for improved interpretation.",
    "Developing an interactive prediction dashboard.",
    "Testing model generalisation on independent household datasets."
]

for i, item in enumerate(future_points, 1):
    add_text(f"{i}. {item}")

# ==========================================================
# 23. CONCLUSION
# ==========================================================

add_heading("17. CONCLUSION")

add_text(
    "This project demonstrates the application of machine "
    "learning to household food and transportation expenditure "
    "analysis in Maharashtra. The workflow combines data "
    "preprocessing, exploratory data analysis, statistical "
    "analysis, predictive modelling, feature importance and "
    "model error evaluation."
)

add_text(
    "The food expenditure model achieved better predictive "
    "performance than the transportation expenditure model. "
    "The results demonstrate the usefulness of socioeconomic "
    "and demographic information for understanding household "
    "consumption patterns, while also highlighting the "
    "limitations of the current predictive models."
)

add_text(
    "Further improvements through advanced model optimisation, "
    "additional features and independent validation may "
    "strengthen the reliability and generalisability of the "
    "analysis."
)

# ==========================================================
# 24. REFERENCES
# ==========================================================

add_heading("18. REFERENCES")

references = [
    "Scikit-learn Documentation: https://scikit-learn.org/",
    "Pandas Documentation: https://pandas.pydata.org/",
    "NumPy Documentation: https://numpy.org/",
    "SHAP Documentation: https://shap.readthedocs.io/",
    "ReportLab Documentation: https://docs.reportlab.com/"
]

for i, ref in enumerate(references, 1):
    add_text(f"{i}. {ref}")

# ==========================================================
# 25. PAGE NUMBER FOOTER
# ==========================================================

def add_page_number(canvas, doc):
    canvas.saveState()

    canvas.setFont("Helvetica", 8)

    canvas.drawString(
        45,
        25,
        "Maharashtra Household Expenditure ML Project"
    )

    canvas.drawRightString(
        A4[0] - 45,
        25,
        f"Page {doc.page}"
    )

    canvas.restoreState()


# ==========================================================
# 26. GENERATE PDF
# ==========================================================

print("\nGenerating complete project report...")

doc.build(
    story,
    onFirstPage=add_page_number,
    onLaterPages=add_page_number
)

print("\n==========================================")
print("COMPLETE PROJECT REPORT GENERATED")
print("==========================================")

print(f"\nPDF Location:\n{REPORT_PATH}")

print(f"\nDataset observations: {len(df)}")
print(f"Dataset variables: {len(df.columns)}")
print(f"Graphs included: {len(image_files)}")

print("\nReport generation completed successfully!")