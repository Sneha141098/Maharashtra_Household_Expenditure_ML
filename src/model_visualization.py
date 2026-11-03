import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------
# 1. LOAD MODEL RESULTS
# ---------------------------------------------

print("\n===== STAGE 6: MODEL PERFORMANCE ANALYSIS =====")

df = pd.read_csv("outputs/results/model_comparison.csv")

os.makedirs("outputs/graphs", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

sns.set_theme(style="whitegrid")

print("\nAvailable Models:")
print(df["Model"].unique())


# ---------------------------------------------
# 2. R2 SCORE COMPARISON
# ---------------------------------------------

plt.figure(figsize=(13, 7))

sns.barplot(
    data=df,
    x="Model",
    y="R2_Score",
    hue="Target"
)

plt.title("R2 Score Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("R2 Score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    "outputs/graphs/r2_comparison.png",
    dpi=300
)

plt.close()


# ---------------------------------------------
# 3. MAE COMPARISON
# ---------------------------------------------

plt.figure(figsize=(13, 7))

sns.barplot(
    data=df,
    x="Model",
    y="MAE",
    hue="Target"
)

plt.title("MAE Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("Mean Absolute Error (INR)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    "outputs/graphs/mae_comparison.png",
    dpi=300
)

plt.close()


# ---------------------------------------------
# 4. RMSE COMPARISON
# ---------------------------------------------

plt.figure(figsize=(13, 7))

sns.barplot(
    data=df,
    x="Model",
    y="RMSE",
    hue="Target"
)

plt.title("RMSE Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("Root Mean Squared Error (INR)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    "outputs/graphs/rmse_comparison.png",
    dpi=300
)

plt.close()


# ---------------------------------------------
# 5. IDENTIFY BEST MODEL
# ---------------------------------------------

print("\n===== BEST MODEL FOR EACH TARGET =====")

best_models = []

for target in df["Target"].unique():

    target_df = df[df["Target"] == target]

    # Highest R2 score
    best_r2 = target_df.loc[
        target_df["R2_Score"].idxmax()
    ]

    # Lowest MAE
    best_mae = target_df.loc[
        target_df["MAE"].idxmin()
    ]

    # Lowest RMSE
    best_rmse = target_df.loc[
        target_df["RMSE"].idxmin()
    ]

    print(f"\nTarget: {target}")

    print(
        f"Best R2 Model: {best_r2['Model']} "
        f"({best_r2['R2_Score']:.4f})"
    )

    print(
        f"Best MAE Model: {best_mae['Model']} "
        f"({best_mae['MAE']:.2f})"
    )

    print(
        f"Best RMSE Model: {best_rmse['Model']} "
        f"({best_rmse['RMSE']:.2f})"
    )

    best_models.append({
        "Target": target,
        "Best_R2_Model": best_r2["Model"],
        "Best_R2_Score": best_r2["R2_Score"],
        "Best_MAE_Model": best_mae["Model"],
        "Best_MAE": best_mae["MAE"],
        "Best_RMSE_Model": best_rmse["Model"],
        "Best_RMSE": best_rmse["RMSE"]
    })


# ---------------------------------------------
# 6. SAVE BEST MODEL SUMMARY
# ---------------------------------------------

summary_df = pd.DataFrame(best_models)

summary_df.to_csv(
    "outputs/results/best_model_summary.csv",
    index=False
)

print("\n===== FINAL PERFORMANCE SUMMARY =====")

print(summary_df.to_string(index=False))


# ---------------------------------------------
# 7. SAVE COMPLETE COMPARISON
# ---------------------------------------------

df.sort_values(
    by=["Target", "R2_Score"],
    ascending=[True, False]
).to_csv(
    "outputs/results/sorted_model_comparison.csv",
    index=False
)

print("\nStage 6 completed successfully!")

print("\nGraphs saved in outputs/graphs/")
print("Results saved in outputs/results/")

# ---------------------------------------------
# 8. GENERATE PDF SUMMARY REPORT
# ---------------------------------------------

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)

pdf_path = "outputs/Stage_6_Model_Performance_Report.pdf"

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

# TITLE
story.append(Paragraph(
    "Machine Learning-Based Analysis of Household Expenditure",
    styles["Title"]
))

story.append(Spacer(1, 12))

story.append(Paragraph(
    "Urban Households of Maharashtra",
    styles["Heading2"]
))

story.append(Spacer(1, 15))

# PROJECT OVERVIEW
story.append(Paragraph("1. Project Overview", styles["Heading1"]))

overview = """
This project investigates household expenditure patterns in urban
Maharashtra using machine learning techniques. It focuses on predicting
monthly transportation and food expenditure using socioeconomic and
demographic characteristics.
"""

story.append(Paragraph(overview, styles["BodyText"]))
story.append(Spacer(1, 12))

# DATASET DETAILS
story.append(Paragraph("2. Dataset Information", styles["Heading1"]))

dataset_data = [
    ["Parameter", "Details"],
    ["Total observations", "10,567 households"],
    ["Training observations", "8,453 (80%)"],
    ["Testing observations", "2,114 (20%)"],
    ["Numerical feature", "Total household income"],
    ["Categorical features",
     "District, age, occupation, education, gender, household size"],
    ["Prediction targets",
     "Monthly transportation and food expenditure"]
]

table = Table(dataset_data, colWidths=[150, 330])

table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("PADDING", (0, 0), (-1, -1), 7),
    ("VALIGN", (0, 0), (-1, -1), "TOP")
]))

story.append(table)
story.append(Spacer(1, 15))

# MODELS
story.append(Paragraph(
    "3. Machine Learning Models",
    styles["Heading1"]
))

model_text = """
1. Linear Regression<br/>
2. Decision Tree Regressor<br/>
3. Random Forest Regressor<br/>
4. Gradient Boosting Regressor<br/>
5. Bootstrap Aggregation (Bagging Regressor)<br/>
6. Support Vector Regression (SVR)
"""

story.append(Paragraph(model_text, styles["BodyText"]))
story.append(Spacer(1, 15))

# RESULTS TABLE
story.append(Paragraph(
    "4. Model Performance Comparison",
    styles["Heading1"]
))

results_data = [
    ["Target", "Model", "R²", "MAE", "RMSE"]
]

for _, row in df.sort_values(
    by=["Target", "R2_Score"],
    ascending=[True, False]
).iterrows():

    results_data.append([
        row["Target"],
        row["Model"],
        f'{row["R2_Score"]:.4f}',
        f'{row["MAE"]:.2f}',
        f'{row["RMSE"]:.2f}'
    ])

results_table = Table(
    results_data,
    colWidths=[75, 145, 55, 85, 85],
    repeatRows=1
)

results_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("PADDING", (0, 0), (-1, -1), 6),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
]))

story.append(results_table)
story.append(Spacer(1, 15))

# BEST MODELS
story.append(Paragraph(
    "5. Best Performing Models",
    styles["Heading1"]
))

for _, row in summary_df.iterrows():

    story.append(Paragraph(
        f'<b>{row["Target"]}:</b><br/>'
        f'Best R² model: {row["Best_R2_Model"]} '
        f'(R² = {row["Best_R2_Score"]:.4f})<br/>'
        f'Best MAE model: {row["Best_MAE_Model"]} '
        f'(MAE = {row["Best_MAE"]:.2f})<br/>'
        f'Best RMSE model: {row["Best_RMSE_Model"]} '
        f'(RMSE = {row["Best_RMSE"]:.2f})',
        styles["BodyText"]
    ))

    story.append(Spacer(1, 10))

# GRAPHS
story.append(PageBreak())

story.append(Paragraph(
    "6. Visual Model Comparison",
    styles["Heading1"]
))

for graph in [
    "r2_comparison.png",
    "mae_comparison.png",
    "rmse_comparison.png"
]:

    graph_path = f"outputs/graphs/{graph}"

    story.append(Image(
        graph_path,
        width=6.5 * inch,
        height=3.5 * inch
    ))

    story.append(Spacer(1, 12))

# CONCLUSION
story.append(Paragraph(
    "7. Conclusion",
    styles["Heading1"]
))

conclusion = """
The six machine learning algorithms were evaluated for predicting
monthly household transportation and food expenditure. R², MAE and
RMSE were used to compare predictive performance.

The best-performing model may differ between the two expenditure
categories. These results represent predictive associations and
should not be interpreted as evidence of causation.
"""

story.append(Paragraph(conclusion, styles["BodyText"]))

# BUILD PDF
doc.build(story)

print("\nPDF report generated successfully!")
print(f"Location: {pdf_path}")