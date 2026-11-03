import pandas as pd
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
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

print("\n===== STAGE 8: FINAL PROJECT REPORT =====")

# ------------------------------------------------
# 1. LOAD RESULTS
# ------------------------------------------------

results = pd.read_csv(
    "outputs/results/model_comparison.csv"
)

importance = pd.read_csv(
    "outputs/results/feature_importance.csv"
)

data = pd.read_csv(
    "outputs/results/cleaned_household_data.csv"
)

os.makedirs("outputs", exist_ok=True)

pdf_path = "outputs/Final_Maharashtra_Household_Expenditure_Report.pdf"

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

# ------------------------------------------------
# 2. TITLE
# ------------------------------------------------

story.append(Spacer(1, 70))

story.append(Paragraph(
    "MACHINE LEARNING-BASED ANALYSIS OF HOUSEHOLD "
    "EXPENDITURE AND CONSUMPTION PATTERNS",
    styles["Title"]
))

story.append(Spacer(1, 20))

story.append(Paragraph(
    "A Study of Urban Households in Maharashtra",
    styles["Heading2"]
))

story.append(Spacer(1, 40))

story.append(Paragraph(
    "Machine Learning Project Report",
    styles["Heading2"]
))

story.append(Spacer(1, 15))

story.append(Paragraph(
    "Study Area: Maharashtra, India",
    styles["BodyText"]
))

story.append(PageBreak())

# ------------------------------------------------
# 3. ABSTRACT
# ------------------------------------------------

story.append(Paragraph("ABSTRACT", styles["Heading1"]))

abstract = """
This study applies machine learning techniques to examine household
expenditure patterns among urban households in Maharashtra. The
analysis focuses on monthly food and transportation expenditure.

The dataset contains 10,567 household observations and includes
income, district, age group, occupation, education, gender composition
and household size.

Six machine learning algorithms were evaluated using an 80:20
training-testing split. Their predictive performance was assessed
using R-squared, Mean Absolute Error and Root Mean Squared Error.

The best-performing models identified in the current evaluation were
Linear Regression for transportation expenditure and Bagging
Regression for food expenditure. Their test R-squared scores were
0.2542 and 0.5904, respectively.

Permutation importance identified district as the most influential
predictive feature for both expenditure categories, followed by
household income. Household size also contributed substantially to
food expenditure prediction.
"""

story.append(Paragraph(abstract, styles["BodyText"]))
story.append(Spacer(1, 15))

# ------------------------------------------------
# 4. INTRODUCTION
# ------------------------------------------------

story.append(Paragraph("1. INTRODUCTION", styles["Heading1"]))

intro = """
Household expenditure is an important indicator of economic
well-being and consumption behaviour. Understanding how socioeconomic
and demographic characteristics relate to household spending can
help identify differences in consumption patterns.

This project investigates food and transportation expenditure among
urban households in Maharashtra using statistical analysis and
machine learning techniques.
"""

story.append(Paragraph(intro, styles["BodyText"]))

# ------------------------------------------------
# 5. OBJECTIVES
# ------------------------------------------------

story.append(Paragraph("2. RESEARCH OBJECTIVES", styles["Heading1"]))

objectives = """
1. Examine household food and transportation expenditure patterns.<br/>
2. Investigate relationships between income and expenditure.<br/>
3. Predict household expenditure using machine learning.<br/>
4. Compare six regression algorithms.<br/>
5. Identify important socioeconomic and demographic predictors.
"""

story.append(Paragraph(objectives, styles["BodyText"]))

# ------------------------------------------------
# 6. DATASET
# ------------------------------------------------

story.append(Paragraph("3. DATASET DESCRIPTION", styles["Heading1"]))

dataset_info = [
    ["Parameter", "Description"],
    ["Total observations", str(len(data))],
    ["Geographical coverage", "24 districts of Maharashtra"],
    ["Region", "Urban households"],
    ["Numerical predictor", "Total household income"],
    ["Categorical predictors",
     "District, age, occupation, education, gender, household size"],
    ["Targets",
     "Monthly food and transportation expenditure"]
]

table = Table(dataset_info, colWidths=[150, 330])

table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("PADDING", (0, 0), (-1, -1), 7),
    ("VALIGN", (0, 0), (-1, -1), "TOP")
]))

story.append(table)

# ------------------------------------------------
# 7. METHODOLOGY
# ------------------------------------------------

story.append(Paragraph("4. METHODOLOGY", styles["Heading1"]))

methodology = """
The analysis was conducted in several stages:

1. Data integration: Household income and expenditure datasets
were merged using household ID.

2. Data cleaning: Missing values, duplicate records, data types
and unusual observations were examined.

3. Exploratory analysis: Distributions and relationships between
income and expenditure were investigated.

4. Statistical analysis: Descriptive statistics, Pearson
correlation and Spearman correlation were calculated.

5. Machine learning preparation: Numerical and categorical
predictors were identified. Categorical variables were one-hot
encoded and numerical variables were standardised within the
training pipeline.

6. Model evaluation: Six regression algorithms were compared
using an 80:20 train-test split.

7. Feature importance: Permutation importance was calculated
using the selected best-performing model for each target.
"""

story.append(Paragraph(methodology, styles["BodyText"]))

# ------------------------------------------------
# 8. ML MODELS
# ------------------------------------------------

story.append(Paragraph("5. MACHINE LEARNING MODELS", styles["Heading1"]))

models_text = """
The following algorithms were evaluated:

1. Linear Regression<br/>
2. Decision Tree Regressor<br/>
3. Random Forest Regressor<br/>
4. Gradient Boosting Regressor<br/>
5. Bootstrap Aggregation (Bagging Regressor)<br/>
6. Support Vector Regression
"""

story.append(Paragraph(models_text, styles["BodyText"]))

# ------------------------------------------------
# 9. MODEL RESULTS
# ------------------------------------------------

story.append(Paragraph(
    "6. MODEL PERFORMANCE COMPARISON",
    styles["Heading1"]
))

result_table = [
    ["Target", "Model", "R2", "MAE", "RMSE"]
]

for _, row in results.iterrows():

    result_table.append([
        row["Target"],
        row["Model"],
        f'{row["R2_Score"]:.4f}',
        f'{row["MAE"]:.2f}',
        f'{row["RMSE"]:.2f}'
    ])

table = Table(
    result_table,
    colWidths=[75, 145, 55, 85, 85],
    repeatRows=1
)

table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("PADDING", (0, 0), (-1, -1), 6)
]))

story.append(table)

# ------------------------------------------------
# 10. FEATURE IMPORTANCE
# ------------------------------------------------

story.append(Paragraph(
    "7. FEATURE IMPORTANCE ANALYSIS",
    styles["Heading1"]
))

for target in importance["Target"].unique():

    target_data = importance[
        importance["Target"] == target
    ].sort_values(
        "Importance_Mean",
        ascending=False
    )

    story.append(Paragraph(
        target + " Expenditure",
        styles["Heading2"]
    ))

    best_model = target_data["Best_Model"].iloc[0]

    story.append(Paragraph(
        f"Selected model: {best_model}",
        styles["BodyText"]
    ))

    for _, row in target_data.iterrows():

        story.append(Paragraph(
            f'{row["Feature"]}: '
            f'{row["Importance_Mean"]:.5f}',
            styles["BodyText"]
        ))

    story.append(Spacer(1, 10))

# ------------------------------------------------
# 11. GRAPHS
# ------------------------------------------------

story.append(PageBreak())

story.append(Paragraph(
    "8. VISUAL RESULTS",
    styles["Heading1"]
))

graphs = [
    "r2_comparison.png",
    "mae_comparison.png",
    "rmse_comparison.png",
    "transportation_feature_importance.png",
    "food_feature_importance.png"
]

for graph in graphs:

    path = f"outputs/graphs/{graph}"

    if os.path.exists(path):

        story.append(Image(
            path,
            width=6.2 * inch,
            height=3.5 * inch
        ))

        story.append(Spacer(1, 12))

# ------------------------------------------------
# 12. RESULTS AND DISCUSSION
# ------------------------------------------------

story.append(Paragraph(
    "9. RESULTS AND DISCUSSION",
    styles["Heading1"]
))

discussion = """
The transportation expenditure model achieved a test R-squared
of 0.2542 using Linear Regression. This indicates comparatively
limited predictive performance.

The food expenditure model achieved a test R-squared of 0.5904
using Bagging Regression, indicating better predictive performance
than the transportation model.

District was the highest-ranked feature for both expenditure
categories. Total household income ranked second in both cases.
Household size was particularly relevant to food expenditure.

These findings indicate that geographical and economic
characteristics are important predictors in the evaluated models.
However, predictive importance does not establish causality.
"""

story.append(Paragraph(discussion, styles["BodyText"]))

# ------------------------------------------------
# 13. LIMITATIONS
# ------------------------------------------------

story.append(Paragraph("10. LIMITATIONS", styles["Heading1"]))

limitations = """
1. The analysis is limited to the available urban household sample
from Maharashtra.

2. Transportation expenditure has a highly right-skewed distribution.

3. The transportation prediction model has relatively low R-squared.

4. Important variables such as local prices, transport infrastructure,
distance travelled and household consumption preferences are not
included in the available dataset.

5. Model performance is based on a single train-test split.

6. Feature importance indicates predictive relevance rather than
causal influence.
"""

story.append(Paragraph(limitations, styles["BodyText"]))

# ------------------------------------------------
# 14. CONCLUSION
# ------------------------------------------------

story.append(Paragraph("11. CONCLUSION", styles["Heading1"]))

conclusion = """
This project demonstrates the application of statistical methods
and machine learning algorithms to household expenditure analysis
in urban Maharashtra.

Among the evaluated models, Linear Regression performed best for
transportation expenditure, while Bagging Regression performed
best for food expenditure.

District emerged as the most important predictive feature for both
targets. Income was also important, while household size had a
substantial contribution to food expenditure prediction.

The findings highlight differences in the predictability of
household food and transportation spending and provide a foundation
for more detailed socioeconomic consumption research.
"""

story.append(Paragraph(conclusion, styles["BodyText"]))

# ------------------------------------------------
# 15. FUTURE SCOPE
# ------------------------------------------------

story.append(Paragraph("12. FUTURE SCOPE", styles["Heading1"]))

future = """
Future research can incorporate local price indices, fuel prices,
public transport availability, household assets and detailed
consumption characteristics.

Additional improvements could include cross-validation,
hyperparameter optimisation, target transformation for highly
skewed expenditure variables and external validation using
independent household datasets.
"""

story.append(Paragraph(future, styles["BodyText"]))

# ------------------------------------------------
# 16. REFERENCES
# ------------------------------------------------

story.append(Paragraph("13. REFERENCES", styles["Heading1"]))

references = """
1. Scikit-learn documentation: Machine Learning in Python.<br/>
2. Pandas documentation: Data analysis and manipulation.<br/>
3. NumPy documentation: Numerical computing in Python.<br/>
4. Matplotlib and Seaborn documentation: Data visualisation.<br/>
5. Original household income and expenditure Excel datasets
provided for this project.
"""

story.append(Paragraph(references, styles["BodyText"]))

# ------------------------------------------------
# 17. BUILD REPORT
# ------------------------------------------------

doc.build(story)

print("\nFinal report generated successfully!")
print("Saved at:", pdf_path)