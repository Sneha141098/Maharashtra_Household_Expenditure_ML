# ==========================================
# STAGE 3: EXPLORATORY DATA ANALYSIS
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("\n===== STAGE 3: EDA =====")

# Load cleaned dataset

df = pd.read_csv(
    "outputs/results/cleaned_household_data.csv"
)

os.makedirs("outputs/graphs", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)

sns.set_theme(style="whitegrid")

# Descriptive statistics

print("\nDESCRIPTIVE STATISTICS")

print(
    df[
        ["TOT_INCOME", "M_EXP_FOOD", "M_EXP_TRANSPORT"]
    ].describe()
)

# Save descriptive statistics

df[
    ["TOT_INCOME", "M_EXP_FOOD", "M_EXP_TRANSPORT"]
].describe().to_csv(
    "outputs/results/descriptive_statistics.csv"
)

# Income Distribution

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="TOT_INCOME",
    bins=40,
    kde=True
)

plt.title("Distribution of Household Income")
plt.xlabel("Monthly Household Income")
plt.ylabel("Number of Households")

plt.savefig(
    "outputs/graphs/income_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="M_EXP_TRANSPORT",
    bins=40,
    kde=True
)

plt.title("Distribution of Transportation Expenditure")
plt.xlabel("Monthly Transportation Expenditure")
plt.ylabel("Number of Households")

plt.savefig(
    "outputs/graphs/transport_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="M_EXP_FOOD",
    bins=40,
    kde=True
)

plt.title("Distribution of Food Expenditure")
plt.xlabel("Monthly Food Expenditure")
plt.ylabel("Number of Households")

plt.savefig(
    "outputs/graphs/food_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="TOT_INCOME",
    y="M_EXP_TRANSPORT",
    alpha=0.5
)

plt.title("Household Income vs Transportation Expenditure")
plt.xlabel("Monthly Household Income")
plt.ylabel("Monthly Transportation Expenditure")

plt.savefig(
    "outputs/graphs/income_vs_transport.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="TOT_INCOME",
    y="M_EXP_FOOD",
    alpha=0.5
)

plt.title("Household Income vs Food Expenditure")
plt.xlabel("Monthly Household Income")
plt.ylabel("Monthly Food Expenditure")

plt.savefig(
    "outputs/graphs/income_vs_food.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

from matplotlib.backends.backend_pdf import PdfPages

# Combine all graphs into one PDF

with PdfPages("outputs/EDA_Analysis.pdf") as pdf:

    for graph in [
        "income_distribution.png",
        "transport_distribution.png",
        "food_distribution.png",
        "income_vs_transport.png",
        "income_vs_food.png"
    ]:

        img = plt.imread("outputs/graphs/" + graph)

        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(graph.replace(".png", "").replace("_", " "))

        pdf.savefig(bbox_inches="tight")
        plt.close()

print("EDA PDF generated successfully!")