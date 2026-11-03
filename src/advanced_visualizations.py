import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
data = pd.read_csv(
    "outputs/results/cleaned_household_data.csv"
)

os.makedirs("outputs/graphs/advanced", exist_ok=True)

sns.set_theme(style="whitegrid")

# ======================================
# 1. CORRELATION HEATMAP
# ======================================

plt.figure(figsize=(10, 7))

numeric_data = data.select_dtypes(include="number")

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Household Expenditure Correlation")

plt.tight_layout()

plt.savefig(
    "outputs/graphs/advanced/correlation_heatmap.png"
)

plt.close()


# ======================================
# 2. INCOME VS FOOD EXPENDITURE
# ======================================

plt.figure(figsize=(9, 6))

sns.regplot(
    data=data,
    x="TOT_INCOME",
    y="M_EXP_FOOD",
    scatter_kws={"alpha": 0.3},
    line_kws={"color": "red"}
)

plt.title("Income vs Food Expenditure")

plt.tight_layout()

plt.savefig(
    "outputs/graphs/advanced/income_vs_food.png"
)

plt.close()


# ======================================
# 3. INCOME VS TRANSPORT EXPENDITURE
# ======================================

plt.figure(figsize=(9, 6))

sns.regplot(
    data=data,
    x="TOT_INCOME",
    y="M_EXP_TRANSPORT",
    scatter_kws={"alpha": 0.3},
    line_kws={"color": "red"}
)

plt.title("Income vs Transportation Expenditure")

plt.tight_layout()

plt.savefig(
    "outputs/graphs/advanced/income_vs_transport.png"
)

plt.close()


# ======================================
# 4. DISTRICT-WISE COMPARISON
# ======================================

district_data = data.groupby("DISTRICT")[
    ["M_EXP_FOOD", "M_EXP_TRANSPORT"]
].mean().sort_values(
    "M_EXP_FOOD",
    ascending=False
)

district_data.plot(
    kind="bar",
    figsize=(15, 7)
)

plt.title("District-wise Average Household Expenditure")

plt.ylabel("Monthly Expenditure (₹)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "outputs/graphs/advanced/district_comparison.png"
)

plt.close()


# ======================================
# 5. EXPENDITURE DISTRIBUTION
# ======================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.histplot(
    data["M_EXP_FOOD"],
    kde=True,
    ax=axes[0]
)

axes[0].set_title("Food Expenditure Distribution")

sns.histplot(
    data["M_EXP_TRANSPORT"],
    kde=True,
    ax=axes[1]
)

axes[1].set_title("Transportation Expenditure Distribution")

plt.tight_layout()

plt.savefig(
    "outputs/graphs/advanced/expenditure_distribution.png"
)

plt.close()


print("Advanced visualisations generated successfully!")