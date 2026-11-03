import streamlit as st
import pandas as pd
import joblib
import os

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Household Expenditure Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Maharashtra Household Expenditure Predictor")

st.write(
    "A Machine Learning application to predict "
    "monthly household food and transportation expenditure."
)

# =====================================
# LOAD DATA AND MODELS
# =====================================

DATA_PATH = "outputs/results/cleaned_household_data.csv"

FOOD_MODEL_PATH = "outputs/models/M_EXP_FOOD_model.joblib"
TRANSPORT_MODEL_PATH = "outputs/models/M_EXP_TRANSPORT_model.joblib"

if not all(os.path.exists(p) for p in [
    DATA_PATH, FOOD_MODEL_PATH, TRANSPORT_MODEL_PATH
]):
    st.error("Required dataset or model files are missing.")
    st.stop()

data = pd.read_csv(DATA_PATH)

food_model = joblib.load(FOOD_MODEL_PATH)
transport_model = joblib.load(TRANSPORT_MODEL_PATH)

# =====================================
# HOUSEHOLD INPUT
# =====================================

st.header("Enter Household Information")

income = st.number_input(
    "Monthly Household Income (₹)",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

district = st.selectbox(
    "District",
    sorted(data["DISTRICT"].dropna().unique())
)

age = st.selectbox(
    "Age Group",
    sorted(data["AGE_GROUP"].dropna().unique())
)

occupation = st.selectbox(
    "Occupation Group",
    sorted(data["OCCUPATION_GROUP"].dropna().unique())
)

education = st.selectbox(
    "Education Group",
    sorted(data["EDU_GROUP"].dropna().unique())
)

gender = st.selectbox(
    "Gender Group",
    sorted(data["GENDER_GROUP"].dropna().unique())
)

size = st.selectbox(
    "Household Size Group",
    sorted(data["SIZE_GROUP"].dropna().unique())
)

# =====================================
# PREDICTION
# =====================================

if st.button("Predict Monthly Expenditure"):

    input_data = pd.DataFrame([{
        "TOT_INCOME": income,
        "DISTRICT": district,
        "AGE_GROUP": age,
        "OCCUPATION_GROUP": occupation,
        "EDU_GROUP": education,
        "GENDER_GROUP": gender,
        "SIZE_GROUP": size
    }])

    food_prediction = food_model.predict(input_data)[0]

    transport_prediction = transport_model.predict(input_data)[0]

    # Prevent negative expenditure display
    food_prediction = max(0, food_prediction)
    transport_prediction = max(0, transport_prediction)

    total_prediction = food_prediction + transport_prediction

    # =====================================
    # DISPLAY RESULTS
    # =====================================

    st.divider()

    st.header("Predicted Monthly Expenditure")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Food Expenditure",
            f"₹{food_prediction:,.2f}"
        )

    with col2:
        st.metric(
            "Transportation Expenditure",
            f"₹{transport_prediction:,.2f}"
        )

    st.success(
        f"Combined Predicted Expenditure: ₹{total_prediction:,.2f}"
    )

    st.caption(
        "Note: These are machine-learning estimates, "
        "not guaranteed actual household expenses."
    )

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "Maharashtra Household Expenditure Analysis | "
    "Machine Learning Project"
)
# =====================================
# PROJECT DASHBOARD
# =====================================

st.divider()

st.header("📊 Project Analytics Dashboard")

# Dataset overview
st.subheader("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Households", f"{len(data):,}")

with col2:
    st.metric(
        "Number of Districts",
        data["DISTRICT"].nunique()
    )

with col3:
    st.metric(
        "Average Monthly Income",
        f"₹{data['TOT_INCOME'].mean():,.0f}"
    )

# District-wise expenditure
st.subheader("2. District-wise Average Expenditure")

district_data = (
    data.groupby("DISTRICT")[
        ["M_EXP_FOOD", "M_EXP_TRANSPORT"]
    ]
    .mean()
    .sort_values("M_EXP_FOOD", ascending=False)
)

st.bar_chart(district_data)

# Model performance
st.subheader("3. Model Performance")

performance_path = "outputs/results/final_model_performance.csv"

if os.path.exists(performance_path):

    performance = pd.read_csv(performance_path)

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        performance.set_index("Target")["R2"]
    )

else:
    st.info("Model performance report is not available yet.")

st.caption(
    "Developed using Python, Pandas, Scikit-learn and Streamlit."
)