import streamlit as st
import pandas as pd
import joblib
import plotly.express as px



st.set_page_config(
    page_title="Retail Sales Forecasting Dashboard",
    layout="wide"
)


st.markdown("""
<style>

.stApp{
    background-color:#0F172A;
    color:white;
}

/* Make all text visible */
h1,h2,h3,h4,h5,h6,p,label,span,div{
    color:white !important;
}

/* Sidebar */
[data-testid="stSidebar"] * {
    background-color:#E2E8F0;
    color: black !important;
}

/* KPI cards style */
.kpi{
    background:#1E293B;
    padding:20px;
    border-radius:15px;
    border:1px solid #334155;
    text-align:center;
}

/* Buttons */
.stButton>button{
    background:#2563EB;
    color:white;
    border-radius:10px;
    height:50px;
    width:100%;
    font-size:16px;
}

</style>
""", unsafe_allow_html=True)


model = joblib.load("Streamlit/Sales_model.pkl")


df = pd.read_csv("../data/Walmart.csv")


st.title("📊 Retail Sales Forecasting Dashboard")
st.caption("AI Powered Business Intelligence System")

st.markdown("---")


total_sales = float(df["Weekly_Sales"].sum())
avg_sales = float(df["Weekly_Sales"].mean())
total_stores = int(df["Store"].nunique())

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Sales", f"₹ {total_sales:,.0f}")

with col2:
    st.metric("Average Weekly Sales", f"₹ {avg_sales:,.0f}")

with col3:
    st.metric("Total Stores", total_stores)

st.markdown("---")


st.sidebar.header("Forecast Inputs")

store = st.sidebar.number_input("Store", 1, 50, 1)
holiday = st.sidebar.selectbox("Holiday Flag", [0, 1])
temperature = st.sidebar.number_input("Temperature", value=60.0)
fuel_price = st.sidebar.number_input("Fuel Price", value=3.0)
cpi = st.sidebar.number_input("CPI", value=200.0)
unemployment = st.sidebar.number_input("Unemployment", value=8.0)
month = st.sidebar.slider("Month", 1, 12, 1)


st.subheader("🔮 Sales Prediction")

if st.button("Generate Forecast"):

    input_data = pd.DataFrame({
        "Store": [store],
        "Holiday_Flag": [holiday],
        "Temperature": [temperature],
        "Fuel_Price": [fuel_price],
        "CPI": [cpi],
        "Unemployment": [unemployment],
        "Month": [month]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Weekly Sales: ₹ {prediction[0]:,.2f}")

    st.metric("Forecasted Sales", f"₹ {prediction[0]:,.0f}")

st.markdown("---")


st.subheader("📈 Sales Trend Over Time")

sales_trend = df.groupby("Date")["Weekly_Sales"].sum().reset_index()

fig1 = px.line(
    sales_trend,
    x="Date",
    y="Weekly_Sales",
    title="Sales Trend"
)

fig1.update_layout(
    paper_bgcolor="#0F172A",
    plot_bgcolor="#0F172A",
    font_color="white"
)

st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")


st.subheader("📊 Feature Importance")

features = [
    "Store",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Month"
]

importance = model.feature_importances_

fig2 = px.bar(
    x=importance,
    y=features,
    orientation="h",
    color=importance,
    color_continuous_scale="Blues",
    title="Feature Importance"
)

fig2.update_layout(
    paper_bgcolor="#0F172A",
    plot_bgcolor="#0F172A",
    font_color="white"
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")


r2 = 0.9408
mae = 71813.87

st.subheader("🤖 Model Performance")

col4, col5 = st.columns(2)

with col4:
    st.metric("R² Score", f"{r2:.4f}")

with col5:
    st.metric("MAE", f"{mae:,.2f}")

st.markdown("---")
st.caption("Project Made by Pratima")

