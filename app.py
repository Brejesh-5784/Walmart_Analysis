import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils.preprocess import load_and_engineer_data

# --- Page Configuration ---
st.set_page_config(page_title="Weekly Sales Dashboard", layout="wide", page_icon="📈")

# --- Title ---
st.title("📊 Walmart Weekly Sales Analysis & Prediction")

# --- Load Model and Dataset ---
model = pickle.load(open("models/xgboost_model.pkl", "rb"))
df = load_and_engineer_data("data/Walmart_Sales.csv")
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# --- Sidebar Input ---
st.sidebar.header("🧾 Prediction Input")
store = st.sidebar.number_input("Store ID", 1, 50, step=1)
date = st.sidebar.date_input("Date", value=datetime.today())
holiday_flag = st.sidebar.selectbox("Holiday", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
temperature = st.sidebar.slider("Temperature (°F)", 20.0, 120.0, 70.0)
fuel_price = st.sidebar.slider("Fuel Price ($)", 1.0, 5.0, 2.5)
cpi = st.sidebar.slider("CPI", 100.0, 300.0, 200.0)
unemployment = st.sidebar.slider("Unemployment Rate", 2.0, 15.0, 7.0)

# --- Feature Engineering ---
date_obj = pd.to_datetime(date)
year = date_obj.year
month = date_obj.month
week = date_obj.isocalendar().week
day = date_obj.day
is_weekend = 1 if date_obj.weekday() >= 5 else 0

input_data = pd.DataFrame([{
    'Store': store,
    'Holiday_Flag': holiday_flag,
    'Temperature': temperature,
    'Fuel_Price': fuel_price,
    'CPI': cpi,
    'Unemployment': unemployment,
    'Year': year,
    'Month': month,
    'Week': week,
    'Day': day,
    'Is_Weekend': is_weekend
}])

# --- Prediction ---
if st.sidebar.button("🚀 Predict Weekly Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"### Predicted Weekly Sales: **${prediction:,.2f}**")
    st.dataframe(input_data.style.highlight_max(axis=1, color='lightgreen'))

    fig_input = px.bar(input_data.T.reset_index(), x='index', y=0,
                       labels={"index": "Feature", "0": "Value"},
                       title="🔧 Input Feature Overview")
    st.plotly_chart(fig_input, use_container_width=True)

# --- Analysis Section ---
st.markdown("---")
st.header("📈 Exploratory Visualizations")

# -- 1. Line Chart: Weekly Sales Over Time (selected store)
st.subheader("📆 Sales Trend Over Time (Store-wise)")
store_selected = st.selectbox("Choose a Store", df['Store'].unique())
df_store = df[df['Store'] == store_selected]
fig_line = px.line(df_store, x='Date', y='Weekly_Sales',
                   title=f"Store {store_selected} Weekly Sales Trend",
                   labels={"Weekly_Sales": "Sales ($)"})
st.plotly_chart(fig_line, use_container_width=True)

# -- 2. Heatmap: Correlation
st.subheader("📊 Feature Correlation Heatmap")
corr = df.drop(columns=['Date']).corr()
fig_corr, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig_corr)

# -- 3. Histogram: Weekly Sales Distribution
st.subheader("📉 Weekly Sales Distribution")
fig_hist = px.histogram(df, x="Weekly_Sales", nbins=50, title="Distribution of Weekly Sales")
st.plotly_chart(fig_hist, use_container_width=True)

# -- 4. Box Plot: Sales vs Month
st.subheader("📦 Monthly Sales Pattern (Seasonality)")
fig_box = px.box(df, x="Month", y="Weekly_Sales", title="Sales by Month")
st.plotly_chart(fig_box, use_container_width=True)

# -- 5. Bar Chart: Average Sales per Store
st.subheader("🏬 Average Weekly Sales per Store")
avg_sales = df.groupby("Store")["Weekly_Sales"].mean().reset_index()
fig_bar = px.bar(avg_sales, x="Store", y="Weekly_Sales",
                 title="Average Weekly Sales by Store", color="Weekly_Sales")
st.plotly_chart(fig_bar, use_container_width=True)

# -- 6. Scatter Plot: CPI vs Weekly Sales
st.subheader("💹 CPI vs Weekly Sales")
fig_cpi = px.scatter(df, x="CPI", y="Weekly_Sales", color="Store",
                     title="CPI vs Weekly Sales", opacity=0.6)
st.plotly_chart(fig_cpi, use_container_width=True)

# -- 7. Scatter Plot: Unemployment vs Weekly Sales
st.subheader("📉 Unemployment vs Weekly Sales")
fig_unemp = px.scatter(df, x="Unemployment", y="Weekly_Sales", color="Store",
                       title="Unemployment vs Weekly Sales", opacity=0.6)
st.plotly_chart(fig_unemp, use_container_width=True)

# Footer
st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)
st.markdown("<center>Made with ❤️ by brejesh V.D. | Data Science Project</center>", unsafe_allow_html=True)
