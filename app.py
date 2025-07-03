import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils.preprocess import load_and_engineer_data

# --- Streamlit Page Config ---
st.set_page_config(page_title="Weekly Sales Dashboard", layout="wide", page_icon="📈")

# --- Custom Theme Colors ---
primary_color = "#6C63FF"
bg_color = "#F7F9FB"
accent_color = "#FF6B6B"
success_color = "#00C49F"
text_color = "#333333"

# --- Title Header with Background ---
st.markdown(f"""
    <div style='background-color:{bg_color}; padding: 20px; border-radius: 10px;'>
        <h1 style='text-align: center; color: {primary_color};'>📊 Walmart Weekly Sales Analysis & Forecast</h1>
        <p style='text-align: center; color: {text_color}; font-size:18px;'>
            Uncover insights, predict sales, and visualize trends across Walmart stores
        </p>
    </div>
    <hr style='border:1px solid {primary_color};'>
""", unsafe_allow_html=True)

# --- Load Model and Data ---
model = pickle.load(open("models/xgboost_model.pkl", "rb"))
df = load_and_engineer_data("data/Walmart_Sales.csv")
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# --- Sidebar Input with Emojis ---
with st.sidebar.expander("📥 Input Features for Prediction", expanded=True):
    store = st.number_input("🏪 Store ID", 1, 50, step=1)
    date = st.date_input("🗕️ Select Date", value=datetime.today())
    holiday_flag = st.selectbox("🎉 Is it a Holiday?", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
    temperature = st.slider("🌡️ Temperature (°F)", 20.0, 120.0, 70.0)
    fuel_price = st.slider("⛽ Fuel Price ($)", 1.0, 5.0, 2.5)
    cpi = st.slider("📈 CPI", 100.0, 300.0, 200.0)
    unemployment = st.slider("💼 Unemployment Rate (%)", 2.0, 15.0, 7.0)

# --- Feature Engineering for Prediction ---
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
    st.markdown(f"<h3 style='color:{success_color}'>💰 Predicted Weekly Sales: ${prediction:,.2f}</h3>", unsafe_allow_html=True)

    st.dataframe(input_data.style.highlight_max(axis=1, color='#B5F5EC'))

    fig_input = px.bar(input_data.T.reset_index(), x='index', y=0,
                       labels={"index": "Feature", "0": "Value"},
                       title="🔧 Input Feature Overview",
                       template="plotly")
    st.plotly_chart(fig_input, use_container_width=True)

# --- Data Exploration Section ---
st.markdown("---")
st.header("🎨 Exploratory Visualizations")

# -- 1. Sales Trend Over Time --
st.subheader("📆 Store-wise Weekly Sales Trend")
store_selected = st.selectbox("🎢 Choose a Store to View Trend", df['Store'].unique())
df_store = df[df['Store'] == store_selected]
fig_line = px.line(df_store, x='Date', y='Weekly_Sales',
                   title=f"📈 Weekly Sales Trend for Store {store_selected}",
                   labels={"Weekly_Sales": "Sales ($)"},
                   template="plotly_dark",
                   color_discrete_sequence=[accent_color])
st.plotly_chart(fig_line, use_container_width=True)

# -- 2. Correlation Heatmap --
st.subheader("📊 Feature Correlation Heatmap")
corr = df.drop(columns=['Date']).corr()
fig_corr, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig_corr)

# -- 3. Histogram of Weekly Sales --
st.subheader("📉 Weekly Sales Distribution")
fig_hist = px.histogram(df, x="Weekly_Sales", nbins=50,
                        title="Histogram: Weekly Sales Distribution",
                        template="seaborn",
                        color_discrete_sequence=[primary_color])
st.plotly_chart(fig_hist, use_container_width=True)

# -- 4. Monthly Seasonality with Box Plot --
st.subheader("📦 Seasonal Pattern - Sales by Month")
fig_box = px.box(df, x="Month", y="Weekly_Sales",
                 title="Monthly Distribution of Weekly Sales",
                 template="ggplot2",
                 color_discrete_sequence=[accent_color])
st.plotly_chart(fig_box, use_container_width=True)

# -- 5. Bar Chart: Avg Weekly Sales per Store --
st.subheader("🏪 Store Performance: Avg Weekly Sales")
avg_sales = df.groupby("Store")["Weekly_Sales"].mean().reset_index()
fig_bar = px.bar(avg_sales, x="Store", y="Weekly_Sales",
                 title="Average Weekly Sales by Store",
                 color="Weekly_Sales", template="plotly_dark",
                 color_continuous_scale='blues')
st.plotly_chart(fig_bar, use_container_width=True)

# -- 6. Scatter Plot: CPI vs Weekly Sales --
st.subheader("📈 CPI vs Weekly Sales")
fig_cpi = px.scatter(df, x="CPI", y="Weekly_Sales", color="Store",
                     title="Effect of CPI on Sales", opacity=0.7,
                     template="simple_white")
st.plotly_chart(fig_cpi, use_container_width=True)

# -- 7. Scatter Plot: Unemployment vs Weekly Sales --
st.subheader("📉 Unemployment vs Weekly Sales")
fig_unemp = px.scatter(df, x="Unemployment", y="Weekly_Sales", color="Store",
                       title="Effect of Unemployment on Sales", opacity=0.7,
                       template="simple_white")
st.plotly_chart(fig_unemp, use_container_width=True)



# --- Footer ---
st.markdown("<hr style='border:1px solid #aaa;'>", unsafe_allow_html=True)
st.markdown(f"""
    <center style='color:#888;'>
     By <b>Brejesh V.D.</b><br>
        <span style='font-size: 14px;'>Powered by Streamlit, Plotly & XGBoost</span>
    </center>
""", unsafe_allow_html=True)
