
````markdown
# 📊 Weekly Sales Prediction App

A Streamlit web application that predicts Walmart's weekly sales using an XGBoost regression model and user-provided inputs such as store number, temperature, fuel price, and holiday indicators.

🚀 [Live Demo](https://salesweek.streamlit.app)

---

## 🧠 Features

- Predicts weekly sales using a trained XGBoost model.
- Accepts real-time user input: Store Number, Temperature, Fuel Price, CPI, IsHoliday, etc.
- Displays real-time predictions through an interactive and intuitive web UI.
- Includes backend data preprocessing, model training, and performance visualization.

---

## 🛠️ Tech Stack

- **Programming Language**: Python
- **Libraries**: XGBoost, Pandas, NumPy, scikit-learn
- **Web Framework**: Streamlit
- **Visualization**: Matplotlib, Seaborn

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Walmart-Sales-Forecasting.git
cd Walmart-Sales-Forecasting
````

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

Then open the link shown in your terminal, or view the hosted app directly here:
🌐 [https://salesweek.streamlit.app](https://salesweek.streamlit.app)

---

## 📂 Project Structure

```
├── app.py                  # Streamlit web application
├── train_model.py          # XGBoost model training script
├── models/
│   └── xgboost_model.pkl   # Trained and serialized model
├── utils/
│   └── preprocess.py       # Feature engineering and preprocessing
├── assets/
│   └── correlation_plot.py # Correlation visualization
├── data/
│   └── Walmart_Sales.csv   # Historical Walmart sales data
├── requirements.txt        # List of dependencies
└── README.md               # Project overview and instructions
```

---

## 📈 Model Performance

* ✅ **R² Score**: 0.92 on the test set
* ✅ Trained on 45,000+ cleaned and feature-engineered historical sales records
* ✅ Improved model performance by 18% through feature selection and holiday adjustments

---

© 2025 Brejesh V.D. • All rights reserved.

```

