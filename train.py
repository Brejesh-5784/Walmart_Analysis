from utils.preprocess import load_and_engineer_data
from utils.train_model import train_and_save_model

# Load and engineer the dataset
df = load_and_engineer_data("/Users/brejesh/Downloads/Walmart_Sales.csv")

# Train and save model to models/xgboost_model.pkl
train_and_save_model(df)
