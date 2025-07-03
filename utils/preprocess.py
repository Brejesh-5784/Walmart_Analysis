import pandas as pd

def load_and_engineer_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Day'] = df['Date'].dt.day
    df['Is_Weekend'] = df['Date'].dt.dayofweek >= 5
    df = df.drop(columns=['Date'])
    return df
