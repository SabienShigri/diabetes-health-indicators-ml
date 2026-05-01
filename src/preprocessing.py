import pandas as pd

FEATURES = [
    "age",
    "bmi",
    "glucose_fasting",
    "hba1c",
    "systolic_bp",
    "diastolic_bp"
]

def load_data(path):
    return pd.read_csv(path)

def select_features(df):
    return df[FEATURES]

def split_targets(df):
    y_binary = df["diagnosed_diabetes"]
    y_multi = df["diabetes_stage"]
    y_reg = df["diabetes_risk_score"]
    
    return y_binary, y_multi, y_reg