import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.utils import IndexInstance
import numpy as np
from .split_data_by_ratio import split_data_by_ratio

def process_data(data: pd.DataFrame, feature_cols: list[str], look_back_days: int, split_ratio: float = 0.8, is_split_by_date: bool = True):
    feature_cols += ["Target", "TargetClass","TargetNextClose","Close_RSI", "Close_MACD", "Close_MACD_Sig", "GDP", "Inflation", "Unemployment"]
    data["Target"] = data["Adj Close"] - data["Open"]
    data["Target"] = data["Target"].shift(-1)
    data["TargetClass"] = np.where(data["Target"] > 0, 1, 0)
    data["TargetNextClose"] = data["Adj Close"].shift(-1)

    # Check for NaN values in Target and TargetNextClose and fill with 0
    data["Target"] = data["Target"].fillna(0)
    data["TargetNextClose"] = data["TargetNextClose"].fillna(0)

    data["Close_RSI"] = IndexInstance.calculateRSI(data)
    data["Close_RSI"] = data["Close_RSI"].fillna(0)
    dt, sig = IndexInstance.calculateMACD(data)
    data["Close_MACD"] = dt
    data["Close_MACD_Sig"] = sig

    scalers = {}

    for f in feature_cols:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(data[f].values.reshape(-1, 1))
        scalers[f] = scaler

    x_val, y_val = [], []

    for i in range(look_back_days, len(data)):
        x_val.append(data[feature_cols].iloc[i-look_back_days:i].values)
        y_val.append(data['TargetNextClose'].iloc[i])

    x_val, y_val = np.array(x_val), np.array(y_val)

    if is_split_by_date and split_ratio:
        ratio = int(len(data) * split_ratio)
        x_train, x_test = x_val[:ratio], x_val[ratio:]
        y_train, y_test = y_val[:ratio], y_val[ratio:]
    else:
        x_train, x_test, y_train, y_test = split_data_by_ratio(x_val, y_val, split_ratio, is_split_by_date)

    return data, x_train, x_test, y_train, y_test, scalers
