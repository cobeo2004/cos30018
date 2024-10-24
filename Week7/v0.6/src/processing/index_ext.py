from .load_ext import load_data
from .process_ext import process_data
from .constants import start, end, ticker, num_look_back_days, FEATURE_COLS
import numpy as np
import pandas as pd

def prepare_data():
    raw_data = load_data(start=start, end=end, ticker=ticker)

    # Check for NaN values in raw_data
    if raw_data.isnull().values.any():
        print("Warning: NaN values found in raw_data")
        print("Columns with NaN values:")
        print(raw_data.columns[raw_data.isnull().any()].tolist())
        print("Number of NaN values in each column:")
        print(raw_data.isnull().sum())

    data, x_train, x_test, y_train, y_test, scalers = process_data(raw_data, FEATURE_COLS, num_look_back_days)

    if data.isnull().values.any():
        print("Warning: NaN values found in data")
        print("Columns with NaN values:")
        print(data.columns[data.isnull().any()].tolist())
        print("Number of NaN values in each column:")
        print(data.isnull().sum())

    return data, x_train, x_test, y_train, y_test, scalers
