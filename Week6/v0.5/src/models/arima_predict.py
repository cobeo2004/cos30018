import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing_extensions import Tuple
import numpy as np


def arima_predict(train_data: pd.DataFrame, test_data: pd.DataFrame, order: Tuple[int, int, int] = (1, 1, 1)) -> np.ndarray:
    hist = train_data['Close'].tolist()
    predictions = hist.copy()

    print("Starting ARIMA predictions...")

    for t in range(len(test_data)):
        model = ARIMA(hist, order=order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output.iloc[0] if isinstance(output, pd.Series) else output[0]
        predictions.append(yhat)
        obs = test_data.iloc[t]['Close']
        hist.append(obs)
        print(f"Prediction {t+1}/{len(test_data)} completed")

    print("ARIMA predictions finished")
    return np.array(predictions)
