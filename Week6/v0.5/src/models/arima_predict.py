import pandas as pd
from typing_extensions import Tuple
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_squared_error


def arima_predict(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    p_range: Tuple[int, int] = (0, 5),
    d_range: Tuple[int, int] = (0, 2),
    q_range: Tuple[int, int] = (0, 5),
    m: int = 7,
    seasonal: bool = True,
) -> Tuple[np.ndarray, float]:
    # Extracting the Close price from train and test data
    train_close = train_data["Close"]
    test_close = test_data["Close"]

    # Combine train and test data for ARIMA modeling
    # full_data = pd.concat([train_data, test_data])

    # ARIMA modelling
    model = pm.auto_arima(
        train_close,
        start_p=p_range[0],
        start_q=q_range[0],
        max_p=p_range[1],
        max_q=q_range[1],
        start_P=0,
        start_Q=0,
        max_P=5,
        max_Q=5,
        m=m,
        d=d_range[0],
        max_d=d_range[1],
        seasonal=seasonal,
        trace=True,
        error_action="warn",
        suppress_warnings=True,
        stepwise=True,
        random_state=42,
        n_fits=50,
    )
    model.summary()

    # Predictions
    predictions = model.predict(n_periods=len(test_close))

    # Calculate RMSE
    rmse = np.sqrt(np.mean((predictions - test_close.values) ** 2))
    print(f"RMSE ARIMA: {rmse}")

    return predictions, rmse


def ensemble_predict(dl_pred, arima_pred, dl_rmse, arima_rmse, test_data):
    dl_pred = dl_pred.flatten()

    overlap_size = min(len(dl_pred), len(arima_pred))
    arima_pred = arima_pred[:overlap_size]
    test_data = test_data[:overlap_size]

    print(f"DL after slicing overlap: {dl_pred}")
    print(f"ARIMA after slicing overlap: {arima_pred}")
    print(f"Test data after slicing overlap: {test_data}")

    # Calculate the average of the predictions
    dl_pred_np = np.array(dl_pred)
    arima_pred_np = np.array(arima_pred)
    avg_pred = (dl_pred_np + arima_pred_np) / 2
    print(f"Average predictions: {avg_pred}")

    # Replace NaN with 0
    test_data = np.nan_to_num(test_data, nan=0.0)
    print(f"Test data after replacing NaN: {test_data}")

    # Check NaN
    print(f"DL predictions NaN: {np.isnan(dl_pred).any()}")
    print(f"ARIMA predictions NaN: {np.isnan(arima_pred).any()}")
    print(f"Test data NaN: {np.isnan(test_data).any()}")

    # Calculate ensembled RMSE
    ensembled_rmse = np.sqrt(mean_squared_error(test_data, avg_pred, squared=False))
    print(f"Ensembled RMSE: {ensembled_rmse}")

    # Calculate average RMSE
    avg_rmse = (dl_rmse + arima_rmse) / 2
    print(f"Average RMSE: {avg_rmse}")

    return avg_pred, ensembled_rmse, avg_rmse
