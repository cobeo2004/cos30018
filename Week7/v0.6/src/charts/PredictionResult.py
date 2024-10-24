from .IChart import IChart
import numpy as np
import matplotlib.pyplot as plt
from src.processing import ticker


class PredictionResultChart(IChart):
    def plot(
        self,
        y_test: np.ndarray,
        xgBoostPrediction: np.ndarray,
        rfPrediction: np.ndarray,
    ) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(y_test, label="Actual", color="red")
        plt.plot(xgBoostPrediction, label="XGBoost", color="blue")
        plt.plot(rfPrediction, label="Random Forest", color="green")
        plt.title(f"Share Price Prediction of {ticker}")
        plt.xlabel("Time")
        plt.ylabel("Share Price")
        plt.legend(loc="best")
        plt.grid(True)

        plt.ylim(
            [
                min(y_test.min(), xgBoostPrediction.min(), rfPrediction.min()),
                max(y_test.max(), xgBoostPrediction.max(), rfPrediction.max()),
            ]
        )
        plt.show()
