import pandas as pd
from .IChart import IChart
import numpy as np
import matplotlib.pyplot as plt


class RSIandMACDChart(IChart):
    def plot(self, data: pd.DataFrame) -> None:
        rsi_val = data["Close_RSI"]
        macd_val = data["Close_MACD"]
        theta_val = np.linspace(0, 2 * np.pi, len(rsi_val))

        plt.figure(figsize=(10, 5))
        ax = plt.subplot(111, projection="polar")
        plt.plot(theta_val, rsi_val, label="RSI", color="red")
        plt.plot(theta_val, macd_val, label="MACD", color="blue")
        plt.title("RSI and MACD")
        plt.legend()
        plt.show()
