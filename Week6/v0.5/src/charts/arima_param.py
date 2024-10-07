from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from typing_extensions import Annotated, Doc
import pandas as pd


def plot_arima_param(data: Annotated[pd.DataFrame, Doc("The data to be plotted")]) -> None:
    """
    Plot the ACF and PACF of the data to determine the parameters for ARIMA model.

    Args:
        data (pd.DataFrame): The input data to be plotted.

    Returns:
        None
    """

    # Finding p value (AR order)
    plot_pacf(data['Close'])
    plt.show()

    # Finding d value (Differencing order)
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(data['Close'])
    ax1.set_title('Original Data')
    ax1.axes.xaxis.set_visible(False)
    ax2.plot(data['Close'].diff())
    ax2.set_title('1st Order Differencing')
    ax2.axes.xaxis.set_visible(False)
    ax3.plot(data['Close'].diff().diff())
    ax3.set_title('2nd Order Differencing')
    plt.show()

    # Finding q value (MA order)
    plot_acf(data['Close'].diff().dropna())
    plt.show()
