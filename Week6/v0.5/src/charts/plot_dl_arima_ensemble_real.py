import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_dl_arima_ensemble_real(df: pd.DataFrame, deep_learning_predictions: np.ndarray, arima_predictions: np.ndarray, ensemble_predictions: np.ndarray):
    """
    Plot actual prices and predictions from deep learning, ARIMA, and ensemble models.

    Args:
        df (pd.DataFrame): DataFrame containing actual stock prices with a 'Close' column and DatetimeIndex.
        deep_learning_predictions (np.ndarray): Array of deep learning model predictions.
        arima_predictions (np.ndarray): Array of ARIMA model predictions.
        ensemble_predictions (np.ndarray): Array of ensemble model predictions.
    """
    plt.figure(figsize=(12, 6))

    # Plot actual prices
    plt.plot(df.index, df['Close'], label='Actual Prices', color='blue')

    # Plot deep learning predictions
    plt.plot(df.index[-len(deep_learning_predictions):], deep_learning_predictions,
             label='Deep Learning Predictions', color='green', linestyle='--')

    # Plot ARIMA predictions
    plt.plot(df.index[-len(arima_predictions):], arima_predictions,
             label='ARIMA Predictions', color='red', linestyle='--')

    # Plot ensemble predictions
    plt.plot(df.index[-len(ensemble_predictions):], ensemble_predictions,
             label='Ensemble Predictions', color='purple', linestyle='--')

    plt.title('Ensemble Model Results')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Rotate and align the tick labels so they look better
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show()

# Example usage:
# plot_dl_arima_ensemble_real(df, deep_learning_predictions, arima_predictions, ensemble_predictions)


def plot_dl_arima_real(df: pd.DataFrame, deep_learning_predictions: np.ndarray, arima_predictions: np.ndarray):
    plt.figure(figsize=(16, 8))  # Increased figure size for better visibility

    # Plot actual prices
    plt.plot(df.index, df['Close'], label='Actual Prices',
             color='blue', linewidth=2)

    # Plot deep learning predictions
    plt.plot(df.index[-len(deep_learning_predictions):], deep_learning_predictions,
             label='Deep Learning Predictions', color='green', linestyle='--', linewidth=2)

    # Plot ARIMA predictions
    plt.plot(df.index[-len(arima_predictions):], arima_predictions,
             label='ARIMA Predictions', color='red', linestyle=':', linewidth=2)

    plt.title('Deep Learning and ARIMA Model Results', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Rotate and align the tick labels so they look better
    plt.gcf().autofmt_xdate()

    # Add some padding to the plot
    plt.margins(x=0.01)

    # Customize tick parameters
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.show()
