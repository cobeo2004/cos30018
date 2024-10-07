import pandas as pd
import matplotlib.pyplot as plt


def plot_actual_vs_predicted(
    actual_df: pd.DataFrame,  # DataFrame containing actual stock prices
    predicted_df: pd.DataFrame,  # DataFrame containing predicted stock prices
    ticker: str  # Stock ticker symbol
):
    """
    Plot actual vs predicted stock prices.

    Args:
        actual_df (pd.DataFrame): DataFrame containing actual stock prices.
            Must have a 'Close' column and a DatetimeIndex.
        predicted_df (pd.DataFrame): DataFrame containing predicted stock prices.
            Must have a 'Predicted_Close' column and a DatetimeIndex.
        ticker (str): Stock ticker symbol for the plot title.
    """
    plt.figure(figsize=(12, 6))

    # Plot actual prices
    print("Actual DF Index Type: ", type(actual_df.index))
    print("Predicted DF Index Type: ", type(predicted_df.index))
    print("Actual DF Index Value: ", actual_df.index.values)
    print("Predicted DF Index Value: ", predicted_df.index.values)
    plt.plot(actual_df.index, actual_df['Close'], label='Actual', color='blue')

    # Plot predicted prices
    plt.plot(predicted_df.index,
             predicted_df['Predicted_Close'], label='Predicted', color='red', linestyle='--')

    plt.title(f'{ticker} Stock Price: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Rotate and align the tick labels so they look better
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming 'df' is your DataFrame with actual prices and 'predict_df' is from your prediction function
# plot_actual_vs_predicted(df.tail(30), predict_df, ticker)
