import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from src.processing.constants import ticker


def plot_candlestick_full(train_df, test_df, predicted_prices, n=1, price_value='Close'):
    # Create deep copies to avoid modifying the original dataframes
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Resampling the data for n trading days
    if n > 1:
        train_df = train_df.resample(f'{n}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        test_df = test_df.resample(f'{n}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        if train_df.empty or test_df.empty:
            raise ValueError(
                "Resampling resulted in an empty DataFrame. Try a smaller value of n.")

        # Adjust the length of predicted_prices to match test_df
        eff_length = len(test_df)
        predicted_prices = predicted_prices[-eff_length:]

    # Compute moving averages for the training data
    train_df['MA50'] = train_df[price_value].rolling(window=50).mean()
    train_df['MA100'] = train_df[price_value].rolling(window=100).mean()
    train_df['MA200'] = train_df[price_value].rolling(window=200).mean()

    # Compute moving averages for the test data
    test_df['MA50'] = test_df[price_value].rolling(window=50).mean()
    test_df['MA100'] = test_df[price_value].rolling(window=100).mean()
    test_df['MA200'] = test_df[price_value].rolling(window=200).mean()

    # Check if predicted_prices is 2D and reshape if necessary
    if predicted_prices.ndim == 2:
        predicted_prices = predicted_prices.reshape(-1)

    # Ensure the length of predicted_prices matches the length of the test data
    if len(predicted_prices) != len(test_df):
        raise ValueError(
            f"Length mismatch: predicted_prices has length {len(predicted_prices)} but test_df has length {len(test_df)}")

    # Add predicted prices to the test dataframe
    test_df['Predicted'] = predicted_prices

    # Concatenate train and test dataframes to form a complete dataframe for plotting
    df_plot = pd.concat([train_df, test_df])

    # Convert the index to a DatetimeIndex
    df_plot.index = pd.to_datetime(df_plot.index)

    # Create a custom plot for the predicted prices and moving averages
    ap = []
    if df_plot['MA50'].dropna().shape[0] > 0:
        aligned_MA50 = df_plot['MA50'].dropna().reindex(
            df_plot.index, fill_value=None)
        ap.append(mpf.make_addplot(aligned_MA50, color='orange', label='MA50'))

    if df_plot['MA100'].dropna().shape[0] > 0:
        aligned_MA100 = df_plot['MA100'].dropna().reindex(
            df_plot.index, fill_value=None)
        ap.append(mpf.make_addplot(aligned_MA100,
                  color='green', label='MA100'))

    if df_plot['MA200'].dropna().shape[0] > 0:
        aligned_MA200 = df_plot['MA200'].dropna().reindex(
            df_plot.index, fill_value=None)
        ap.append(mpf.make_addplot(aligned_MA200,
                  color='magenta', label='MA200'))

    ap.append(mpf.make_addplot(
        df_plot['Predicted'], color='red', linestyle='dashed', label='Predicted'))

    # Plot the candlestick chart
    fig, axes = mpf.plot(df_plot, type='candle', style='charles',
                         title=f"{ticker} Candlestick Chart",
                         ylabel='Price',
                         volume=False,
                         addplot=ap,
                         show_nontrading=False,
                         returnfig=True)

    # Add legend
    axes[0].legend(['MA50', 'MA100', 'MA200', 'Predicted'])

    plt.show()
