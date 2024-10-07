import numpy as np
import plotly.graph_objects as go
from src.processing import ticker

def plot_actual_vs_predicted(train_data, test_data, lstm_predictions=None, arima_predictions=None, rnn_predictions=None, gru_predictions=None, avg_predictions=None):
    # Concatenate the train and test data
    full_data = np.concatenate((train_data['Close'], test_data['Close']))

    # Create arrays for the predictions with NaNs for the training portion
    lstm_with_nan = np.concatenate((np.full(train_data['Close'].shape, np.nan), lstm_predictions.flatten())) if lstm_predictions is not None else None
    arima_with_nan = np.concatenate((np.full(train_data['Close'].shape, np.nan), arima_predictions)) if arima_predictions is not None else None
    rnn_with_nan = np.concatenate((np.full(train_data['Close'].shape, np.nan), rnn_predictions.flatten())) if rnn_predictions is not None else None
    gru_with_nan = np.concatenate((np.full(train_data['Close'].shape, np.nan), gru_predictions.flatten())) if gru_predictions is not None else None
    avg_with_nan = np.concatenate((np.full(train_data['Close'].shape, np.nan), avg_predictions)) if avg_predictions is not None else None

    # Plot using plotly.graph_objects
    fig = go.Figure()

    # Plotting the full historical data
    fig.add_trace(go.Scatter(x=list(range(len(full_data))),
                             y=full_data,
                             mode='lines',
                             name='Historical Data',
                             line=dict(color='blue')))

    if lstm_predictions is not None:
        # Overlaying the LSTM predictions
        fig.add_trace(go.Scatter(x=list(range(len(lstm_with_nan))),
                                 y=lstm_with_nan,
                                 mode='lines',
                                 name='LSTM Predictions',
                                 line=dict(color='green', dash='dash')))

    if arima_predictions is not None:
        # Overlaying the ARIMA predictions
        fig.add_trace(go.Scatter(x=list(range(len(arima_with_nan))),
                                 y=arima_with_nan,
                                 mode='lines',
                                 name='ARIMA Predictions',
                                 line=dict(color='red', dash='dash')))

    if rnn_predictions is not None:
        # Overlaying the RNN predictions
        fig.add_trace(go.Scatter(x=list(range(len(rnn_with_nan))),
                                 y=rnn_with_nan,
                                 mode='lines',
                                 name='RNN Predictions',
                                 line=dict(color='purple', dash='dash')))

    if gru_predictions is not None:
        # Overlaying the GRU predictions
        fig.add_trace(go.Scatter(x=list(range(len(gru_with_nan))),
                                 y=gru_with_nan,
                                 mode='lines',
                                 name='GRU Predictions',
                                 line=dict(color='pink', dash='dot')))

    if avg_predictions is not None:
        # Overlaying the averaged predictions
        fig.add_trace(go.Scatter(x=list(range(len(avg_with_nan))),
                                 y=avg_with_nan,
                                 mode='lines',
                                 name='Averaged Predictions',
                                 line=dict(color='orange', dash='dot')))

    # Formatting the plot
    fig.update_layout(title=f"{ticker} Historical vs Predictions",
                      xaxis_title="Timestep",
                      yaxis_title="Stock Price",
                      plot_bgcolor='#FFFFFF',
                      xaxis=dict(gridcolor='lightgrey'),
                      yaxis=dict(gridcolor='lightgrey'),
                      margin=dict(l=0, r=0, t=30, b=0),
                      legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"))

    fig.show()
