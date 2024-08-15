# Stock Prediction Program
# Predicting stock prices using Long Short-Term Memory (LSTM) Recurrent Neural Network using TensorFlow and Keras
# Author: Abdeladim Fadheil
# Adopter & Type Annotator: Simon Nguyen
# Updated: Apr 2024
# Version: P1
# Tested environment: Python 3.11
# Sees: https://thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras

# Importing essential libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from collections import deque
from numpy.typing import ArrayLike
from typing import List, Tuple, Dict, Any
from typing_extensions import Annotated, Doc
from sklearn.preprocessing import MinMaxScaler
from numpy.typing import NDArray
import os
import time
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Setting up seeds
RANDOM_SEED = 314
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Function definitions


def shuffleInUnison(a: Annotated[ArrayLike, Doc("the first array to shuffle")], b: Annotated[ArrayLike, Doc("the second array to shuffle")]) -> None:
    assert len(a) == len(b)
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def loadData(ticker: Annotated[str, Doc("the ticker you want to load, examples include AAPL, TESL, etc.")],
             n_steps: Annotated[int, Doc(
                 "the historical sequence length (i.e window size) used to predict, default is 50")] = 50,
             scale: Annotated[bool, Doc(
                 "whether to scale prices from 0 to 1, default is True")] = True,
             shuffle: Annotated[bool, Doc(
                 "whether to shuffle the dataset (both training & testing), default is True")] = True,
             lookup_step: Annotated[int, Doc(
                 "the future lookup step to predict, default is 1 (e.g next day)")] = 1,
             split_by_date: Annotated[bool, Doc(
                 "whether we split the dataset into training/testing by date, setting it to False will split datasets in a random way")] = True,
             test_size: Annotated[float, Doc(
                 "ratio for test data, default is 0.2 (20% testing data)")] = 0.2,
             feature_columns: Annotated[List[str], Doc("the list of features to use to feed into the model, default is everything grabbed from yahoo_fin")] = ['adjclose', 'volume', 'open', 'high', 'low']):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        split_by_date (bool): whether we split the dataset into training/testing by date, setting it to False will split datasets in a random way
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    # Check if ticker is already a loaded stock from yahoo_fin
    if isinstance(ticker, str):
        # If a string -> load data from Yahoo Finance
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # If a DataFrame -> use it
        df = ticker
    else:
        raise TypeError(
            "Invalid input type for ticker. Please provide a string or a pandas DataFrame.")

    # A dictionary to store the returned values from this function
    result: Dict[str, Any] = {}
    # Add a copy dataframe to the result
    result['df'] = df.copy()
    # Ensure that the feature_columns are in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"Error: '{col}' not found in the DataFrame columns."
    # Add date as a column to the dataframe
    if "date" not in df.columns:
        df['date'] = df.index
    if scale:
        column_scaler: Dict[str, Any] = {}
        # Scale the data (prices) from 0 to 1
        for col in feature_columns:
            scaler: MinMaxScaler = preprocessing.MinMaxScaler()
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
            column_scaler[col] = scaler
        # Add the MinMaxScaler to the result
        result["column_scaler"] = column_scaler
    # Add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)
    # Since `lookup_step` columns contains NaN in future column -> get em beffore dropping NaNs
    last_sequence: NDArray | list[NDArray] = np.array(
        df[feature_columns].tail(n_steps))
    # Drop NaNs
    df.dropna(inplace=True)
    sequence_data: List[List[NDArray]] = []
    sequences: deque = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # Get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # For example, suppose n_step = 50 and lookup_step = 1, then the last sequence should be 60 (50 + 10) length
    # This last sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)]
                         for s in sequences]) + list(last_sequence)
    # Add the last sequence to the result
    result['last_sequence'] = last_sequence
    # Construct the X's and Y's
    X:  NDArray = []
    Y:  NDArray = []
    for seq, target in sequence_data:
        X.append(seq)
        Y.append(target)
    # Convert the lists to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    if split_by_date:
        # If split_by_date is True, split the dataset into training & testing by date (not randomly)
        train_samples: int = int((1 - test_size) * len(X))
        result['X_train'] = X[:train_samples]
        result['X_test'] = X[train_samples:]
        result['Y_train'] = Y[train_samples:]
        result['Y_test'] = Y[train_samples:]
        if shuffle:
            shuffleInUnison(result['X_train'], result['Y_train'])
            shuffleInUnison(result['X_test'], result['Y_test'])
    else:
        # If split_by_date is False, split the dataset randomly
        result["X_train"], result["X_test"], result["Y_train"], result["Y_test"] = train_test_split(
            X, Y, test_size=test_size, shuffle=shuffle)

    # Get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # Retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # Remove duplicated dates in the tesiting dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(
        keep='first')]
    # Remove dates from the training / testing sets and convert to float32 arr
    result["X_train"] = result["X_train"][:, :,
                                          :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :,
                                        :len(feature_columns)].astype(np.float32)

    return result


def createModel(sequence_length: Annotated[int, Doc("the historical sequence length (i.e window size) used to predict, default is 50")] = 50,
                n_features: Annotated[int, Doc(
                    "the number of features to use to feed into the model, default is 5")] = 5,
                units: Annotated[int, Doc(
                    "the number of RNN cell units, default is 50")] = 50,
                n_layers: Annotated[int, Doc(
                    "the number of RNN layers, default is 1")] = 1,
                cell: Annotated[tf.keras.layers.Layer, Doc(
                    "the RNN cell to use, default is LSTM")] = LSTM,
                dropout: Annotated[float, Doc(
                    "the dropout rate, default is 0.2")] = 0.2,
                optimizer: Annotated[str, Doc(
                    "the optimizer to use, default is rmsprop")] = "rmsprop",
                loss: Annotated[str, Doc(
                    "the loss function to use, default is mean_absolute_error")] = "mean_absolute_error",
                bidirectional: Annotated[bool, Doc("whether to use bidirectional RNNs, default is False")] = False) -> Sequential:
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True),
                          input_shape=(sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True,
                          input_shape=(sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=[
                  "mean_absolute_error"], optimizer=optimizer)
    return model


def plot_graph(test_df: Annotated[pd.DataFrame, Doc("the testing dataframe")]):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df[f"true_adjclose_{LOOKUP_STEP}"], c='b')
    plt.plot(test_df[f"adjclose_{LOOKUP_STEP}"], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


def getFinalDF(model: Annotated[Sequential, Doc("the model to use")], data: Annotated[Dict[str, Any], Doc("the data to use")]) -> pd.DataFrame:
    """
    This function takes the model and the data that was returned by `createModel()` and `loadData()` functions respectively, and constructs a dataframe that includes the predicted adjclose along with true future adjclose, as well as calculating buy and sell profit.
    """
    # if predicted future price > current -> calculate the true future price - current price, to get the buy profit
    def buy_profit(current, pred_future, true_future): return true_future - \
        current if pred_future > current else 0
    # if predicted future price < current -> calculate the current price - true future price, to get the sell profit
    def sell_profit(current, pred_future, true_future): return current - \
        true_future if pred_future < current else 0
    # Perform prediction and get prices
    X_test = data["X_test"]
    Y_test = data["Y_test"]
    Y_Predicted = model.predict(X_test)
    if SCALE:
        Y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(
            np.expand_dims(Y_test, axis=0)))
        Y_Predicted = np.squeeze(
            data["column_scaler"]["adjclose"].inverse_transform(Y_Predicted))

    test_df = data["test_df"]
    # Add predicted future prices to the dataframe
    test_df[f"adjclose_{LOOKUP_STEP}"] = Y_Predicted
    # Add true future prices to the dataframe
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = Y_test
    # Sort dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df
    # Add the buy profit columns
    final_df["buy_profit"] = list(map(buy_profit, final_df["adjclose"],
                                  final_df[f"adjclose_{LOOKUP_STEP}"], final_df[f"true_adjclose_{LOOKUP_STEP}"]))
    # Add the sell profit columns
    final_df["sell_profit"] = list(map(sell_profit, final_df["adjclose"],
                                   final_df[f"adjclose_{LOOKUP_STEP}"], final_df[f"true_adjclose_{LOOKUP_STEP}"]))
    return final_df


def predict(model: Annotated[Sequential, Doc("the model to use")], data: Annotated[Dict[str, Any], Doc("the data to use")]):
    """
    This function takes the model and the data that was returned by `createModel()` and `loadData()` functions respectively, and predicts the next future price.
    """
    # Retrieve the last sequence from data
    last_sequence = data["last_sequence"][-N_STEPS:]
    # Expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # Get the prediction (scaled from 0 -> 1)
    prediction = model.predict(last_sequence)
    # Get the price value by inverting the scaling
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[
            0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price


# Main execution
if __name__ == "__main__":
    # Window size or the sequence length
    N_STEPS = 50
    # Lookup step, 1 is the next day
    LOOKUP_STEP = 15
    # whether to scale feature columns & output price as well
    SCALE = True
    scale_str = f"sc-{int(SCALE)}"
    # whether to shuffle the dataset
    SHUFFLE = True
    shuffle_str = f"sh-{int(SHUFFLE)}"
    # whether to split the training/testing set by date
    SPLIT_BY_DATE = False
    split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
    # test ratio size, 0.2 is 20%
    TEST_SIZE = 0.2
    # features to use
    FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
    # date now
    date_now = time.strftime("%Y-%m-%d")
    # model parameters
    N_LAYERS = 2
    # LSTM cell
    CELL = LSTM
    # 256 LSTM neurons
    UNITS = 256
    # 40% dropout
    DROPOUT = 0.4
    # whether to use bidirectional RNNs
    BIDIRECTIONAL = False
    # training parameters
    # mean absolute error loss
    # LOSS = "mae"
    # huber loss
    LOSS = "huber"
    OPTIMIZER = "adam"
    BATCH_SIZE = 64
    EPOCHS = 200
    # Amazon stock market
    ticker = "AMZN"
    ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
    # model name to save, making it as unique as possible based on parameters
    model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
    {LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
    if BIDIRECTIONAL:
        model_name += "-b"

    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("data"):
        os.makedirs("data")

    # Load the data
    data = loadData(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE, shuffle=SHUFFLE,
                    lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)
    data['df'].to_csv(ticker_data_filename)
    # Create the model
    model = createModel(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, optimizer=OPTIMIZER, cell=CELL,
                        dropout=DROPOUT, n_layers=N_LAYERS, units=UNITS, bidirectional=BIDIRECTIONAL)
    # Some callback function for tensorboard
    checkpointer = ModelCheckpoint(os.path.join(
        "results", model_name + ".weights.h5"), save_weights_only=True, save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))
    # Train the model and save the weights whenever we see a new optimal model using ModalCheckpoint
    history = model.fit(data["X_train"], data["Y_train"], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(
        data["X_test"], data["Y_test"]), callbacks=[checkpointer, tensorboard], verbose=1)

    model_path = os.path.join("results", model_name) + ".weights.h5"
    model.load_weights(model_path)

    loss, mae = model.evaluate(data["X_test"], data["Y_test"], verbose=0)
    # Inverse scaling to calculate the the mean absolute error
    if SCALE:
        mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[
            0][0]
    else:
        mean_absolute_error = mae
    print(f"Mean Absolute Error: {mean_absolute_error}")

    final_df = getFinalDF(model, data)
    future_price = predict(model, data)

    # Calculate the accuracy by counting the number of positive profits
    accuracy_score = (len(final_df[final_df["sell_profit"] > 0]) +
                      len(final_df[final_df["buy_profit"] > 0])) / len(final_df)

    # Total buy and sell profit
    total_buy_profit = final_df["buy_profit"].sum()
    total_sell_profit = final_df["sell_profit"].sum()

    # Total profit of both buy and sell
    total_profit = total_buy_profit + total_sell_profit

    # Number of profit per trades
    profit_per_trade = total_profit / len(final_df)

    # printing metrics
    print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
    print(f"{LOSS} loss:", loss)
    print("Mean Absolute Error:", mean_absolute_error)
    print("Accuracy score:", accuracy_score)
    print("Total buy profit:", total_buy_profit)
    print("Total sell profit:", total_sell_profit)
    print("Total profit:", total_profit)
    print("Profit per trade:", profit_per_trade)
