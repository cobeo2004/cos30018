# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


COMPANY = 'CBA.AX'

TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2024-01-01'       # End date to read

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2024, 1, 1)

# data = web.DataReader(company, 'yahoo', start, end)
data = yf.download(COMPANY, TRAIN_START, TRAIN_END)

# Prep data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

pred_days = 60
x_train, y_train = [], []

for x in range(pred_days, len(scaled_data)):
    x_train.append(scaled_data[x-pred_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# LSTM Neural Network Builder
model = Sequential()
# LSTM layer with 50 neurons and return sequences
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# Dropout layer to prevent overfitting
model.add(Dropout(0.2))
# LSTM layer with 50 neurons and return sequences
model.add(LSTM(units=50, return_sequences=True))
# Dropout layer to prevent overfitting
model.add(Dropout(0.2))
# LSTM layer with 50 neurons
model.add(LSTM(units=50))
# Dropout layer to prevent overfitting
model.add(Dropout(0.2))
# Output layer, Predicting the next closest value
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
# Training the model
model.fit(x_train, y_train, epochs=25, batch_size=32)

'''TESTING THE MODEL'''

# Test Data Loading
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'

test_data = yf.download(COMPANY,TEST_START,TEST_END)
actual_prices = test_data['Close'].values

total_ds = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_ds[len(total_ds) - len(test_data) - pred_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

# Making predictions on test data
x_test = []
for x in range(pred_days, len(model_inputs)):
    x_test.append(model_inputs[x - pred_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Prediciting the prices
predicted_prices = model.predict(x_test)
# Inverse transform to get the actual prices
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the test predictions
plt.plot(actual_prices, label=f'Actual {COMPANY} Prices', color='black')
plt.plot(predicted_prices, label=f'Predicted {COMPANY} Prices', color='green')
plt.title(f'{COMPANY} Share Price')
plt.xlabel('Time')
plt.ylabel(f'{COMPANY} Share Price')
plt.legend()
plt.show()

'''Next day prediction'''
real_data = [model_inputs[len(model_inputs) + 1 - pred_days:len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

pred = model.predict(real_data)
pred = scaler.inverse_transform(pred)
print("Prediction: ", pred)
