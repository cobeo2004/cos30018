from src.processing import make_datasets, ENTRY_POINT, ticker
from src.models import (
    model_gru_config,
    train_model,
    save_model,
    model_lstm_config,
    model_rnn_config,
    arima_predict,
    predict_model,
    ensemble_predict,
)
from src.charts import plot_actual_vs_predicted, plot_arima_param
from tensorflow.keras.models import load_model
import os

(
    data,
    df,
    train_data,
    test_data,
    train_feature_scale,
    train_target_scale,
    x_train,
    x_test,
    y_train,
    y_test,
) = make_datasets()

# Get the number of steps and features
n_steps = x_train.shape[1]
n_features = x_train.shape[2]

# Plot ARIMA parameters
plot_arima_param(df)

# RNN model
rnn_model_path = os.path.join(ENTRY_POINT, "rnn_model.keras")
if os.path.exists(rnn_model_path):
    rnn_model = load_model(rnn_model_path)
else:
    rnn_model, _ = train_model(
        model_rnn_config, (n_steps, n_features), x_train, y_train
    )
    save_model(rnn_model, rnn_model_path)

rnn_y_pred, rnn_rmse = predict_model(rnn_model, train_target_scale, x_test, y_test)
print("RNN RMSE: ", rnn_rmse)
print("RNN Y Pred Sz: ", len(rnn_y_pred))
print("RNN Y Pred shape: ", rnn_y_pred.shape)

# LSTM model
lstm_model_path = os.path.join(ENTRY_POINT, "lstm_model.keras")
if os.path.exists(lstm_model_path):
    lstm_model = load_model(lstm_model_path)
else:
    lstm_model, _ = train_model(
        model_lstm_config, (n_steps, n_features), x_train, y_train
    )
    save_model(lstm_model, lstm_model_path)

lstm_y_pred, lstm_rmse = predict_model(lstm_model, train_target_scale, x_test, y_test)
print("LSTM RMSE: ", lstm_rmse)
print("LSTM Y Pred Sz: ", len(lstm_y_pred))
print("LSTM Y Pred shape: ", lstm_y_pred.shape)

# GRU model
gru_model_path = os.path.join(ENTRY_POINT, "gru_model.keras")
if os.path.exists(gru_model_path):
    gru_model = load_model(gru_model_path)
else:
    gru_model, _ = train_model(
        model_gru_config, (n_steps, n_features), x_train, y_train
    )
    save_model(gru_model, gru_model_path)

gru_y_pred, gru_rmse = predict_model(gru_model, train_target_scale, x_test, y_test)
print("GRU RMSE: ", gru_rmse)
print("GRU Y Pred Sz: ", len(gru_y_pred))
print("GRU Y Pred shape: ", gru_y_pred.shape)

arima_res, arima_rmse = arima_predict(train_data, test_data)
print("ARIMA res sz: ", len(arima_res))
print("ARIMA res shape: ", arima_res.shape)
print("ARIMA RMSE: ", arima_rmse)

# LSTM ensemble
lstm_avg_pred, lstm_ensemble_rmse, lstm_avg_rmse = ensemble_predict(
    lstm_y_pred,
    arima_res,
    lstm_rmse,
    arima_rmse,
    test_data["Close"],
)
print("LSTM Avg pred: ", lstm_avg_pred)
print("LSTM Ensemble RMSE: ", lstm_ensemble_rmse)
print("LSTM Avg RMSE: ", lstm_avg_rmse)

# GRU ensemble
gru_avg_pred, gru_ensemble_rmse, gru_avg_rmse = ensemble_predict(
    gru_y_pred,
    arima_res,
    gru_rmse,
    arima_rmse,
    test_data["Close"],
)
print("GRU Avg pred: ", gru_avg_pred)
print("GRU Ensemble RMSE: ", gru_ensemble_rmse)
print("GRU Avg RMSE: ", gru_avg_rmse)

# RNN ensemble
rnn_avg_pred, rnn_ensemble_rmse, rnn_avg_rmse = ensemble_predict(
    rnn_y_pred,
    arima_res,
    rnn_rmse,
    arima_rmse,
    test_data["Close"],
)
print("RNN Avg pred: ", rnn_avg_pred)
print("RNN Ensemble RMSE: ", rnn_ensemble_rmse)
print("RNN Avg RMSE: ", rnn_avg_rmse)

# Plot LSTM predictions
plot_actual_vs_predicted(
    train_data=train_data,
    test_data=test_data,
    lstm_predictions=lstm_y_pred,
    arima_predictions=arima_res,
    avg_predictions=lstm_avg_pred,
)

# Plot GRU predictions
plot_actual_vs_predicted(
    train_data=train_data,
    test_data=test_data,
    gru_predictions=gru_y_pred,
    arima_predictions=arima_res,
    avg_predictions=gru_avg_pred,
)

# Plot RNN predictions
plot_actual_vs_predicted(
    train_data=train_data,
    test_data=test_data,
    rnn_predictions=rnn_y_pred,
    arima_predictions=arima_res,
    avg_predictions=rnn_avg_pred,
)
