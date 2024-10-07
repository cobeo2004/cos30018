from src.processing import make_datasets, ENTRY_POINT, ticker, ensemble_model_predict
from src.models import model_gru_config, train_model, save_model, model_lstm_config, model_rnn_config, arima_predict
from src.charts import metric_plot, plot_arima_param, plot_predicted_vs_actual, plot_candlestick_full, plot_dl_arima_ensemble_real, plot_dl_arima_real
from tensorflow.keras.models import load_model
data, df, train_data, test_data, train_feature_scale, train_target_scale, x_train, x_test, y_train, y_test = make_datasets()

# plot_arima_param(df)

n_steps = x_train.shape[1]
n_features = x_train.shape[2]

# rnn_model, rnn_mts = train_model(
#     model_rnn_config, (n_steps, n_features), x_train, y_train)

# save_model(rnn_model, ENTRY_POINT + "/rnn_model.keras")
rnn_model = load_model(ENTRY_POINT + "/rnn_model.keras")
rnn_y_pred = rnn_model.predict(x_test)
rnn_y_pred = train_target_scale.inverse_transform(rnn_y_pred)

# lstm_model, lstm_mts = train_model(
#     model_lstm_config, (n_steps, n_features), x_train, y_train)

# save_model(lstm_model, ENTRY_POINT + "/lstm_model.keras")
lstm_model = load_model(ENTRY_POINT + "/lstm_model.keras")

lstm_y_pred = lstm_model.predict(x_test)
lstm_y_pred = train_target_scale.inverse_transform(lstm_y_pred)

# gru_model, gru_mts = train_model(
#     model_gru_config, (n_steps, n_features), x_train, y_train)

# save_model(gru_model, ENTRY_POINT + "/gru_model.keras")
gru_model = load_model(ENTRY_POINT + "/gru_model.keras")

gru_y_pred = gru_model.predict(x_test)
gru_y_pred = train_target_scale.inverse_transform(gru_y_pred)
print(len(gru_y_pred))

# truncated_test_data = test_data.iloc[-len(y_pred):]
# plot_candlestick_full(train_data, truncated_test_data, y_pred, n=4)

arima_res = arima_predict(x_train, x_test, order=(2, 5, 1))
# ensembled = ensemble_model_predict(arima_res, gru_y_pred)
print("Arima res: ", len(arima_res))
# print("Ensembled: ", ensembled)
# plot_predicted_vs_actual(test_data, arima_res, ticker)
# plot_dl_arima_ensemble_real(df, lstm_y_pred, arima_res,
#                             ensemble_model_predict(arima_res, lstm_y_pred))

# plot_dl_arima_ensemble_real(df, gru_y_pred, arima_res,
#                             ensemble_model_predict(arima_res, gru_y_pred))

# plot_dl_arima_real(df, gru_y_pred, arima_res)
# plot_dl_arima_real(df, lstm_y_pred, arima_res)
# plot_dl_arima_real(df, rnn_y_pred, arima_res)
