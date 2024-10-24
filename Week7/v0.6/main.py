if __name__ == "__main__":
    from src.processing.index_ext import prepare_data
    from src.model import XGBoostInstance, RandomForestInstance
    from src.processing.constants import FEATURE_COLS
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from src.charts import PredictionResultInstance, RSIandMACDInstance
    data, x_train, x_test, y_train, y_test, scalers = prepare_data()
    xgb_train = XGBoostInstance.train(x_train.reshape(x_train.shape[0], x_train.shape[1] * len(FEATURE_COLS)), y_train)
    xgb_predict = XGBoostInstance.predict(x_test.reshape(x_test.shape[0], x_test.shape[1] * len(FEATURE_COLS)))
    # Check NaN
    xgb_mse = mean_squared_error(y_test, xgb_predict)
    xgb_r2 = r2_score(y_test, xgb_predict)
    xgb_mae = mean_absolute_error(y_test, xgb_predict)
    # print("XGBoost Model Trained: ", xgb_predict)
    print("XGBoost Model MSE: ", xgb_mse)
    print("XGBoost Model R2: ", xgb_r2)
    print("XGBoost Model MAE: ", xgb_mae)

    rf_train = RandomForestInstance.train(x_train.reshape(x_train.shape[0], x_train.shape[1] * len(FEATURE_COLS)), y_train)
    rf_predict = RandomForestInstance.predict(x_test.reshape(x_test.shape[0], x_test.shape[1] * len(FEATURE_COLS)))
    rf_mse = mean_squared_error(y_test, rf_predict)
    rf_r2 = r2_score(y_test, rf_predict)
    rf_mae = mean_absolute_error(y_test, rf_predict)
    # print("Random Forest Model Trained: ", rf_predict)
    print("Random Forest Model MSE: ", rf_mse)
    print("Random Forest Model R2: ", rf_r2)
    print("Random Forest Model MAE: ", rf_mae)

    xgb_predict_inverse = scalers["TargetNextClose"].inverse_transform(xgb_predict.reshape(-1, 1))
    rf_predict_inverse = scalers["TargetNextClose"].inverse_transform(rf_predict.reshape(-1, 1))

    y_test_inverse = scalers["TargetNextClose"].inverse_transform(y_test.reshape(-1, 1))

    PredictionResultInstance.plot(y_test_inverse, xgb_predict_inverse, rf_predict_inverse)
    RSIandMACDInstance.plot(data)



