"""
### Create datasets
**The given procedure, which is the heart to execute successfully the program, does the following steps:**
- Load and validate data: The procedure initially calls to the defined load_data() function, which will either load the existing raw data or download the data if it does not exist. Once the data is loaded successfully, it will call the validate_data() function to process and validate the data to be in the right format, which adds in extra technical analysis indicators and target values.
- Check the existence of the prepared datasets: After the data is validated, it will check if the train and test datasets exist. If the datasets are existed, it will do the following steps:
    - Load from the processed CSV file the existing datasets, including train and test datasets.
    - Print out the shapes of both train and test datasets for validation.
    - Load the saved scaler values of both feature and target.
    - Retrieve the value from the x and y array of both train and test arrays.
- In case where the datasets are not existed, the following logic will be executed:
    - The validated data will be split into train and test data using the defined split_data_by_ratio() function. With the split rate defined at 0.8 (80% of train data, 20% of test data).
    - Scale the training data for the features using the defined scaling_data() function and based on the defined feature_cols columns.
    - Make training arrays by creating a sequence for each sample based on the defined variable num_look_back_days.
    - Scale the testing data for the features and target using the defined scaling_data() function and based on the defined feature_cols and target_cols columns.
    - Make test arrays by creating a sequence of each sample based on the defined variable num_look_back_days.
    - Save the train and test data to a CSV file by using the to_csv()function from pandas and save the feature and target scalers using the defined save_or_load_object() function.
    - Save the created train and test arrays to a .npz file using savez()function from numpy.

"""

from .constants import PREPARED_DATA_DIRECTORY, ticker, start, end, split_ratio, num_look_back_days
import os
import pandas as pd
import numpy as np
from .utils import check_file_exists, save_or_load_object
from .validate_data import validate_data
from .load_data import load_data
from .split_data_by_ratio import split_data_by_ratio
from .scaling_data import scaling_data


def make_datasets():
    PREPARED_TRAIN_ALL = os.path.join(
        PREPARED_DATA_DIRECTORY, f"xytrain_data_{start}-{end}-{ticker}_stock.npz")
    PREPARED_TRAIN_DATASET = os.path.join(
        PREPARED_DATA_DIRECTORY, f"train_dataset_of_{ticker}_from_{start}_to_{end}.csv")
    PREPARED_TEST_DATASET = os.path.join(
        PREPARED_DATA_DIRECTORY, f"test_dataset_of_{ticker}_from_{start}_to_{end}.csv")
    PREPARED_SCALER_FEATURE = os.path.join(
        PREPARED_DATA_DIRECTORY, f"feature_scaler_of_{ticker}_from_{start}_to_{end}.pkl")
    PREPARED_SCALER_TARGET = os.path.join(
        PREPARED_DATA_DIRECTORY, f"targe_scaler_of_{ticker}_from_{start}_to_{end}.pkl")
    PREPARED_TRAIN_ARRAY = os.path.join(
        PREPARED_DATA_DIRECTORY, f"xytrain_train_array_of_{ticker}_from_{start}_to_{end}.npz")
    PREPARED_TEST_ARRAY = os.path.join(
        PREPARED_DATA_DIRECTORY, f"xytrain_test_array_of_{ticker}_from_{start}_to_{end}.npz")

    data = load_data(start, end, ticker)
    df = validate_data(start, end, ticker)

    if check_file_exists(PREPARED_TRAIN_DATASET) and check_file_exists(PREPARED_TEST_DATASET):
        print('Loading Existed Train and Test Data...')
        train_data = pd.read_csv(PREPARED_TRAIN_DATASET)
        test_data = pd.read_csv(PREPARED_TEST_DATASET)

        print(f"Train data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")

        train_feature_scale = save_or_load_object(
            None, PREPARED_SCALER_FEATURE, "load")
        train_target_scale = save_or_load_object(
            None, PREPARED_SCALER_TARGET, "load")
        train_arrays = np.load(PREPARED_TRAIN_ARRAY)
        x_train = train_arrays['x_train']
        y_train = train_arrays['y_train']

        test_arrays = np.load(PREPARED_TEST_ARRAY)
        x_test = test_arrays['x_test']
        y_test = test_arrays['y_test']
    else:
        print('Processing Train and Test Data...')
        train_data, test_data = split_data_by_ratio(df, split_ratio)

        feature_cols = ['Open', 'High', 'Low', 'Close_RSI',
                        'Close_EMA20', 'Close_EMA100', 'Close_EMA200']
        target_cols = 'TargetNextClose'

        scaled_train_data, train_feature_scale = scaling_data(
            train_data[feature_cols])
        converted_2d_train_data = train_data[target_cols].values.reshape(-1, 1)
        scaled_train_target, train_target_scale = scaling_data(
            converted_2d_train_data)

        x_train, y_train = [], []
        for i in range(num_look_back_days, len(scaled_train_data)):
            x_train.append(scaled_train_data[i-num_look_back_days:i])
            y_train.append(scaled_train_target[i])

        x_train, y_train = np.array(x_train), np.array(y_train)

        scaled_test_data = train_feature_scale.transform(
            test_data[feature_cols])
        converted_2d_test_data = test_data[target_cols].values.reshape(-1, 1)
        scaled_test_target = train_target_scale.transform(
            converted_2d_test_data)

        x_test, y_test = [], []
        for i in range(num_look_back_days, len(scaled_test_data)):
            x_test.append(scaled_test_data[i-num_look_back_days:i])
            y_test.append(scaled_test_target[i])

        x_test, y_test = np.array(x_test), np.array(y_test)

        train_data.to_csv(PREPARED_TRAIN_DATASET, index=False)
        test_data.to_csv(PREPARED_TEST_DATASET, index=False)

        save_or_load_object(train_feature_scale,
                            PREPARED_SCALER_FEATURE, "save")
        save_or_load_object(train_target_scale, PREPARED_SCALER_TARGET, "save")

        np.savez(PREPARED_TRAIN_ARRAY, x_train=x_train, y_train=y_train)
        np.savez(PREPARED_TEST_ARRAY, x_test=x_test, y_test=y_test)

    return data, df, train_data, test_data, train_feature_scale, train_target_scale, x_train, x_test, y_train, y_test
