
"""
### Declare essential variables
- Define the start date, the end date and the ticker (stock) that we wanted to download from `yfinance`.
- Define the split ratio for splitting the train and test data, and the number of days for looking back.
- Define the directory for storing the raw and prepared datasets.
- Define the files that are used for storing essential information, such as raw data imported from yfinance and the processed data.
"""

import os
# import sys
# from google.colab import drive
# DRIVE_DISK = '/content/drive'
# drive.flush_and_unmount();
# drive.mount(DRIVE_DISK)

# # Start, end and ticker for the datasets
# start="2015-01-01"
# end="2023-08-25"
# ticker="TSLA"

# ENTRY_POINT = f"{DRIVE_DISK}/MyDrive/COS30018/Week7/datasets/"
# os.chdir(ENTRY_POINT)


# # Split ratio, 0.8 equals to 80% data for training and 20% for testing
# split_ratio = 0.8

# # Number of days to look back for the prediction, could be changed to any value
# num_look_back_days = 30

# Start, end and ticker for the datasets
start = "2015-01-01"
end = "2023-08-25"
ticker = "TSLA"

# Split ratio, 0.8 equals to 80% data for training and 20% for testing
split_ratio = 0.8

# Number of days to look back for the prediction, could be changed to any value
num_look_back_days = 30

# Define entry directory for the raw and prepared datasets
# ENTRY_POINT = f"datasets/{ticker}/"
ENTRY_POINT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'datasets'))


# Define entry directory for the raw and prepared datasets
# ENTRY_POINT = f"datasets/{ticker}/from_{start}_to_{end}"
RAW_DATA_DIRECTORY = os.path.join(ENTRY_POINT, "data")
PREPARED_DATA_DIRECTORY = os.path.join(ENTRY_POINT, "prepared-data")

# # Raw data to be saved as the same for
# RAW_CSV_FILE = os.path.join(RAW_DATA_DIRECTORY, f"raw_data_from_{start}_to_{end}_of_{ticker}_stock.csv")

# # Prepared data to be saved as the same for
# PREPARED_CSV = os.path.join(PREPARED_DATA_DIRECTORY, f"prepared_data_from_{start}_to_{end}_of_{ticker}_stock.csv")
# PREPARED_TRAIN_ALL = os.path.join(PREPARED_DATA_DIRECTORY, f"xytrain_data_{start}-{end}-{ticker}_stock.npz")
# PREPARED_TRAIN_DATASET = os.path.join(PREPARED_DATA_DIRECTORY, f"train_dataset_of_{ticker}_from_{start}_to_{end}.csv")
# PREPARED_TEST_DATASET = os.path.join(PREPARED_DATA_DIRECTORY, f"test_dataset_of_{ticker}_from_{start}_to_{end}.csv")
# PREPARED_SCALER_FEATURE = os.path.join(PREPARED_DATA_DIRECTORY, f"feature_scaler_of_{ticker}_from_{start}_to_{end}.pkl")
# PREPARED_SCALER_TARGET = os.path.join(PREPARED_DATA_DIRECTORY, f"targe_scaler_of_{ticker}_from_{start}_to_{end}.pkl")
# PREPARED_TRAIN_ARRAY = os.path.join(PREPARED_DATA_DIRECTORY, f"xytrain_train_array_of_{ticker}_from_{start}_to_{end}.npz")
# PREPARED_TEST_ARRAY = os.path.join(PREPARED_DATA_DIRECTORY, f"xytrain_test_array_of_{ticker}_from_{start}_to_{end}.npz")
