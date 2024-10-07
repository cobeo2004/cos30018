
"""
### Validate data
**This function will check if the directory that contains processed data exists, if not then it will create a directory used for storing processed data.
Once the directory is created, it will check the existence of the processed CSV file. If the CSV file exists, it will read the CSV file using pd.read_csv() and return it. Otherwise, it will start the following procedure**:
- Read the unprocessed CSV file: It will read the unprocessed CSV file stored in the unprocessed folder.
- Convert and set index: After the unprocessed CSV file is successfully read, it will convert the value of Date column into the DateTime type and set the value inside the Date column as index. This will make it easier to operate time-based searching, simplify the plotting procedure and enhance the data organization.
- Adding essential indicators to the data: This function will also add several required technical analysis indicators to the data, such as Relative Strength Index (RSI) and several different date ranges (20 days, 100 days and 150 days) of Exponential Moving Averages (EMA) using the built-in functions provided by TA-lib.
- Calculate target price: After adding indicators to the data, it will start calculating the target price by subtracting the value of Adjustment Close Price and the Open Price. Once subtracted, it will shift the value back by one to assume that is the targeted price.
- Indicate if the value is increased or not: It will calculate the TargetClass based on the target price value and indicate if the price is increased or not, if the price is increased then the TargetClass value will be 1, otherwise it will be 0.
- Drop all undefined values: By using dropna() function, it will look up all of the undefined (NaN) values and drop it.
- Save processed data: Once the datasets are processed, it will save those datasets into a CSV File by using to_csv() function and return the processed datasets to the user.
"""

from typing_extensions import Annotated, Doc
import pandas as pd
from .utils import check_directory_exists, create_directory, check_file_exists
from .constants import RAW_DATA_DIRECTORY, PREPARED_DATA_DIRECTORY, ticker
import os
import numpy as np
import talib as ta


def validate_data(start: Annotated[str, Doc("Start date")],
                  end: Annotated[str, Doc("End date")],
                  ticker: Annotated[int, Doc("Ticker")] = ticker) -> pd.DataFrame:
    """
    Validate the data

    Args:
        start (Annotated[str, Doc("Start date")]): The start date
        end (Annotated[str, Doc("End date")]): The end date
        ticker (Annotated[int, Doc("Ticker")]): The ticker

    Returns:
        pd.DataFrame: The processed data
    """
    RAW_CSV_FILE = os.path.join(
        RAW_DATA_DIRECTORY, f"raw_data_from_{start}_to_{end}_of_{ticker}_stock.csv")
    PREPARED_CSV = os.path.join(
        PREPARED_DATA_DIRECTORY, f"prepared_data_from_{start}_to_{end}_of_{ticker}_stock.csv")

    # Check if the prepared data directory exists, if not then create it
    if not check_directory_exists(PREPARED_DATA_DIRECTORY):
        create_directory(PREPARED_DATA_DIRECTORY)

    if check_file_exists(PREPARED_CSV):
        print('Loading Prepared Data...')
        df = pd.read_csv(PREPARED_CSV)
    else:
        print('Processing Raw Data...')
        # Read the raw data
        df = pd.read_csv(RAW_CSV_FILE)
        # Convert the date column to datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df['Date'] = pd.to_datetime(df['Date'])
            # Set the date column as the index
            df.set_index('Date', inplace=True)
        print("Type of index after converted: ", type(df.index))
        # Adding RSI, EMA20(EMA for 20 days), EMA100(EMA for 100 days), EMA200(EMA for 200 days)
        df['Close_RSI'] = ta.RSI(df['Close'], timeperiod=14)
        df['Close_EMA20'] = ta.EMA(df['Close'], timeperiod=20)
        df['Close_EMA100'] = ta.EMA(df['Close'], timeperiod=100)
        df['Close_EMA200'] = ta.EMA(df['Close'], timeperiod=200)

        # Calculate the target value by subtracting the open price
        # and the adjusted close price and shifting it by 1 day
        df['Target'] = df['Adj Close'] - df['Open']
        df['Target'] = df['Target'].shift(-1)

        # Convert the target class to binary class if the target is greater than 0
        df['TargetClass'] = np.where(df['Target'] > 0, 1, 0)
        # Shift the adjusted close price by 1 day
        df['TargetNextClose'] = df['Adj Close'].shift(-1)

        # Drop the NaN values
        df.dropna(inplace=True)

        # Convert to csv and save to the folder
        df.to_csv(PREPARED_CSV, index=False)
    return df
