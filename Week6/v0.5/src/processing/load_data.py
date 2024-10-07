"""
### Load data function. Check if exists data, if not then load and save from the `yfinance`
- **The major purpose of this function is to check if the unprocessed CSV file that contains the stock data from the start date and end date is existed in the file or not, if it is not existed then it will download the data from yfinance and save it to the directory that contains unprocessed CSV data. Otherwise, it will load the unprocessed CSV File. The following procedures are:**
    - The code will check if the directory that contains unprocessed CSV file exists, if not then it will start to create the directory and download the CSV from the yfinance. Once it’s downloaded then it will convert the data to CSV Format and save it to the created directory.
    - However, in the case where the directory exists then it will read the file in that directory.

"""

import os
from typing_extensions import Annotated, Doc
import yfinance as yf
import pandas as pd
from .utils import check_directory_exists, create_directory, check_file_exists
from .constants import RAW_DATA_DIRECTORY


def load_data(start: Annotated[str, Doc("The start date of the stock to be loaded")],
              end: Annotated[str, Doc("The end date of the stock to be loaded")],
              ticker: Annotated[str, Doc("The ticker symbol of the stock to be loaded")] = "CBA.AX"):
    """
    The major purpose of this function is to check if the unprocessed CSV file that contains the stock data from the start date and end date is existed in the file or not, if it is not existed then it will download the data from yfinance and save it to the directory that contains unprocessed CSV data. Otherwise, it will load the unprocessed CSV File

    Args:
        start (Annotated[str, Doc("The start date of the stock to be loaded")]): The start date of the stock to be loaded
        end (Annotated[str, Doc("The end date of the stock to be loaded")]): The end date of the stock to be loaded
        ticker (Annotated[str, Doc("The ticker symbol of the stock to be loaded")]): The ticker symbol of the stock to be loaded

    Raises:
        FileNotFoundError: The file is not found

    Returns:
        pd.DataFrame: The loaded data
    """
    try:
        RAW_CSV_FILE = os.path.join(
            RAW_DATA_DIRECTORY, f"raw_data_from_{start}_to_{end}_of_{ticker}_stock.csv")
        # Check if the raw data directory exists, if not then create it
        if not check_directory_exists(RAW_DATA_DIRECTORY):
            create_directory(RAW_DATA_DIRECTORY)
            data = yf.download(ticker, start=start, end=end)
            data.to_csv(RAW_CSV_FILE)
        else:
            # Load the data from the local machine
            if not check_file_exists(RAW_CSV_FILE):
                data = yf.download(ticker, start=start, end=end)
                data.to_csv(RAW_CSV_FILE)
            else:
                data = pd.read_csv(RAW_CSV_FILE)
        return data
    except FileNotFoundError as e:
        raise FileNotFoundError
