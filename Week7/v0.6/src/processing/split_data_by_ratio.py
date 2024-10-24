"""
### Split data by ratio
**This function will do the following procedures:**
- If is_split_by_date is set to True: This function will perform calculation based on the split ratio (0.8 will equal 80% of train data and 20% of test data). Once the calculation is finished, the function will allocate the train and test data to the train_data and test_data variables.
- If is_split_by_date is set to False: This function will use the built-in function provided by scikit-learn, which is train_test_split() to split the data randomly based on the given ratio value. The random_state is fixed at 42 to make sure that the splitted data is reproducible.
- Once the data has been successfully splitted, it will then print out the shape (x,y) of both train and test data and return it to the user for further usages.
"""

from typing_extensions import Annotated, Doc, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data_by_ratio(data: Annotated[pd.DataFrame, Doc("The data to be split")],
                        ratio: Annotated[float, Doc("The ratio to be used in percentage: 0.8 for 80% train and 20% test")],
                        is_split_by_date: Annotated[bool, Doc("Choose to split by date or random")] = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test data based on the ratio

    Args:
        data (Annotated[pd.DataFrame, Doc("The data to be split")]): The data to be split
        ratio (Annotated[float, Doc("The ratio to be used in percentage: 0.8 for 80% train and 20% test")]): The ratio to be used in percentage: 0.8 for 80% train and 20% test
        is_split_by_date (Annotated[bool, Doc("Choose to split by date or random")]): Choose to split by date or random

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The train and test data
    """
    # Check if the data is split by date
    if is_split_by_date:
        # Calculate the train data size of the train data
        train_size = int(len(data) * ratio)
        # Split the data into train and test data
        train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]
    else:
        # Split the data into train and test data randomly
        train_data, test_data = train_test_split(
            data, train_size=1 - (1 - ratio), test_size=1 - ratio, random_state=42)

    # Print the shape of the train and test data
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    return train_data, test_data
