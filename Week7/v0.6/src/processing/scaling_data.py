"""
### Scaling data using `MinMaxScaler()`
- **This function will check if the is_scale equals to True or not, if it is not then the function will simply return back the data without any scaling data. Otherwise, it will do the following procedures:**
    - Scaling data: This function will use the MinMaxScaler class provided by scikit-learn to scale the data to the range of (0, 1) by setting the feature_range to a tuple of (0, 1).
    - Reshaping data: After the data is scaled, the data will be converted into a two-dimensional array using reshape(-1, 1) function. This procedure is essential as the fit_transform() function only accepts a 2D array.
    - Fitting and returning the scaled array: Once the array is transformed to 2D array, it will be fitted using the fit_transform() function provided by class MinMaxScaler. Once the transform is finished we will return the scaled data with the instance of MinMaxScaler for further usage.
"""

from typing_extensions import Annotated, Doc, Tuple
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scaling_data(data: Annotated[pd.DataFrame, Doc("The data to be scaled")], is_scale: Annotated[bool, Doc("Choose to scale or not, default set to True")] = True) -> Tuple[pd.DataFrame, MinMaxScaler | None]:
    """
    Scaling the data to the range of (0, 1)

    Args:
        data (Annotated[pd.DataFrame, Doc]): The data to be scaled
        is_scale (Annotated[bool, Doc, optional]): Choose to scale or not, default set to True

    Returns:
        Tuple[pd.DataFrame, MinMaxScaler | None]: The scaled data and the scaler
    """
    if is_scale:
        # Using MinMaxScaler to scale the data to the range of (0, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        if len(data.shape) == 1:
            # If the data array is 1D, convert to 2D array
            data = data.values.reshape(-1, 1)
        # Fit the 2D-transformed data into the scaler
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler
    else:
        return data, None
