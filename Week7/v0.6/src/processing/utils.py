"""
### Ensure that directory must exists
- check_directory_exists() & create_directory(): The purpose of these two functions are two check if the given dir_path is exists or not using os.path.isdir(), if not then the check_directory_exists() will return a False and use create_directory() to create a folder based on the given dir_path, otherwise return True.
- check_file_exists(): This function will check if the file exists using os.path.exists() function, it will return True if exists otherwise return False.
- save_or_load_object(): This function will save the object to the file if the mode is save, otherwise load the object from the file if the mode is load. If the object is None, then load the object from the file. If the mode is invalid, it will raise an error.
- ensemble_model_predict(): This function will ensemble the prediction results from ARIMA and DL models
"""

import os
from typing_extensions import Annotated, Doc, Literal, TypeVar
import pickle
import numpy as np


def check_directory_exists(dir_path: Annotated[str, Doc("The path to the directory to be checked")]) -> bool:
    """
    check_directory_exists() & create_directory(): The purpose of these two functions are two check if the given dir_path is exists or not using os.path.isdir(), if not then the check_directory_exists() will return a False and use create_directory() to create a folder based on the given dir_path, otherwise return True.

    Args:
        dir_path (Annotated[str, Doc]): The path to the directory to be checked

    Returns:
        bool: True if the directory exists, otherwise False
    """
    return True if os.path.isdir(dir_path) else False


def create_directory(dir_path: Annotated[str, Doc("The path to the directory to be created")]) -> None:
    """
    Create a directory based on the given dir_path

    Args:
        dir_path (Annotated[str, Doc]): The path to the directory to be created
    """
    os.makedirs(dir_path)


def check_file_exists(file_path: Annotated[str, Doc("The path to the file to be checked")]) -> bool:
    """
    check_file_exists(): This function will check if the file exists using os.path.exists() function, it will return True if exists otherwise return False.

    Args:
        file_path (Annotated[str, Doc]): The path to the file to be checked

    Returns:
        bool: True if the file exists, otherwise False
    """
    return True if os.path.exists(file_path) else False


T = TypeVar("T", bound=Literal["load", "save"])
U = TypeVar("U", any, object)


def save_or_load_object(obj: Annotated[U | None, Doc("The object to be saved or loaded")],
                        fn: Annotated[str, Doc("The filename to the directory to be checked")],
                        mode: Annotated[T, Doc("The mode to be used")]) -> U | None:
    """
    Save or load the object to the file based on the mode
    - If the mode is set to save, the function will start opening up the file defined in fn parameter. Once the file is successfully opened, it will write all of the given objects defined in obj by using pickle.dump() to ensure that the object will be type-safe when loading the object from the file.
- If the mode is set to load, the function will start reading from the file defined in fn and load the object using pickle.load() function

    Args:
        obj (Annotated[U | None, Doc("The object to be saved or loaded")]): The object to be saved or loaded
        fn (Annotated[str, Doc("The filename to the directory to be checked")]): The filename to the directory to be checked
        mode (Annotated[T, Doc("The mode to be used")]): The mode to be used

    Returns:
        U | None: The object to be saved or loaded
    """
    # Save the object to the file if the mode is save
    if mode == "save":
        with open(fn, "wb") as i:
            pickle.dump(obj, i)
    # Load the object from the file if the mode is load
    elif mode == "load":
        # Check if the object is None, if so then load the object from the file
        if obj is None:
            with open(fn, "rb") as i:
                return pickle.load(i)
        else:
            raise ValueError("obj must be None when mode is load")
    # Raise an error if the mode is invalid
    else:
        raise ValueError("Invalid mode")


def ensemble_model_predict(arima_pred_res: np.ndarray, dl_pred_res: np.ndarray) -> np.ndarray:
    """
    Ensemble the prediction results from ARIMA and DL models
    """
    return (arima_pred_res + dl_pred_res) / 2
