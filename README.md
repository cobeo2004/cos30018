# Stock Price Prediction System - COS30019 Option B

## Overview
### This project aims to develop a stock price prediction model using `Python` and `Tensorflow`. This system will handle **fetching data** from stocks, **clean** it and **create** a predictive model that could be used for **predicting** the stock price and **visualing** the graph.

## Author
### - Simon Nguyen.

## Technology
- [`Python V3.11+`](https://www.python.org/) as the main **programming langauge**.
- [`Anaconda`](https://www.anaconda.com/) as the main **virtual environment and distributor**.
- [`Tensorflow`](https://www.tensorflow.org/?hl=en) as the main framework for **creating neural network**.
- [`Scikit-learn`](https://scikit-learn.org/stable/) as them main library for **classifying, regressing and clustering**.
- [`Numpy`](https://numpy.org/) for handling **multiple-dimension arrays.**

## Initial setup
### Here is the initial setup procedure for creating an virtual environment using `Anaconda`.
- Refer to [`Anaconda`](https://www.anaconda.com/) for downloading procedure for the `CLI` and the `GUI` version.
- Once the `Anaconda` is installed, using the below command to create an `Python v3.11` with the name `test-env`:
    ```sh
        conda create -n test-env python=3.11
    ```
- Once the `conda` environment is created, you can check if your env is existed or not by typing the following command:
    ```sh
    conda env list
    # OUTPUT:

    # conda environments:
    #
    base                  *  /Users/<name>/miniconda3
    test-env              * /Users/<name>/miniconda3/test-env

    ```
- To **activate** the `conda` environment, type the command below to **activate** the environment:
    ```
    conda activate test-env
    ```
- To **check** if the environment is successfully created, simply check `Python` version by using:
    ```
    python --version # Or python -V
    ```
### With each sub-folder, there should be a `requirements.txt` file that indicates the required libraries. Make sure that you have created a virtual environment, then simply use this command to install all libraries in one:

    pip install -U -r requirements.txt

# Weekly Tasks
### Here are the weekly tasks that will get upgrade every single week
- [`Task B.1 - Set Up`](https://github.com/cobeo2004/cos30018/tree/main/Week1)
