import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from colorama import Fore

# TODO: to test with my own LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from linearRegression import linearRegression
from utils import standardize, reverse_standardize, plot_linear_regression, plot_loss_function, plot_all
from test import test

BASE_DIR = "./data/"
EPOCHS = 200



def load_csv(csv_filename) -> 'pandas.core.frame.DataFrame':
    if not csv_filename.endswith('.csv'):
        raise ValueError("The provided file is not a CSV file.")
    df = pd.read_csv(f'./data/{csv_filename}')
    column_names = df.columns
    x = column_names[0]
    y = column_names[1]
    return df, x, y

def gradient_descent(x, y):
    model = linearRegression(x, y)
    nb_iterations = 0   
    costs = []
    while nb_iterations < EPOCHS:
        Y_predictions = model.predict()
        cost = model.loss_function(Y_predictions)
        print('cost: ', cost)
        costs.append(cost)
        model.update_theta(Y_predictions)
        nb_iterations += 1

    theta0 = model.theta[0]
    theta1 = model.theta[1]
    print('THETA0', theta0, 'THETA1', theta1)
    return model, Y_predictions, costs

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            print("Usage: python train.py <csv_file_path>")
            sys.exit(1)
        
        df, x, y = load_csv(sys.argv[1])
        df_std, means, stds = standardize(df, [x, y])
        # print(stds, means)
        model, my_predictions, costs = gradient_descent(np.array(df_std[x]), np.array(df_std[y]))        
        # my_predictions = reverse_standardize(my_predictions, means[1], stds[1])
        test_predictions = test(df[x], df[y])
        print(stds, means)
        
        # intercept, slope = model.reverse_standardized_theta(stds, means)
        # print('SLOPE', slope, 'INTERCEPT', intercept)
        print('MEANS', means, 'std', stds)
        plot_all(df, x, y, my_predictions, test_predictions, costs)
    except Exception as error:
        print(error)
