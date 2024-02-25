import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from colorama import Fore
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from linearRegression import linearRegression
from utils import feature_scaling, reverse_feature_scaling, plot_all, load_csv, save_parameters_to_file
from test import test

EPOCHS = 300

def gradient_descent(x, y):
    """
    Train a linear regression model using gradient descent

    Args:
        x (np.array): input x data (independent variable)
        y (np.array): input y data (dependent variable)

    Returns:
        tuple: the trained model, predictions resulting from the training and training costs
    """
    model = linearRegression(x, y)
    nb_iterations = 0   
    costs = []
    while nb_iterations < EPOCHS:
        Y_predictions = model.predict()
        cost = model.loss_function(Y_predictions)
        costs.append(cost)
        model.update_theta(Y_predictions)
        nb_iterations += 1

    return model, Y_predictions, costs

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            print("Usage: python train.py <csv_file_path>")
            sys.exit(1)
        # load data
        df, x, y = load_csv(sys.argv[1])
        # feature scaling 
        df_scaled, means, stds = feature_scaling(df, [x, y])
        # train model
        model, my_predictions, costs = gradient_descent(np.array(df_scaled[x]), np.array(df_scaled[y]))        
        # reverse feature scaling to plot it with original scale
        my_predictions = reverse_feature_scaling(my_predictions, means[1], stds[1])
        # train data with scikit-learn library
        scikit_learn_predictions = test(df[x], df[y])
        # save parameters to file
        parameters_to_save = {'theta0': model.theta[0], 'theta1': model.theta[1], 'x_mean': means[0], 'y_mean': means[1], 'x_std': stds[0], 'y_std': stds[1]}
        save_parameters_to_file(parameters_to_save, 'parameters')
        
        # plot result
        plot_all(df, x, y, my_predictions, scikit_learn_predictions, costs)
    except Exception as error:
        print(error)
