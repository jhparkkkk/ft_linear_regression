import matplotlib.pyplot as plt
from colorama import Fore
import numpy as np
from utils import feature_scaling, reverse_feature_scaling, plot_linear_regression, plot_loss_function, plot_all

class linearRegression:
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        self.learning_rate = 0.1
        self.theta = [0, 0]
        self.m = len(self.Y)
        
    def __getitem__(self, index: int) -> float:
        """
        Get the parameter at the index of the theta list

        Args:
            index (int): index to access in the theta list

        Returns:
            float: value of theta0 or theta1
        """
        return self.theta[index]

    def predict(self) -> np.array:
        """
        Predict using linear regression model
        y = theta0 + theta1 * x

        Args:
            X (list, optional): _description_. Defaults to [].

        Returns:
            _type_: _description_
        """
        Y_predictions = np.array([])
        for x in self.X:
            Y_predictions = np.append(Y_predictions, self.theta[0] + (self.theta[1] * x))
        return Y_predictions
    
    def loss_function(self, Y_pred: np.array) -> float:
        """
        Mean Square Error
        Tells how much predictions are off from the actual result

        Args:
            Y_pred (_type_): Predicted values

        Returns:
            float: error 
        """
        mse = np.sum((self.Y - Y_pred) ** 2) / self.m
        return mse

    def update_theta(self, Y_predictions: list):
        """
        Update the parameters theta0 and theta1 using gradient descent
        errors = sum(Y_predictions - Y)
        theta0 = theta0 - learning_rate * 1/m * errors
        theta1 = theta1 - learning_rate * 1/m * errors * X

        Args:
            Y_predictions (list): predicted values
        """
        errors = [a - b for a, b in zip(Y_predictions, self.Y)]
        self.theta[0] -= self.learning_rate * (1/self.m) * sum(errors)

        x_scaled_errors = [a * b for a, b in zip(errors, self.X)]
        self.theta[1] -= self.learning_rate * (1/self.m) * sum(x_scaled_errors)

