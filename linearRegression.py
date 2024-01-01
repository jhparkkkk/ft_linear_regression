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
        
    def __getitem__(self, index):
        return self.theta[index]

    def predict(self, X=[]):
        Y_predictions = np.array([])
        if not X:
            X = self.X
        for x in X:
            Y_predictions = np.append(Y_predictions, self.theta[0] + (self.theta[1] * x))
        return Y_predictions
    
    def loss_function(self, Y_pred) -> float:
        """
        Mean Square Error
        Tells me how much I am off from the actual result

        Args:
            Y_pred (_type_): Predicted values

        Returns:
            float: error 
        """
        mse = (1 / (2 * self.m)) * (np.sum(Y_pred - self.Y)**2)
        return mse

    def update_theta(self, Y_predictions):
        errors = [a - b for a, b in zip(Y_predictions, self.Y)]
        self.theta[0] -= self.learning_rate * (1/self.m) * sum(errors)
        
        x_scaled_errors = [a * b for a, b in zip(errors, self.X)]
        self.theta[1] -= self.learning_rate * (1/self.m) * sum(x_scaled_errors)

