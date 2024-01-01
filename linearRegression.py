import matplotlib.pyplot as plt
from colorama import Fore
import numpy as np

class linearRegression:
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        self.learning_rate = 0.1
        # theta[0] = intercept, theta[1] = slope
        self.theta = [0, 0]
        # number of data points
        self.m = len(self.Y)
        print(f"price: {self.Y}\nmileage: {self.X}\ntheta[0] = {self.theta[0]}\ntheta[1] = {self.theta[1]}\nsize = {self.m}\nlearning rate = {self.learning_rate}")

    def __getitem__(self, index):
        return self.theta

    def predict(self, X=[]):
        Y_predictions = np.array([])
        if not X:
            X = self.X
        for x in X:
            Y_predictions = np.append(Y_predictions, self.theta[0] + (self.theta[1] * x))
        print(Fore.YELLOW, Y_predictions)
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
        print('theta[0]: ', self.theta[0], 'theta[1]: ', self.theta[1], '\n')

    def reverse_standardized_theta(self, stds, means):
        self.theta[1] = means[1] - self.theta[0] * (means[0] / stds[0])
        self.theta[0] = self.theta[0] * (stds[1] / stds[0])
        # print('self.theta[0]', self.theta[0], 'y_intersect', y_intersect)
        return self.theta[0], self.theta[1]


    def plot_best_fit(self, Y_pred, fig=''):
        f = plt.figure(fig)
        plt.scatter(self.X, self.Y, color='b')
        plt.plot(self.X, Y_pred, color='g')
        plt.show()
        # print('ici')
    
