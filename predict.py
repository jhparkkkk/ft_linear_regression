import sys as sys
import pickle
from utils import standardize, reverse_standardize, load_parameters_from_file


parameters_list = ['theta0', 'theta1', 'x_mean', 'y_mean', 'x_std', 'y_std']


def set_parameters(parameters):
    """
    Set global variables based on parameters provided by the linear regression trained model

    Args:
        parameters (dict or None): parameters retrived in parameters pickle file saved by the trained model
    """
    if parameters is not None:
        for param_name in parameters_list:
            globals()[param_name] = parameters.get(param_name, 0)
    else:
        print('ici')
        for param_name in parameters_list:
            globals()[param_name] = 0

def estimate(x, theta0, theta1):
    """
    Calculate the predicted y value based on linear regression parameters

    Args:
        x (float): dependent variable
        theta0 (float): y coordinates when x=0
        theta1 (float): slope

    Returns:
        float: _description_
    """
    return theta0 + (theta1 * x)

def getUserInput() -> int | float:
    """
    Get user input for the x value

    Returns:
        userInput: indenpendent x variable from the linear regression
    """
    while True:
        try:
            userInput = input("What is the mileage?\n")
            if userInput.replace('.', '', 1).isdigit():
                return float(userInput)
            else:
                print("Invalid input")
        except KeyboardInterrupt:
                print('\nKeyboardInterrupt caught. Exiting...')
                exit()

if __name__ == "__main__":
    set_parameters(load_parameters_from_file('parameters'))
    x = getUserInput()
    if x and x_std:
        x_standardized = standardize(x, x_mean, x_std)
        y = estimate(x_standardized, theta0, theta1)
        y = reverse_standardize(y, y_mean, y_std)
        print(f"For a mileage of {x} km the predicted price of a car is {round(y)} euros")
    else:
        print(f"For a mileage of {x} km the predicted price of a car is {round(estimate(x, theta0, theta1))} euros")

