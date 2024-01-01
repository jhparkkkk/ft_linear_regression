import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

#------------------------------------------------------------------#
#                           maths utils                            #
# -----------------------------------------------------------------#
def mean(df_column) -> int | float:
    """

    Calculate average of a given list of numbers
    mean = (sum of observations) ÷ (total number of observations)
    Args:
        data_list (list): data set to operate on

    Returns:
        int | float: means of the data set passed as parameters
    
    """
    data = [float(x) for x in df_column]
    return (sum(data) / len(data))


def variance(data: list) -> float:
    """

    Calculate the variance ofthe given data. 
    A large variance indicates that the data is spread out

    Args:
        data (list): data set to operate on

    Returns:
        float: the variance value
    
    """
    mean_value = mean(data)
    variance = 0
    for value in data:
        variance += (value - mean_value) ** 2
    return(variance / len(data))


def std(df_column) -> float:
    """

    Calculate the standard deviation : describe the variation
    of data points meaning how close they are to the mean.

    Args:
        data_list (list): data set to operate on

    Returns:
        float: the standard deviation value
    
    """
    data = [float(x) for x in df_column]
    return variance(data) ** (1/2)

def standardize(x: float, mean: float, std: float) -> float:
    """
    
    Standardize a value using the mean and standard deviation

    Args:
        x (float): value to be  standardized
        mean (float): mean of the original data
        std (float): standard deviation of the original data

    Returns:
        float: standardized value
    
    """
    return ((x - mean) / std)

def reverse_standardize(x, mean, std):
    """

    Reverse the standardization of a value using the mean and standard deviation

    Args:
        x (float): standardized value
        mean (float): mean of the original data
        std (float): standard deviation of the original data

    Returns:
        float: reversed value
    
    """
    return (x * std + mean)

def feature_scaling(df, column_names):
    """

    Standardize the specified columns in a DataFrame

    Args:
        df (pd.DataFrame): dataframe containing the data
        column_names (list[str]): lsit of column names to be standardized

    Returns:
        tuple:
        1. pd.DataFrame: standardized data for specified columns 
        2. list: means of the specified columns
        2. list standard deviations of the specified columns

    """
    x = column_names[0]
    y = column_names[1]

    means = []
    means.append(mean(df[x]))
    means.append(mean(df[y]))

    standard_deviation = []
    standard_deviation.append(std(df[x]))
    standard_deviation.append(std(df[y]))

    standardized_data = (df[column_names] - means) / standard_deviation
    return standardized_data, means, standard_deviation

def reverse_feature_scaling(standardized, mean, std):
    """

    Reverse the standardization transformation

    Args:
        standardized (_type_): value in standardized scale
        mean (_type_): mean value used during standardization
        std (_type_): standard deviation value used during standardization

    Returns:
        reverse_standardized: values in original scale

    """
    scaled_data = [x * std for x in standardized]
    reverse_standardized = [x + mean for x in scaled_data]
    return reverse_standardized

#------------------------------------------------------------------#
#                       file management utils                      #
# -----------------------------------------------------------------#
def load_csv(csv_filename) -> 'pandas.core.frame.DataFrame':
    if not csv_filename.endswith('.csv'):
        raise ValueError("The provided file is not a CSV file.")
    df = pd.read_csv(csv_filename)
    # print(df.shape())
    if df.shape[1] != 2:
        raise TypeError("Invalid dataset to perform linear regression.")
    column_names = df.columns
    return df, column_names[0], column_names[1]

def save_parameters_to_file(parameters, filename):
    with open(filename, 'wb') as file:
        pickle.dump(parameters, file)
    print(f"Parameters saved to {filename}")

def load_parameters_from_file(filename):
    """
    Load parameters from a file using pickle.

    Parameters:
    - filename: The name of the file containing the parameters.

    Returns:
    - parameters: The loaded parameters.
    """
    try:
        with open(filename, 'rb') as file:
            parameters = pickle.load(file)
            print(f"Parameters loaded from the trained model: I am able to predict price of a car for a given mileage.")
            return parameters
    except FileNotFoundError:
        print(f"Parameters '{filename}' not loaded. Model hasn't been trained yet")
        return None 
# 

#------------------------------------------------------------------#
#                           graphs utils                           #
# -----------------------------------------------------------------#

def plot_linear_regression(ax, df, x, y, y_predictions, title, color1, color2):
    """
    Plot linear regression results on a single subplot

    Args:
        ax (matplotlib.axes._subplots.AxesSubplot): subplot axis
        df (pd.dataframe): dataframe containing the data
        x (str): name of the x column
        y (str): name of the y column 
        y_predictions (list): predicted y values from linear regression
        title (str): title of the plot
        color1 (str): color for the scatter points
        color2 (str): color for the regression line 
    """
    ax.scatter(df[x], df[y], color=color1)
    ax.plot(df[x], y_predictions, color=color2)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

def plot_loss_function(ax, costs):
    """
    Plot the training loss function over iterations

    Args:
        axs (matplotlib.axes._axes.Axes): subplot axe
        costs (list): list of training cost over iterations
    """
    ax.plot(range(len(costs)), costs, 'g', label='Training loss')
    ax.set_title('Training costs')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Costs')
    ax.legend()

def plot_all(df, x_name, y_name, my_predictions, test_predictions, costs):
    """
    Plot all visualizations for linear regression analysis

    Args:
        df (pd.DataFrame): dataframe containing the data
        x_name (str): name of the column x 
        y_name (str): name of the column y
        my_predictions (list): y predictions from custom linear regression
        test_predictions (numpy.ndarray): y predictions from scikit-learn linear regression
        costs (list): list of training costs over iterations
    """
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    plot_linear_regression(axs[1], df, x_name, y_name, my_predictions, 'My Linear Regression', 'b', 'm')
    plot_linear_regression(axs[0], df, x_name, y_name, test_predictions, 'SciKit-Learn Linear Regression', 'g', 'r')
    plot_loss_function(axs[2], costs)
    plt.tight_layout()
    plt.show()
