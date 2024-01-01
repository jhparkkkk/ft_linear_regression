import matplotlib.pyplot as plt
import numpy as np

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

def standardize(df, column_names):
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

    print('----------STANDARDIZE--------------')
    print(df[y])
    print(means[1])
    print(standard_deviation[1])


    standardized_data = (df[column_names] - means) / standard_deviation
    
    return standardized_data, means, standard_deviation

def reverse_standardize(standardized, mean, std):
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
#                           graphs utils                           #
# -----------------------------------------------------------------#

def plot_linear_regression(axs, df, x, y,  my_predictions, test_predictions):
    # reshape
    np.array(df[x]).reshape(-1, 1)
    np.array(df[y]).reshape(-1, 1)

    # Scikit-learn linear regression
    axs[0].scatter(df[x], df[y], color='b')
    axs[0].plot(df[x], test_predictions, color='k')
    axs[0].set_title('Scikit-Learn Linear Regression')
    axs[0].set_xlabel('Mileage (km)')
    axs[0].set_ylabel('Price (euros)')

    # My linear regression
    scatter = axs[1].scatter(df[x], df[y], c=df['price'], cmap='cool')
    axs[1].plot(df[x], my_predictions, color='r')
    axs[1].set_title('My Linear Regression')
    axs[1].set_xlabel('Mileage (km)')
    axs[1].set_ylabel('Price (euros)')

def plot_loss_function(axs, costs):
    axs[2].plot(range(len(costs)), costs, 'g', label='Training loss')
    axs[2].set_title('Training costs')
    axs[2].set_xlabel('Iterations')
    axs[2].set_ylabel('Costs')
    axs[2].legend()

def plot_all(df, x, y, my_predictions, test_predictions, costs):
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    plot_linear_regression(axs, df, x, y, my_predictions, test_predictions)
    plot_loss_function(axs, costs)
    plt.tight_layout()
    plt.show()
