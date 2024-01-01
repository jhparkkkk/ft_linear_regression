from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from linearRegression import linearRegression
import numpy as np 
import matplotlib.pyplot as plt

def test(x, y):
    test = LinearRegression()
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    test.fit(x, y)
    y_predictions = test.predict(x)
    return y_predictions
    # # mse = mean_squared_error(y_true=price,test_y_pred=test_y_pred)
    # plt.scatter(mileage, price, color ='b')
    # plt.plot(mileage, test_y_pred, color ='k')
    
    #  # Create subplots
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))


    # # Plot the first figure
    # axs[0].scatter(mileage, price, color='b')
    # axs[0].plot(mileage, test_y_pred, color='k')
    # axs[0].set_title('Scikit-Learn Linear Regression')

    # # Plot the second figure with color-coded points based on 'price'
    # scatter = axs[1].scatter(np.array(df[column_names[0]]), np.array(df[column_names[1]]), c=df['price'], cmap='cool')
    # axs[1].plot(x, Y_predictions, color='r')
    # axs[1].set_title('My Linear Regression')        

    # # Add colorbar for the scatter plot
    # # fig.colorbar(scatter, ax=axs[1])

    # # Adjust layout for better spacing
    # plt.tight_layout()
    # plt.show()