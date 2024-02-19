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