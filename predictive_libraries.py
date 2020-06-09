from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np

import numpy.polynomial.polynomial as poly

np.random.seed(123)

def df_to_linear_fit(df,colX,colY,wgt=None):

    X = df[colX].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = df[colY].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y,sample_weight=wgt)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    coef = float(linear_regressor.coef_)

    return Y_pred,coef

def df_to_exponential_fit(df,colX,colY,wgt=None):

    X = df[colX].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = df[colY].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column

    # Y = np.log(df[colY].values.reshape(-1, 1)) # -1 means that calculate the dimension of rows, but have 1 column
    transformer = FunctionTransformer(np.log, validate=True)
    y_trans = transformer.fit_transform(Y)     

    linear_regressor = LinearRegression()  # create object for the class
    results = linear_regressor.fit(X, y_trans,sample_weight=wgt)

    linear_regressor.fit(X, y_trans,sample_weight=wgt)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    coef = float(linear_regressor.coef_)

    return Y_pred,coef

def df_to_polynomial_fit(df,colX,colY,power,wgt=None,x_new=None):

	# X = df[colX].values.reshape(-1, 1)  # values converts it into a numpy array
	# Y = df[colY].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
	X = df[colX].squeeze().T
	Y = df[colY].squeeze().T

	coefs = poly.polyfit(X,Y,power)

	if x_new is None: x_new = np.linspace(0, 40, num=100)
	ffit = poly.polyval(x_new, coefs)

	return x_new,ffit

