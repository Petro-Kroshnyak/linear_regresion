#Set interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Load diabetes data set
from sklearn.datasets import load_diabetes


def get_X_y(features=None, verbose=False):
    X, y = load_diabetes(return_X_y=True)

    if features is None:
        print('Selecting all features')

    elif type(features) == int or (type(features) == list and len(features) == 1):
        print('Selecting one feature: {}'.format(features))
        X = X[:, features].reshape(-1, 1)  # single column
    elif type(features) == list:
        print('Selecting features list: {}'.format(features))
        X = X[:, features]
    else:
        print('wrong format of parameter "features"')
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2021)
    if verbose:
        print('X_train.shape= ', X_train.shape)
        print('y_train.shape= ', y_train.shape)
        print('X_train [:5] = \n{}'.format(X_train[:5]))
        print('y_train [:5] = \n{}'.format(y_train[:5]))
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = get_X_y(verbose= True)

#X, y = load_diabetes(return_X_y=True)
## X= X[:,[5,7,-3,-1]]
#df = pd.DataFrame (X)
#df['target'] = y
#pd.plotting.scatter_matrix(df);
#plt.savefig('scatter_matrix.png')

X_train, X_test, y_train, y_test=  get_X_y(verbose= True)

from sklearn.linear_model import Ridge
ridge_reg=Ridge()
ridge_reg.fit(X_train,y_train)
regressor = ridge_reg
print ('Ridge')
print ('R2 train score =', regressor.score(X_train, y_train))
print ('R2 test score =', regressor.score(X_test, y_test))
print ('b: {}, \nw= {}'.format(regressor.intercept_, regressor.coef_))

from sklearn.linear_model import Lasso
lasso_reg=Lasso()
lasso_reg.fit(X_train,y_train)
regressor = lasso_reg
print ('Lasso')
print ('R2 train score =', regressor.score(X_train, y_train))
print ('R2 test score =', regressor.score(X_test, y_test))
print ('b: {}, \nw= {}'.format(regressor.intercept_, regressor.coef_))

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
X_train, X_test, y_train, y_test=  get_X_y()

poly= PolynomialFeatures(degree=2,include_bias=False) # default is True means to return the first feature of all 1 as for degree 0
X_train_poly= poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
print ('X_train.shape= ',X_train.shape)
print ('X_train_poly.shape= ',X_train_poly.shape)
# X_train_poly[:5]
poly_lin_reg = LinearRegression().fit (X_train_poly,y_train)
regressor = poly_lin_reg
print ('Polynomial + Linear Regression')
print ('R2 train score =', regressor.score(X_train_poly, y_train))
print ('R2 test score =', regressor.score(X_test_poly, y_test))
print ('b: {}, \nw= {}'.format(regressor.intercept_, regressor.coef_))