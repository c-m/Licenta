# test various regressor models

import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_grades, preprocess_data
from nn import regressor_eval
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


def polynomial_regression():
    """Polynomial regression using polynomial features and Linear Regression model
    """

    students_data_set = load_grades()
    students_data_set = preprocess_data(students_data_set, poly_features=False)

    X_train = students_data_set['train_data']
    X_test = students_data_set['test_data']

    Y_train = students_data_set['train_continuous_labels'][:,1]
    Y_test = students_data_set['test_continuous_labels'][:,1]

    #construct polynomial features from the coefficients
    poly = PolynomialFeatures(degree=3, interaction_only=False)
    X_train = poly.fit_transform(X_train)
    X_test = poly.fit_transform(X_test)

    regr = LinearRegression(fit_intercept=False)
    regr.fit(X_train, Y_train)

    ridge_regr = Ridge(alpha=5, solver='sag')
    ridge_regr.fit(X_train, Y_train)

    #evaluate regressors
    print '----------LinearRegression----------'
    print 'Coefficients: \n', regr.coef_
    regressor_eval(regr, X_train, Y_train, X_test, Y_test)
    print '----------Ridge----------'
    print 'Coefficients: \n', ridge_regr.coef_
    regressor_eval(ridge_regr, X_train, Y_train, X_test, Y_test)


def main():
    #polynomial regression: extend linear model with a polynomial function
    polynomial_regression()


if __name__ == '__main__':
    main()
