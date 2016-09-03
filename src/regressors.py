# test various regressor models

import matplotlib.pyplot as plt
import numpy as np

from data_loader import get_data, load_data, preprocess_data, Options
from nn import regressor_eval
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


def polynomial_regression():
    """Polynomial regression using polynomial features and Linear Regression model
    """

    students_data = load_data()
    all_dataset = students_data[0]
    logs_dataset = students_data[1]

    option = Options.ALL_FEATURES_AGG
    if option == Options.LOGS_ONLY:
        data = get_data(option, logs_dataset)
    else:
        data = get_data(option, all_dataset)

    data = preprocess_data(data, poly_features=False)

    X_train = data['train_data']
    X_test = data['test_data']

    Y_train = data['train_labels'][:,1]
    Y_test = data['test_labels'][:,1]

    #construct polynomial features from the coefficients
    poly = PolynomialFeatures(degree=2, interaction_only=False)
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
