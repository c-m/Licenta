# test various classifier models

import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_grades, preprocess_data
from nn import LABEL_NAMES_BIN, LABEL_NAMES_MULT
from nn import plot_confusion_matrix
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix, hamming_loss
from sklearn.preprocessing import PolynomialFeatures


def model_eval(clf, X_nn, Y_nn, X_nn_test, Y_nn_test, label_names):
    
    raw_pred = clf.decision_function(X_nn_test)
    _y_test = clf.predict(X_nn_test)
    print Y_nn_test
    print _y_test
    
    #analyze how good/bad is the model
    #reference: http://scikit-learn.org/dev/modules/model_evaluation.html
    #model score (same as accuracy_score from sklearn.metrics module)
    print("Training set score: %f" % clf.score(X_nn, Y_nn))
    print("Test set score: %f" % clf.score(X_nn_test, Y_nn_test))

    #hamming loss
    hloss = hamming_loss(Y_nn_test, _y_test)
    print 'Hamming loss: %f' % hloss


    #confusion matrix
    c_m = confusion_matrix(Y_nn_test, _y_test)
    print 'Confusion matrix'
    print c_m
    plt.figure()
    plot_confusion_matrix(c_m, label_names)

    np.set_printoptions(precision=2)
    c_m_normalized = c_m.astype('float') / c_m.sum(axis=1)[:, np.newaxis]
    print 'Normalized confusion matrix'
    print c_m_normalized
    plt.figure()
    plot_confusion_matrix(c_m_normalized, label_names)

    plt.show()

    #reset numpy output formatter
    np.set_printoptions(edgeitems=3,infstr='inf',
                        linewidth=75, nanstr='nan', precision=8,
                        suppress=False, threshold=1000, formatter=None)


def polynomial_classification():
    """Classification using the Perceptron algorithm and polynomial features
    """
    
    students_data_set = load_grades()
    students_data_set = preprocess_data(students_data_set, poly_features=True)

    X_train = students_data_set['train_data']
    X_test = students_data_set['test_data']

    # #construct polynomial features from the coefficients
    # poly = PolynomialFeatures(degree=3, interaction_only=False)
    # X_train = poly.fit_transform(X_train)
    # X_test = poly.fit_transform(X_test)

    Y_train = students_data_set['train_discrete_labels']
    Y_test = students_data_set['test_discrete_labels']

    #transform Y_nn and Y_nn_test 
    Y_train[Y_train < 5] = 0
    Y_train[Y_train >= 5] = 1

    Y_test[Y_test < 5] = 0
    Y_test[Y_test >= 5] = 1

    clf = Perceptron(penalty=None, fit_intercept=False, n_iter=1000, shuffle=False, verbose=False)
    clf.fit(X_train, Y_train)

    #evaluate the perceptron model with polynomial features
    model_eval(clf, X_train, Y_train, X_test, Y_test, LABEL_NAMES_BIN)


def main():
    #polynomial classification: extend linear model with a polynomial function
    polynomial_classification()


if __name__ == '__main__':
    main()
