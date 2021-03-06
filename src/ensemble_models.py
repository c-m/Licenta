# test some ensemble methods such as
# Random Forests

import matplotlib.pyplot as plt
import numpy as np

from data_loader import load_grades, preprocess_data
from nn import LABEL_NAMES_BIN, LABEL_NAMES_MULT
from nn import model_eval, regressor_eval, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def random_forest_clf(data):

    X_train = data['train_data']
    X_test = data['test_data']
    Y_train = data['train_discrete_labels']
    Y_test = data['test_discrete_labels']

    #transform Y_nn and Y_nn_test 
    Y_train[Y_train < 5] = 0
    Y_train[Y_train >= 5] = 1

    Y_test[Y_test < 5] = 0
    Y_test[Y_test >= 5] = 1

    #max_features values: sqrt(n_features)/2, sqrt(n_features), 2*sqrt(n_features)
    #n_features == sqrt(10) ~ 3.16
    clf = RandomForestClassifier(n_estimators=10, max_features=5, oob_score=True)
    clf.fit(X_train, Y_train)

    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    #plt.show()

    model_eval(clf, X_train, Y_train, X_test, Y_test, LABEL_NAMES_BIN)
    print 'Out-of-bag error score: %f' % clf.oob_score_
    print clf.oob_decision_function_

def random_forest_regr(data):

    X_train = data['train_data']
    X_test = data['test_data']
    Y_train = data['train_continuous_labels'][:,1]
    Y_test = data['test_continuous_labels'][:,1]

    regr = RandomForestRegressor(n_estimators=10, max_features='auto')
    regr.fit(X_train, Y_train)

    importances = regr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regr.estimators_],
             axis=0)

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    regressor_eval(regr, X_train, Y_train, X_test, Y_test)


def adaboost_clf(data):
    
    X_train = data['train_data']
    X_test = data['test_data']
    Y_train = data['train_discrete_labels']
    Y_test = data['test_discrete_labels']

    #transform Y_nn and Y_nn_test 
    Y_train[Y_train < 5] = 0
    Y_train[Y_train >= 5] = 1

    Y_test[Y_test < 5] = 0
    Y_test[Y_test >= 5] = 1

    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), algorithm="SAMME", n_estimators=100)
    clf.fit(X_train, Y_train)

    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    #plt.show()

    model_eval(clf, X_train, Y_train, X_test, Y_test, LABEL_NAMES_BIN)


def adaboost_regr(data):
    
    X_train = data['train_data']
    X_test = data['test_data']
    Y_train = data['train_continuous_labels'][:,1]
    Y_test = data['test_continuous_labels'][:,1]

    regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=10, loss='linear')
    regr.fit(X_train, Y_train)

    importances = regr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regr.estimators_],
             axis=0)

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    regressor_eval(regr, X_train, Y_train, X_test, Y_test)


def main():
    students_data_set = load_grades()
    students_data_set = preprocess_data(students_data_set, poly_features=False)

    #classification with RandomForestClassifier
    #random_forest_clf(students_data_set)

    #regression with RandomForestRegressor
    #random_forest_regr(students_data_set)

    #classification with AdaBoostClassifier
    #adaboost_clf(students_data_set)

    #regression with AdaBoostRegressor
    #for i in range(10):
    adaboost_regr(students_data_set)


if __name__ == '__main__':
    main()
