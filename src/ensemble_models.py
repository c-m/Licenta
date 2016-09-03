# test some ensemble methods such as
# Random Forests

import matplotlib.pyplot as plt
import numpy as np

from data_loader import get_data, load_data, preprocess_data, Options
from nn import LABEL_NAMES_BIN, LABEL_NAMES_MULT
from nn import model_eval, regressor_eval, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def random_forest_clf(data):

    X_train = data['train_data']
    X_test = data['test_data']
    Y_train = data['train_labels'][:,1]
    Y_test = data['test_labels'][:,1]

    #print X_train

    #transform Y_nn and Y_nn_test 
    Y_train[Y_train < 5] = 0
    Y_train[Y_train >= 5] = 1

    Y_test[Y_test < 5] = 0
    Y_test[Y_test >= 5] = 1

    #max_features values: sqrt(n_features)/2, sqrt(n_features), 2*sqrt(n_features)
    #n_features == sqrt(10) ~ 3.16
    #clf = RandomForestClassifier(n_estimators=10, max_features='sqrt', oob_score=True)
    clf = ExtraTreesClassifier(n_estimators=10, max_features='sqrt')
    clf.fit(X_train, Y_train)

    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)

    indices = np.argsort(importances)[::-1]
    idx_names = list()
    for i in range(X_train.shape[1]):
        idx_names.append(data['feature_names'][indices[i]])

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d, name: %s, (%f)" % \
            (f + 1, indices[f], data['feature_names'][indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), idx_names)
    plt.xlim([-1, X_train.shape[1]])
    #plt.show()

    model_eval(clf, X_train, Y_train, X_test, Y_test, LABEL_NAMES_BIN)
    #print 'Out-of-bag error score: %f' % clf.oob_score_
    #print clf.oob_decision_function_


def random_forest_3clf(data):

    X_train = data['train_data']
    X_test = data['test_data']
    Y_train = data['train_labels'][:,1]
    Y_test = data['test_labels'][:,1]

    #print X_train.shape, X_test.shape

    #transform Y_nn and Y_nn_test 
    Y_train_list = Y_train.tolist()
    Y_test_list = Y_test.tolist()
    for i in range(0, len(Y_train_list)):
        if Y_train_list[i] < 5:
            Y_train_list[i] = 0
        if Y_train_list[i] >= 5 and Y_train_list[i] <= 7:
            Y_train_list[i] = 1
        if Y_train_list[i] > 7:
            Y_train_list[i] = 2
    for i in range(0, len(Y_test_list)):
        if Y_test_list[i] < 5:
            Y_test_list[i] = 0
        if Y_test_list[i] >= 5 and Y_test_list[i] <= 7:
            Y_test_list[i] = 1
        if Y_test_list[i] > 7:
            Y_test_list[i] = 2
    Y_train = np.array(Y_train_list, dtype=int)
    Y_test = np.array(Y_test_list, dtype=int)

    #max_features values: sqrt(n_features)/2, sqrt(n_features), 2*sqrt(n_features)
    #n_features == sqrt(10) ~ 3.16
    #clf = RandomForestClassifier(n_estimators=10, max_features='sqrt', oob_score=True)
    clf = ExtraTreesClassifier(n_estimators=10, max_features='sqrt')
    clf.fit(X_train, Y_train)

    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)

    indices = np.argsort(importances)[::-1]
    idx_names = list()
    for i in range(X_train.shape[1]):
        idx_names.append(data['feature_names'][indices[i]])

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d, name: %s, (%f)" % \
            (f + 1, indices[f], data['feature_names'][indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), idx_names)
    plt.xlim([-1, X_train.shape[1]])
    #plt.show()

    model_eval(clf, X_train, Y_train, X_test, Y_test, LABEL_NAMES_MULT)
    #print 'Out-of-bag error score: %f' % clf.oob_score_
    #print clf.oob_decision_function_


def random_forest_regr(data):

    X_train = data['train_data']
    X_test = data['test_data']
    Y_train = data['train_labels'][:,0]
    Y_test = data['test_labels'][:,0]

    regr = RandomForestRegressor(n_estimators=10, max_features='auto')
    regr.fit(X_train, Y_train)

    importances = regr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regr.estimators_],
             axis=0)

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X_train.shape[1]):
        print("%d. feature %d, name: %s, (%f)" % \
            (f + 1, indices[f], data['feature_names'][indices[f]], importances[indices[f]]))

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
    students_data = load_data()
    all_dataset = students_data[0]
    logs_dataset = students_data[1]

    option = Options.ALL_FEATURES_AGG
    if option == Options.LOGS_ONLY:
        data = get_data(option, logs_dataset)
    else:
        data = get_data(option, all_dataset)

    data = preprocess_data(data, poly_features=False, scale=False)

    #classification with RandomForestClassifier
    random_forest_3clf(data)

    #regression with RandomForestRegressor
    #random_forest_regr(data)

    #classification with AdaBoostClassifier
    #adaboost_clf(students_data_set)

    #regression with AdaBoostRegressor
    #for i in range(10):
    #adaboost_regr(students_data_set)


if __name__ == '__main__':
    main()
