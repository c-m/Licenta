# Test module used for PoC

import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser
from data_loader import get_data, load_data, preprocess_data, Options
from ensemble_models import random_forest_clf, random_forest_3clf, random_forest_regr
from nn import LABEL_NAMES_BIN, LABEL_NAMES_MULT
from nn import model_eval, regressor_eval, plot_confusion_matrix
from nn import nn_binary_classifier, nn_classifier, nn_regressor
from sklearn.preprocessing import StandardScaler


def load_arguments():
    parser = ArgumentParser()
    parser.add_argument("--features", type=str, default=Options.ALL_FEATURES_AGG,
                        help="Features used")
    parser.add_argument("--problem", type=str, default='clf',
                        help="Problem to test: clf, 3clf, regr")
    parser.add_argument("--model", type=str, default='nn',
                        help="Model used: nn, rf")
    parser.add_argument("--input", action='store_true',
                        help="Provide test input for the trained model")
    args = parser.parse_args()
    return args


def prepare_data(option):
    students_data = load_data()
    all_dataset = students_data[0]
    logs_dataset = students_data[1]

    if option == Options.LOGS_ONLY:
        data = get_data(option, logs_dataset)
    else:
        data = get_data(option, all_dataset)

    (data, scaler) = preprocess_data(data, poly_features=False)
    return data, scaler


def test_model(data, args):
    model = None
    if args.problem == 'clf':
        if args.model == 'nn':
            model = nn_binary_classifier(data)
        elif args.model == 'rf':
            model = random_forest_clf(data)
    elif args.problem == '3clf':
        if args.model == 'nn':
            model = nn_classifier(data)
        elif args.model == 'rf':
            model = random_forest_3clf(data)
    elif args.problem == 'regr':
        if args.model == 'nn':
            model = nn_regressor(data)
        elif args.model == 'rf':
            model = random_forest_regr(data)
    return model


def realtime_test(model, scaler, args, feature_names):

    #inpt = np.array([0.1, 0.2, 0.2, 3.0, 6.0, 5.0, 10, 5, 30], dtype=np.float)

    vals = []
    print 'Enter values for each feature: '
    print
    for f in feature_names:
        v = raw_input(f + ': ')
        vals.append(float(v))
    inpt = np.array(vals, dtype=np.float)
    inpt = inpt.reshape(1, -1)
    print 'Test example:'
    print inpt
    inpt = scaler.transform(inpt)

    print
    print '-'*80
    print 'Realtime test'
    print '-'*80

    if args.problem == 'clf':
        prob = model.predict_proba(inpt)
        print 'Probability distribution:'
        print prob
        print

        _y_test = model.predict(inpt)
        print 'Predicted class:'
        if _y_test == 0:
            print 'Will fail!'
        elif _y_test == 1:
            print 'Will pass!'

    elif args.problem == '3clf':
        prob = model.predict_proba(inpt)
        print 'Probability distribution:'
        print prob
        print

        _y_test = model.predict(inpt)
        print 'Predicted class:'
        if _y_test == 0:
            print '0-4'
        elif _y_test == 1:
            print '5-7'
        elif _y_test == 2:
            print '8-10'

    elif args.problem == 'regr':
        _y_test = model.predict(inpt)
        print 'Predicted grade:'
        print _y_test


def main():

    args = load_arguments()
    option = args.features
    (data, scaler) = prepare_data(option)
    feature_names = data['feature_names']

    model = test_model(data, args)

    if args.input == True:
        realtime_test(model, scaler, args, feature_names)


if __name__ == '__main__':
    main()
