from sklearn.neural_network import MLPClassifier

LABEL_NAMES_BIN = ['failed', 'passed']

def nn_binary_classifier(data):
    # Binary classification with neural networks.
    X_nn = data['train_data']
    X_nn_test = data['test_data']
    # use the exam grades as labels
    Y_nn = data['train_labels'][:,0]
    Y_nn_test = data['test_labels'][:,0]
    # Transform Y_nn and Y_nn_test for binary 
    # classification
    Y_nn[Y_nn < 5] = 0
    Y_nn[Y_nn >= 5] = 1
    Y_nn_test[Y_nn_test < 5] = 0
    Y_nn_test[Y_nn_test >= 5] = 1
    # Use the MLP classifier
    clf = MLPClassifier(algorithm='sgd', 
                        activation='relu', 
                        hidden_layer_sizes=(10,),
                        max_iter=1000,
                        batch_size='auto',
                        learning_rate_init=0.001,)
    # Training process
    clf.fit(X_nn, Y_nn)
    # Evaluate the trained model
    model_eval(clf, X_nn, Y_nn, X_nn_test, Y_nn_test, LABEL_NAMES_BIN)
