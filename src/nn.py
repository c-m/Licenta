# neural network models for data learning

import matplotlib.pyplot as plt
import numpy as np

from data_loader import get_data, load_data, preprocess_data, Options
from sklearn.metrics import confusion_matrix, hamming_loss, brier_score_loss, log_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder


LABEL_NAMES_BIN = ['failed', 'passed']
LABEL_NAMES_MULT = ['0-1', '1-2', '2-3', '3-4']


def plot_confusion_matrix(cm, label_names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def model_eval(clf, X_nn, Y_nn, X_nn_test, Y_nn_test, label_names):
	prob = clf.predict_proba(X_nn_test)
	print prob
	
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

	#brier score loss
	i = 0
	t = []
	for p in prob:
		if Y_nn_test[i] == 0:
			t.append(p[0])
		else:
			t.append(p[1])
		i += 1
	y_prob = np.array(t)
	print y_prob

	brier_loss = brier_score_loss(Y_nn_test, y_prob)
	print 'Brier score loss (lower is better): %f' % brier_loss

	#cross-entropy loss -> this is an important metric for ML models
	#more on this: http://colah.github.io/posts/2015-09-Visual-Information/
	#H(p(x)) = sum(p(x)*log(1/p(x))); H is the entropy function
	#H_q(p) = sum(p*log(1/q))
	#We are going to use the following formula for evaluating the model:
	#Kullback-Leibler divergence - KL divergence of p with respect to q: 
	#D_q(p) = H_q(p) - H(p) = sum(p*log(1/q)) - sum(p*log(1/p)), for all x,
	#where x is a random variable
	#In our case, the second sum is zero, since p(x) = 0 or 1
	#We have to sum all these divergences for all the test examples and, 
	#in the end, average them for the formula to be correct.
	KL_div = -np.sum(np.log(y_prob)) / _y_test.shape
	print 'KL divergence: %f' % KL_div
	#Alternatively, we could use this function from sk-learn: 
	print log_loss(Y_nn_test, prob)

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


def regressor_eval(clf, X_nn, Y_nn, X_nn_test, Y_nn_test):
	_y_train = clf.predict(X_nn)
	_y_test = clf.predict(X_nn_test)
	print Y_nn_test
	print _y_test

	print '----------Train----------'
	print("Training set score == R_2 score: %f" % clf.score(X_nn, Y_nn))
	print 'Mean absolute error: %f' % mean_absolute_error(Y_nn, _y_train)
	mse = mean_squared_error(Y_nn, _y_train)
	print 'Mean squared error: %f. RMSE: %f' % (mse, np.sqrt(mse))
	print 'Median absolute error: %f' % median_absolute_error(Y_nn, _y_train)
	#r2_score
	#It provides a measure of how well future samples are likely to be predicted by the model. 
	#Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 
	#A constant model that always predicts the expected value of y, disregarding the input features, 
	#would get a R^2 score of 0.0. Source: http://scikit-learn.org/dev/modules/model_evaluation.html
	print 'R_2 score: %f' % r2_score(Y_nn, _y_train)

	print '----------Test----------'
	print("Test set score == R_2 score: %f" % clf.score(X_nn_test, Y_nn_test))
	print 'Mean absolute error: %f' % mean_absolute_error(Y_nn_test, _y_test)
	mse = mean_squared_error(Y_nn_test, _y_test)
	print 'Mean squared error: %f. RMSE: %f' % (mse, np.sqrt(mse))
	print 'Median absolute error: %f' % median_absolute_error(Y_nn_test, _y_test)
	print 'R_2 score: %f' % r2_score(Y_nn_test, _y_test)


def nn_binary_classifier(data):
	"""Classify the dataset in two classes ('failed' and 'passed')
	using fully-connected neural networks.
	"""
	
	X_nn = data['train_data']
	X_nn_test = data['test_data']
	Y_nn = data['train_labels'][:,0]
	Y_nn_test = data['test_labels'][:,0]

	#transform Y_nn and Y_nn_test 
	Y_nn[Y_nn < 5] = 0
	Y_nn[Y_nn >= 5] = 1

	Y_nn_test[Y_nn_test < 5] = 0
	Y_nn_test[Y_nn_test >= 5] = 1

	clf = MLPClassifier(algorithm='sgd', 
						alpha=1e-5,
						activation='relu', 
						hidden_layer_sizes=(100,),
						random_state=1,
						max_iter=1000,
						batch_size='auto',
						learning_rate='constant',
						learning_rate_init=0.001,
						verbose=False)

	clf.fit(X_nn, Y_nn)

	#evaluate de trained model
	model_eval(clf, X_nn, Y_nn, X_nn_test, Y_nn_test, LABEL_NAMES_BIN)


def nn_classifier(data):
	"""Classify students data in four classes based on the final 
	exam grade, which is a real number between 0.0 and 4.0.
	"""
    
	X_nn = data['train_data']
	X_nn_test = data['test_data']
	Y_nn = data['train_continuous_labels'][:,1]
	Y_nn_test = data['test_continuous_labels'][:,1]

	#transform Y_nn and Y_nn_test
	Y_nn = np.ceil(Y_nn)
	Y_nn_test = np.ceil(Y_nn_test)

	#Encode labels to values: 0,1,2,3
	le = LabelEncoder()
	le.fit(Y_nn)
	le.fit(Y_nn_test)

	Y_nn = le.transform(Y_nn)
	Y_nn_test = le.transform(Y_nn_test)

	clf = MLPClassifier(algorithm='adam', 
						alpha=1e-5,
						activation='tanh',
						hidden_layer_sizes=(100,),
						random_state=0,
						max_iter=500,
						batch_size='auto',
						learning_rate='constant',
						learning_rate_init=0.001,
						verbose=True)

	clf.fit(X_nn, Y_nn)

	#evaluate de trained model
	model_eval(clf, X_nn, Y_nn, X_nn_test, Y_nn_test, LABEL_NAMES_MULT)


def nn_regressor(data):
	"""Perform regression on students data using exam grades as continuous output values.
	Exam grade is a real value between 0.0 and 4.0 and it will remain as is for the regressor.
	"""

	X_nn = data['train_data']
	X_nn_test = data['test_data']
	Y_nn = data['train_continuous_labels'][:,1]
	Y_nn_test = data['test_continuous_labels'][:,1]
	
	clf = MLPRegressor(algorithm='sgd', 
						alpha=1e-5,
						activation='tanh',
						hidden_layer_sizes=(100,),
						random_state=1,
						max_iter=500,
						batch_size='auto',
						learning_rate='constant',
						learning_rate_init=0.001,
						verbose=True)

	clf.fit(X_nn, Y_nn)

	#evaluate the MLPRegressor model
	regressor_eval(clf, X_nn, Y_nn, X_nn_test, Y_nn_test)


def main():

	students_data = load_data()
	all_dataset = students_data[0]
	logs_dataset = students_data[1]

	option = Options.AGG_GRADES_ONLY
	if option == Options.LOGS_ONLY:
	    data = get_data(option, logs_dataset)
	else:
	    data = get_data(option, all_dataset)

	data = preprocess_data(data, poly_features=False)

	#binary classificication based on final grade
	nn_binary_classifier(data)

	#4-class classification based on exam grade
	# nn_classifier(students_data_set)

	#regression for exam_grades using MLPRegressor() class
	#nn_regressor(students_data_set)


if __name__ == '__main__':
	main()
