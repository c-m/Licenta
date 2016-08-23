# load datasets from files

import csv
import numpy as np
import os

from collections import OrderedDict
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler


DATA_PATH = '../data_sets/'
GRADE_FEATURES_FILE = 'note_pp.csv'
LOG_FEATURES_FILE = 'logs_pp.csv'
TRAIN_TO_TEST_RATIO = 0.8


class DatasetContaier(dict):

	def __init__(self, **kwargs):
		super(DatasetContaier, self).__init__(kwargs)

	
	def __setattr__(self, key, value):
		self[key] = value


	def __getattr__(self, key):
		try:
			return self[key]
		except KeyError:
			raise AttributeError(key)


	def __setstate__(self, state):
		pass


def load_grades():
	"""Load and return student grades dataset from files
	"""

	program_path = os.path.abspath(os.path.dirname(__file__))

	all_dataset = OrderedDict()
	logs_dataset = OrderedDict()

	with open(os.path.join(program_path, DATA_PATH, GRADE_FEATURES_FILE)) as csv_file:
		
		reader = csv.reader(csv_file)
		fieldnames = reader.next()

		for example in reader:
			name = example[0]
			all_dataset[name] = OrderedDict()
			all_dataset[name]['features'] = OrderedDict()
			for i in range(1, len(fieldnames)-2):
				if fieldnames[i] != 'total':
					all_dataset[name]['features'][fieldnames[i]] = float(example[i])

			# aggregate some of the features(hw, tests, past results) 
			# to try a generalization of the model (e.g.: apply it to other 
			# courses, even if they're different)
			all_dataset[name]['features']['hw_avg'] = float('%.3f' % \
				((all_dataset[name]['features']['hw_1'] + \
				  all_dataset[name]['features']['hw_2'] + \
				  all_dataset[name]['features']['hw_3']) / 3))
			all_dataset[name]['features']['t_avg'] = float('%.3f' % \
				(((all_dataset[name]['features']['t_1'] + \
				   all_dataset[name]['features']['t_2'] + \
				   all_dataset[name]['features']['t_3'] + \
				   all_dataset[name]['features']['t_4'] / 4) + \
				   all_dataset[name]['features']['t_f']) / 2))
			all_dataset[name]['features']['past_results_avg'] = float('%.3f' % \
				((all_dataset[name]['features']['PC_grade'] + \
				  all_dataset[name]['features']['AA_grade']) / 2))

			all_dataset[name]['labels'] = OrderedDict()
			all_dataset[name]['labels'][fieldnames[-2]] = float(example[-2])
			all_dataset[name]['labels'][fieldnames[-1]] = float(example[-1])

	n_examples = len(all_dataset)
	n_total_features = len(all_dataset.values()[0]['features'])


	# data_set = np.zeros((n_examples, n_features))
	# labels_set = np.zeros(n_examples)

	# for i, ex in enumerate(data):
	# 	data_set[i] = np.array(ex[:-1], dtype=np.float)
	# 	discrete_labels_set[i] = np.array(ex[-1], dtype=np.int)


	with open(os.path.join(program_path, DATA_PATH, LOG_FEATURES_FILE)) as csv_file:
		
		reader = csv.reader(csv_file)
		fieldnames = reader.next()

		for example in reader:
			name = example[0]
			logs_dataset[name] = OrderedDict()
			if name in all_dataset:
				for i in range(1, len(fieldnames)):
					all_dataset[name]['features'][fieldnames[i]] = float(example[i])
			for i in range(1, len(fieldnames)):
				logs_dataset[name][fieldnames[i]] = float(example[i])

	n_examples = len(all_dataset)
	n_total_features = len(all_dataset.values()[0]['features'])

	for k, v in all_dataset.iteritems():
		print k, v
	return all_dataset

	# continuous_labels_set = np.zeros((n_examples, 2))

	# for i, ex in enumerate(data):
	# 	continuous_labels_set[i] = np.array(ex, dtype=np.float)

	
	# n_training = int(np.ceil(n_examples*TRAIN_TO_TEST_RATIO))
	# n_test = n_examples - n_training
	
	# train_data_set = data_set[:n_training]
	# test_data_set = data_set[n_training:]

	# train_discrete_labels_set = discrete_labels_set[:n_training]
	# test_discrete_labels_set = discrete_labels_set[n_training:]

	# train_continuous_labels_set = continuous_labels_set[:n_training]
	# test_continuous_labels_set = continuous_labels_set[n_training:]
	
	# return DatasetContaier(train_data=train_data_set, test_data=test_data_set,
	# 					   train_discrete_labels=train_discrete_labels_set, test_discrete_labels=test_discrete_labels_set,
	# 					   train_continuous_labels=train_continuous_labels_set, test_continuous_labels=test_continuous_labels_set,
	# 					   feature_weight=feature_w, test_factor=test_adjustment_factor,
	# 					   feature_names=feature_names)


def preprocess_data(data, poly_features=False):
	"""Apply some preprocessing before using the dataset
	"""

	#scale test grades to the same interval as hw grades (0-0.5)
	train_data = data['train_data']
	test_data = data['test_data']

	i = 0
	for feature in data['feature_names']:
		if feature[0] == 't':
			train_data[:,i] /= data['test_factor']
			test_data[:,i] /= data['test_factor']
		i += 1

	#PCA visualization
	plot_pca = True
	if plot_pca:
		pca = PCA(n_components=2)
		all_examples = np.vstack((train_data, test_data))
		X_r  = pca.fit(all_examples).transform(all_examples)
		print all_examples
		print pca.components_
		# Percentage of variance explained for each components
		print('explained variance ratio (first two components): %s'
      		  % str(pca.explained_variance_ratio_))
		plt.figure()
		target_names = ['failed', 'passed']
		y = np.hstack((data['train_discrete_labels'], data['test_discrete_labels']))
		#transform final_grades for binary classification (failed/passed)
		y[y < 5] = 0
		y[y >= 5] = 1
		for c, i, target_name in zip("rg", [0, 1], target_names):
			plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
		plt.legend()
		plt.title('PCA of students dataset')
		plt.show()

	#scale the dataset to have the mean=0 and variance=1
	scaler = StandardScaler()
	scaler.fit(train_data)
	train_data = scaler.transform(train_data)
	#apply same transformation to test_data
	test_data = scaler.transform(test_data)

	if poly_features == True:
		poly = PolynomialFeatures(degree=train_data.shape[1], interaction_only=True)
		train_data = poly.fit_transform(train_data)
		test_data = poly.fit_transform(test_data)

	data['train_data'] = train_data
	data['test_data'] = test_data

	return data


def main():
	students_data = load_grades()
	#print students_data['test_data']
'''
	students_data = preprocess_data(students_data, poly_features=False)
	#print students_data['test_data'][0]

	sum_features = np.hstack((students_data['train_continuous_labels'][:,0], students_data['test_continuous_labels'][:,0]))
	exam_grades = np.hstack((students_data['train_continuous_labels'][:,1], students_data['test_continuous_labels'][:,1]))
	sum_features = sum_features - exam_grades
	final_grades = np.hstack((students_data['train_discrete_labels'], students_data['test_discrete_labels']))

	#transform final_grades for binary classification (failed/passed)
	final_grades[final_grades < 5] = 0
	final_grades[final_grades >= 5] = 1

	#transform exam_grades to match the four classes (0-1,1-2,2-3,3-4)
	exam_grades = np.ceil(exam_grades)

	#Encode labels to values: 0,1,2,3
	le = LabelEncoder()
	le.fit(exam_grades)
	exam_grades = le.transform(exam_grades)


	plt.xlabel("sum(features) == semester points")
	plt.ylabel("exam_grade")
	plot_data = plt.plot(sum_features, final_grades, 'ro', 
						 label = 'Final grades based on semester points\n0=failed, 1=passed')
	plt.axis([2, 7, -1, 2])
	plt.legend()
	plt.show()
'''
if __name__ == '__main__':
	main()
