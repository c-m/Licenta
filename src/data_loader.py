# load datasets from files

import csv
import numpy as np
import os

from collections import OrderedDict
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

DATA_PATH = '../data_sets/'
GRADE_FEATURES_FILE = 'note_pp.csv'
LOG_FEATURES_FILE = 'logs_pp.csv'
TRAIN_TO_TEST_RATIO = 0.8

# SILLY HACK - global vars
GRADE_FIELDS = list()
LOG_FIELDS = list()

class Options(object):

	GRADES_ONLY = 'grades_only' # hw_1,hw_2,hw_3,t_1,t_2,t_3,t_4,t_f,lab,lecture,aa_grade,pc_grade
	LOGS_ONLY = 'logs_only' # x1,x2,x3,x4 - all logs, even those without labels. UNLABELED
	AGG_GRADES_ONLY = 'agg_grades_only' # hw_avg,t_avg,past_results_avg,lab,lecture
	ALL_FEATURES = 'all_features' # GRADES_ONLY + LOGS_ONLY (intersection of sets, by name as key)
	ALL_FEATURES_AGG = 'all_features_agg' # AGG_GRADES_ONLY + LOGS_ONLY (as above)
	ALL_LOGS = 'all_logs' # LOGS_ONLY. LABELED

	def __init__(self):
		pass


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


def load_data():
	"""Load and return students' data dataset from .csv files
	"""

	program_path = os.path.abspath(os.path.dirname(__file__))

	all_dataset = OrderedDict()
	logs_dataset = OrderedDict()

	global GRADE_FIELDS, LOG_FIELDS

	with open(os.path.join(program_path, DATA_PATH, GRADE_FEATURES_FILE)) as csv_file:
		
		reader = csv.reader(csv_file)
		fieldnames = reader.next()
		GRADE_FIELDS = fieldnames
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
				   all_dataset[name]['features']['t_4']) / 4 + \
				   all_dataset[name]['features']['t_f']) / 2))
			all_dataset[name]['features']['past_results_avg'] = float('%.3f' % \
				((all_dataset[name]['features']['PC_grade'] + \
				  all_dataset[name]['features']['AA_grade']) / 2))

			all_dataset[name]['labels'] = OrderedDict()
			all_dataset[name]['labels'][fieldnames[-2]] = float(example[-2])
			all_dataset[name]['labels'][fieldnames[-1]] = float(example[-1])

	with open(os.path.join(program_path, DATA_PATH, LOG_FEATURES_FILE)) as csv_file:
		
		reader = csv.reader(csv_file)
		fieldnames = reader.next()
		LOG_FIELDS = fieldnames

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

	# for k, v in all_dataset.iteritems():
	# 	print k, v

	return (all_dataset, logs_dataset)
	

def get_data(option, dataset):

	n_examples = len(dataset)
	names = dataset.keys()

	if option == Options.LOGS_ONLY:
		n_total_features = len(dataset.values()[0])
	else:
		n_total_features = len(dataset.values()[0]['features'])

	if option == Options.GRADES_ONLY:
		n_features = 12

		dataset_out = np.zeros((n_examples, n_features))
		labels_out = np.zeros((n_examples, 2))

		feature_names = list()
		feature_names = filter(lambda x: x != 'total', GRADE_FIELDS[1:n_features+2])

		for i, v in enumerate(dataset.itervalues()):
			features = v['features'].values()[:n_features]
			labels = v['labels'].values()
			dataset_out[i] = np.array(features, dtype=np.float)
			labels_out[i] = np.array(labels, dtype=np.float)
	elif option == Options.AGG_GRADES_ONLY:
		n_features = 5

		dataset_out = np.zeros((n_examples, n_features))
		labels_out = np.zeros((n_examples, 2))

		feature_names = ['lab', 'lecture', 'hw_avg', 't_avg', 'past_results_avg']

		for i, v in enumerate(dataset.itervalues()):
			features = list()
			for f_name in feature_names:
				if f_name in v['features']:
					features.append(v['features'][f_name])
			labels = v['labels'].values()
			dataset_out[i] = np.array(features, dtype=np.float)
			labels_out[i] = np.array(labels, dtype=np.float)
	elif option == Options.ALL_FEATURES:
		n_features = n_total_features - 3

		dataset_out = np.zeros((n_examples, n_features))
		labels_out = np.zeros((n_examples, 2))

		feature_names = list()
		feature_names = filter(lambda x: x != 'avg' and x != 'total', GRADE_FIELDS[1:-2])
		feature_names += LOG_FIELDS[1:]

		for i, v in enumerate(dataset.itervalues()):
			features = list()
			for f_name in feature_names:
				if f_name in v['features']:
					features.append(v['features'][f_name])
			labels = v['labels'].values()
			dataset_out[i] = np.array(features, dtype=np.float)
			labels_out[i] = np.array(labels, dtype=np.float)
	elif option == Options.ALL_FEATURES_AGG:
		n_features = 9

		dataset_out = np.zeros((n_examples, n_features))
		labels_out = np.zeros((n_examples, 2))

		feature_names = ['lab', 'lecture', 'hw_avg', 't_avg', 'past_results_avg']
		feature_names += LOG_FIELDS[1:]

		for i, v in enumerate(dataset.itervalues()):
			features = list()
			for f_name in feature_names:
				if f_name in v['features']:
					features.append(v['features'][f_name])
			labels = v['labels'].values()
			dataset_out[i] = np.array(features, dtype=np.float)
			labels_out[i] = np.array(labels, dtype=np.float)
	elif option == Options.LOGS_ONLY:
		n_features = n_total_features

		dataset_out = np.zeros((n_examples, n_features))
		feature_names = LOG_FIELDS[1:]

		for i, v in enumerate(dataset.itervalues()):
			features = v.values()[:n_features]
			dataset_out[i] = np.array(features, dtype=np.float)
	elif option == Options.ALL_LOGS:
		n_features = 4

		dataset_out = np.zeros((n_examples, n_features))
		labels_out = np.zeros((n_examples, 2))

		feature_names = LOG_FIELDS[1:]

		for i, v in enumerate(dataset.itervalues()):
			features = list()
			for f_name in feature_names:
				if f_name in v['features']:
					features.append(v['features'][f_name])
			labels = v['labels'].values()
			dataset_out[i] = np.array(features, dtype=np.float)
			labels_out[i] = np.array(labels, dtype=np.float)
		print dataset_out

	if option != Options.LOGS_ONLY:
		n_training = int(np.ceil(n_examples*TRAIN_TO_TEST_RATIO))
		n_test = n_examples - n_training
		
		train_data_set = dataset_out[:n_training]
		test_data_set = dataset_out[n_training:]

		train_labels_set = labels_out[:n_training]
		test_labels_set = labels_out[n_training:]

		#print labels_out.shape
		#print labels_out[labels_out[:,0]>=4].shape
		#print dataset_out
		#print labels_out

		return DatasetContaier(train_data=train_data_set, test_data=test_data_set,
	 					       train_labels=train_labels_set, test_labels=test_labels_set,
	 					       feature_names=feature_names, example_names=names)
	else:
		return DatasetContaier(data_set=dataset_out, 
							   feature_names=feature_names, 
			                   example_names=names)


def preprocess_data(data, poly_features=False):
	"""Apply some preprocessing before using the dataset
	"""

	#scale test grades to the same interval as hw grades (0-0.5)
	train_data = data['train_data']
	test_data = data['test_data']

	#PCA visualization
	plot_pca = True
	if plot_pca:
		pca = PCA(n_components=3)
		all_examples = np.vstack((train_data, test_data))
		#print all_examples
		pca.fit(all_examples)
		X_r = pca.transform(all_examples)

		#print pca.components_
		# Percentage of variance explained for each components
		print('explained variance ratio (first two components): %s'
      		  % str(pca.explained_variance_ratio_))
		#plt.figure()
		target_names = ['failed', 'passed']
		y = np.hstack((data['train_labels'][:,0], data['test_labels'][:,0]))
		#print y
		#transform final_grades for binary classification (failed/passed)
		y[y < 5] = 0
		y[y >= 5] = 1
		#print y
		# for c, i, target_name in zip("rg", [0, 1], target_names):
		# 	plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
		# plt.legend()
		# plt.title('PCA of PP students dataset')
		# plt.show()

		# To getter a better understanding of interaction of the dimensions
		# plot the first three PCA dimensions
		fig = plt.figure(1, figsize=(8, 6))
		ax = Axes3D(fig, elev=-150, azim=110)
		ax.scatter(X_r[:, 0], X_r[:, 1], X_r[:, 2], c=y,
		           cmap=plt.cm.Paired)
		ax.set_title("First three PCA directions")
		ax.set_xlabel("1st eigenvector")
		ax.w_xaxis.set_ticklabels([])
		ax.set_ylabel("2nd eigenvector")
		ax.w_yaxis.set_ticklabels([])
		ax.set_zlabel("3rd eigenvector")
		ax.w_zaxis.set_ticklabels([])

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
	students_data = load_data()
	all_dataset = students_data[0]
	logs_dataset = students_data[1]

	option = Options.ALL_LOGS
	if option == Options.LOGS_ONLY:
		data = get_data(option, logs_dataset)
	else:
		data = get_data(option, all_dataset)

	data = preprocess_data(data, poly_features=False)	

	# plot labels
	exam_grades = np.hstack((data['train_labels'][:,0], data['test_labels'][:,0]))
	final_grades = np.hstack((data['train_labels'][:,1], data['test_labels'][:,1]))

	final_grades = np.round(final_grades)
	exam_grades = np.round(exam_grades)

	plt.xlabel("exam_grade")
	plt.ylabel("final_grade")
	plot_data = plt.plot(exam_grades, final_grades, 'bs', 
						 label = 'Exam grades vs final grades')
	plt.plot(range(0,12), 'r--')
	plt.axis([0, 11, 0, 11])
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()
