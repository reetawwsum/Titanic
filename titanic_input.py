from __future__ import division
import csv
import numpy as np
from sklearn import preprocessing

file_path = 'dataset/'
train_file = 'train.csv'
test_file = 'test.csv'

def read_train_file():
	raw_data = []
	target = []

	with open(file_path + train_file, 'rb') as f:
		csv_reader = csv.reader(f, delimiter=',')

		# Skipping first line
		csv_reader.next()

		for row in csv_reader:
			target.append(row[1])
			raw_data.append(row[2:])

		raw_data = np.array(raw_data)
		target = np.array(target).astype(int)

	return raw_data, target

def read_test_file():
	raw_data = []

	with open(file_path + test_file, 'rb') as f:
		csv_reader = csv.reader(f, delimiter=',')

		# Skipping first line
		csv_reader.next()

		for row in csv_reader:
			raw_data.append(row[1:])

		raw_data = np.array(raw_data)

	return raw_data

def process_raw_data(raw_data):
	data = []

	# Adding passenger class to the feature list
	data = np.reshape(raw_data[:, 0], (-1, 1)).astype(int)

	# Adding sex to the feature list
	le = preprocessing.LabelEncoder()
	le.fit(raw_data[:, 2])

	data = np.append(data, np.reshape(le.transform(raw_data[:, 2]), (-1, 1)), 1)

	# Adding Sibsp to the feature list
	data = np.append(data, np.reshape(raw_data[:, 4], (-1, 1)).astype(int), 1)

	# Adding Parch to the feature list
	data = np.append(data, np.reshape(raw_data[:, 5], (-1, 1)).astype(int), 1)

	# Adding fare to the feature list
	raw_data[((raw_data[:, 7] == '0') | (raw_data[:, 7] == '')) & (raw_data[:, 0] == '1'), 7] = '86.15'
	raw_data[((raw_data[:, 7] == '0') | (raw_data[:, 7] == '')) & (raw_data[:, 0] == '2'), 7] = '21.36'
	raw_data[((raw_data[:, 7] == '0') | (raw_data[:, 7] == '')) & (raw_data[:, 0] == '3'), 7] = '13.79'

	data = np.append(data, np.reshape(raw_data[:, 7], (-1, 1)).astype(float), 1)

	return data

raw_data, target = read_train_file()

data = process_raw_data(raw_data)

