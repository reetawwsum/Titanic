import csv
import numpy as np
from sklearn import preprocessing

file_path = 'dataset/'
train_file = 'train.csv'
test_file = 'test.csv'

data = []
target = []
raw_data = []

with open(file_path + train_file, 'rb') as f:
	csv_reader = csv.reader(f, delimiter=',')

	# Skipping first line
	csv_reader.next()

	for row in csv_reader:
		target.append(row[1])
		raw_data.append(row[2:])

	raw_data = np.array(raw_data)

# Adding passenger class to the feature list
data = np.reshape(raw_data[:, 0], (-1, 1)).astype(int)

le = preprocessing.LabelEncoder()
le.fit(raw_data[:, 2])

# Adding sex to the feature list
data = np.append(data, np.reshape(le.transform(raw_data[:, 2]), (-1, 1)), 1)

