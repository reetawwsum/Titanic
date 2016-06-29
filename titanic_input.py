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

def write_output(predictions, file_name):
	with open(file_path + file_name, 'wb') as f:
		csv_writer = csv.writer(f, delimiter=',')
		csv_writer.writerow(['PassengerId', 'Survived'])

		for i, prediction in enumerate(predictions):
			csv_writer.writerow([i+892, prediction])

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

	# Adding embarked to the feature list
	raw_data[raw_data[:, 9] == '', 9] = 'S'

	le = preprocessing.LabelEncoder()
	le.fit(raw_data[:, 9])

	data = np.append(data, np.reshape(le.transform(raw_data[:, 9]), (-1, 1)), 1)

	# Adding family size to the feature list
	data = np.append(data, np.reshape(data[:, 2] + data[:, 3], (-1, 1)), 1)

	# Adding title of the name to the feature list
	title_labels = ['Mrs', 'Mr', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms', 'Major', 'Col']
	names = raw_data[:, 1]
	titles = []

	for name in names:
		title_index = -1

		for i, title in enumerate(title_labels):
			if title in name:
				title_index = i
				break

		titles.append(title_index)

	titles = np.array(titles)

	data = np.append(data, np.reshape(titles, (-1, 1)), 1)

	# Adding age to the feature list
	ages = raw_data[:, 3]
	
	ages[ages == ''] = np.mean(ages[ages != ''].astype(float))

	data = np.append(data, np.reshape(ages, (-1, 1)).astype(float), 1)

	# Adding Numeric - 0 & Alphanumeric - 1 ticket number to the feature list
	ticket_numbers = raw_data[:, 6]
	tickets = []

	for ticket in ticket_numbers:
		if ticket.isdigit():
			tickets.append(0)
		else:
			tickets.append(1)

	data = np.append(data, np.reshape(tickets, (-1, 1)), 1)

	# Adding number of digits in numeric tickets to the feature list. Adding -1 in case of alphanumeric ticket
	digits = []

	for ticket in ticket_numbers:
		if ticket.isdigit():
			digits.append(len(ticket))
		else:
			digits.append(-1)

	data = np.append(data, np.reshape(digits, (-1, 1)), 1)

	# Adding ticket type in case of alphanumeric tickets to the feature list. Adding -1 in case of numeric ticket
	ticket_types = ['A/5', 'A./5', 'A.5', 'PC', 'STON', 'PP', 'SC', 'S.C', 'C.A', 'CA', 'SOTON', 'F.C.C', 'S.O.C', 'A/4', 'A4', 'SP', 'S.P', 'SO/C', 'W./C', 'W.E.P', 'C', 'S.O.P', 'Fa', 'LINE', 'A/S', 'WE/P', 'S.O', 'LP', 'AQ', 'A']
	types = []

	for ticket in ticket_numbers:
		if ticket.isdigit():
			types.append(-1)
			continue

		for i, ticket_type in enumerate(ticket_types):
			if ticket_type in ticket:
				ticket_type_index = i
				break

		types.append(ticket_type_index)

	types = np.array(types)

	data = np.append(data, np.reshape(types, (-1, 1)), 1)

	return data

if __name__ == '__main__':
	raw_train_data, target = read_train_file()
	raw_test_data = read_test_file()

	raw_data = np.append(raw_train_data, raw_test_data, axis=0)

	data = process_raw_data(raw_data)
