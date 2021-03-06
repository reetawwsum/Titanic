import numpy as np
from titanic_input import *
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

raw_train_data, y_train = read_train_file()
raw_test_data = read_test_file()

raw_data = np.append(raw_train_data, raw_test_data, axis=0)

data = process_raw_data(raw_data)

X_train = data[:891]
X_test = data[891:]

clf = RandomForestClassifier(n_estimators=100)

def perform_cross_validation(X, y, clf):
	sss = StratifiedShuffleSplit(y, 3, test_size=0.1, random_state=42)

	for train_index, test_index in sss:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		clf.fit(X_train, y_train)

		y_predict = clf.predict(X_test)

		print metrics.accuracy_score(y_predict, y_test)
		print metrics.confusion_matrix(y_predict, y_test)

clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
write_output(y_predict, 'output.csv')
