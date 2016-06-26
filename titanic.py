import numpy as np
from titanic_input import *
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

raw_train_data, y = read_train_file()
X = process_raw_data(raw_train_data)

sss = StratifiedShuffleSplit(y, 3, test_size=0.1, random_state=42)

for train_index, test_index in sss:
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]

	clf = RandomForestClassifier(n_estimators=100)
	clf.fit(X_train, y_train)

	y_predict = clf.predict(X_test)

	print metrics.accuracy_score(y_predict, y_test)
	print metrics.confusion_matrix(y_predict, y_test)