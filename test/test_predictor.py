'''Mock test. TODO: score type check & estimator presence''' 
import unittest

from tiresia.predictor import AutoPredictor

from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

class TestPredictor(unittest.TestCase):
	
	def test_regressor(self):
		reg_train, reg_target = make_regression(1000, 20)
		reg_x_train, reg_x_test, reg_y_train, reg_y_test = train_test_split(reg_train, reg_target, test_size=0.3)
		reg_autopred = AutoPredictor(estimator_type="regressor")
		reg_autopred.fit(reg_x_train, reg_y_train, reg_x_test, reg_y_test)

		print(reg_autopred.results)
		self.assertIsNotNone(reg_autopred.results)

	def test_classifier(self):
		clf_train, clf_target = make_classification(1000, 20)
		clf_x_train, clf_x_test, clf_y_train, clf_y_test = train_test_split(clf_train, clf_target, test_size=0.3)

		clf_autopred = AutoPredictor(estimator_type="classifier")
		clf_autopred.fit(clf_x_train, clf_y_train, clf_x_test, clf_y_test)

		print(clf_autopred.results)
		self.assertIsNotNone(clf_autopred.results)

if __name__ == "__main__":
	unittest.main()

