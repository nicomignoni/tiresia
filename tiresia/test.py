from predictor import AutoPredictor

from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

test_type = "classifier"

if test_type == "classifier":
    train, target = make_classification(10000, 20)
elif test_type == "regressor":
    train, target = make_regression(5000, 20)
    
x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.3)

autopred = AutoPredictor(estimator_type=test_type,
                         scoring=roc_auc_score,
                         greater_is_better=True)
autopred.fit(x_train, y_train, x_test, y_test)

