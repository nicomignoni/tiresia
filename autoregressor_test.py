from regression import AutoRegressor

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

train, target = make_regression(10000, 20)
x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.3)

autoreg = AutoRegressor()
autoreg.fit(x_train, y_train, x_test, y_test)

