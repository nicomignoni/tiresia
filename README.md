<!---
<div align="center">
    <img src="docs/tiresia.png" width=150 height=180>
</div>
--->

# tiresia

[![PyPI version shields.io](https://img.shields.io/pypi/v/tiresia.svg)](https://pypi.python.org/pypi/tiresia/)

## Installation
```
pip install tiresia
```

## Description
[tiresia](https://en.wikipedia.org/wiki/Tiresias) is just a wrapper around [scikit-learn](https://scikit-learn.org/stable/) ```GridSearchCV```. The idea is to simplify the model testing workflow. With tiresia you can choose which models to test and to exclude and provide a ```param_grid``` for the ones you want to explore deeper, while keeping the parameters of the less interesting ones on default. 

## Example
```python
from tiresia.predictor import AutoPredictor

from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score

test_type = "classifier"

if test_type == "classifier":
    train, target = make_classification(1000, 20)
elif test_type == "regressor":
    train, target = make_regression(1000, 20)
    
x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.3)

autopred = AutoPredictor(estimator_type=test_type)
        
autopred.fit(x_train, y_train, x_test, y_test, scoring=roc_auc_score, greater_is_better=True)

predictions = autopred.predictions
results     = autopred.results

print(results)

                       CLASSIFIER roc_auc_score
4          DecisionTreeClassifier         0.990
10     GradientBoostingClassifier         0.990
22         RandomForestClassifier         0.973
0              AdaBoostClassifier         0.963
1               BaggingClassifier         0.960
7            ExtraTreesClassifier         0.960
24              RidgeClassifierCV         0.936
23                RidgeClassifier         0.936
18                          NuSVC         0.936
14     LinearDiscriminantAnalysis         0.936
16             LogisticRegression         0.923
26                            SVC         0.923
3          CalibratedClassifierCV         0.920
15                      LinearSVC         0.917
17                NearestCentroid         0.917
25                  SGDClassifier         0.916
6             ExtraTreeClassifier         0.907
19    PassiveAggressiveClassifier         0.907
20                     Perceptron         0.906
11           KNeighborsClassifier         0.893
8                      GaussianNB         0.884
9       GaussianProcessClassifier         0.870
2                     BernoulliNB         0.840
12               LabelPropagation         0.827
13                 LabelSpreading         0.827
21  QuadraticDiscriminantAnalysis         0.615
5                 DummyClassifier         0.473

```
