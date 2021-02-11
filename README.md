<!---
<div align="center">
    <img src="docs/tiresia.png" width=150 height=180>
</div>
--->

# tiresia

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

test_type = "regressor"

if test_type == "classifier":
    train, target = make_classification(5000, 20)
elif test_type == "regressor":
    train, target = make_regression(5000, 20)
    
x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.3)

autopred = AutoPredictor(estimator_type=test_type,
                         scoring=r2_score,
                         greater_is_better=True)
autopred.fit(x_train, y_train, x_test, y_test)

predictions = autopred.predictions
results     = autopred.results

print(results)

                        REGRESSOR r2_score
18                        LassoCV    1.000
12                 HuberRegressor    1.000
23                      LinearSVR    1.000
22               LinearRegression    1.000
21                    LassoLarsIC    1.000
20                    LassoLarsCV    1.000
29                RANSACRegressor    1.000
17                          Lasso    1.000
16                         LarsCV    1.000
15                           Lars    1.000
14                    KernelRidge    1.000
28     PassiveAggressiveRegressor    1.000
31                          Ridge    1.000
32                        RidgeCV    1.000
33                   SGDRegressor    1.000
35     TransformedTargetRegressor    1.000
2                   BayesianRidge    1.000
24                   MLPRegressor    0.998
6                    ElasticNetCV    0.992
11  HistGradientBoostingRegressor    0.957
10      GradientBoostingRegressor    0.925
5                      ElasticNet    0.881
8             ExtraTreesRegressor    0.853
27    OrthogonalMatchingPursuitCV    0.827
30          RandomForestRegressor    0.824
1                BaggingRegressor    0.794
0               AdaBoostRegressor    0.751
36               TweedieRegressor    0.743
13            KNeighborsRegressor    0.686
3           DecisionTreeRegressor    0.486
26      OrthogonalMatchingPursuit    0.387
7              ExtraTreeRegressor    0.386
34                            SVR    0.357
19                      LassoLars    0.353
25                          NuSVR    0.285
9        GaussianProcessRegressor    0.038
4                  DummyRegressor   -0.003
```
