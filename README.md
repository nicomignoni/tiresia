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
```
The last line will print the DataFrame containing the results for each tested models.

| REGRESSOR                     | r2_score |
|-------------------------------|----------|
| LassoCV                       | 1.000    |
| HuberRegressor                | 1.000    |
| LinearSVR                     | 1.000    |
| LinearRegression              | 1.000    |
| LassoLarsIC                   | 1.000    |
| LassoLarsCV                   | 1.000    |
| RANSACRegressor               | 1.000    |
| Lasso                         | 1.000    |
| LarsCV                        | 1.000    |
| Lars                          | 1.000    |
| KernelRidge                   | 1.000    |
| PassiveAggressiveRegressor    | 1.000    |
| Ridge                         | 1.000    |
| RidgeCV                       | 1.000    |
| SGDRegressor                  | 1.000    |
| TransformedTargetRegressor    | 1.000    |
| BayesianRidge                 | 1.000    |
| MLPRegressor                  | 0.998    |
| ElasticNetCV                  | 0.992    |
| HistGradientBoostingRegressor | 0.957    |
| GradientBoostingRegressor     | 0.925    |
| ElasticNet                    | 0.881    |
| ExtraTreesRegressor           | 0.853    |
| OrthogonalMatchingPursuitCV   | 0.827    |
| RandomForestRegressor         | 0.824    |
| BaggingRegressor              | 0.794    |
| AdaBoostRegressor             | 0.751    |
| TweedieRegressor              | 0.743    |
| KNeighborsRegressor           | 0.686    |
| DecisionTreeRegressor         | 0.486    |
| OrthogonalMatchingPursuit     | 0.387    |
| ExtraTreeRegressor            | 0.386    |
| SVR                           | 0.357    |
| LassoLars                     | 0.353    |
| NuSVR                         | 0.285    |
| GaussianProcessRegressor      | 0.038    |
| DummyRegressor                | -0.003   |
