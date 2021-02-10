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
```
