import os

# Models
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import SGDRegressor
from sklearn.svm          import SVR
from sklearn.ensemble     import AdaBoostRegressor
from sklearn.ensemble     import RandomForestRegressor
from sklearn.ensemble     import GradientBoostingRegressor

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble     import HistGradientBoostingRegressor

# Metrics
from sklearn.metrics import *

# Utils
from sklearn.model_selection import GridSearchCV
from joblib                  import dump
from datetime                import datetime


class AutoRegressor:
    def __init__(self, models_to_test='all',
                 models_to_exclude=list(),
                 score_function=mean_squared_error,
                 greater_is_better=False,
                 n_jobs=None,
                 verbose=0,
                 log_path=f"tiresias-{datetime.now().strftime('%m-%d-%Y_(%H-%M-%S)')}",
                 disable_warnings=False):
        
        self.estimators = {
              'Ridge':                         {'func'  : Ridge(),
                                                'params': {'alpha': [0.1, 0.5, 1, 1.5, 2]}},
              'Lasso':                         {'func'  : Lasso(),
                                                'params': {'alpha': [0.1, 0.5, 1, 1.5, 2]}},
              'ElasticNet':                    {'func'  : ElasticNet(),
                                                'params': {'alpha'   : [0.1, 0.5, 1, 1.5, 2],
                                                           'l1_ratio': [0, 0.5, 1]}},
              'LassoLars':                     {'func'  : LassoLars(),
                                                'params': {'alpha': [0.1, 0.5, 1, 1.5, 2]}},
              'OrthogonalMatchingPursuit':     {'func'  : OrthogonalMatchingPursuit(),
                                                'params': {'n_nonzero_coefs': [None]}},
              'BayesianRidge':                 {'func'  : BayesianRidge(),
                                                'params': {'n_iter'  : [500],
                                                           'alpha_1' : [1e-8, 1e-6, 1e-4, 1e-2],
                                                           'alpha_2' : [1e-8, 1e-6, 1e-4, 1e-2],
                                                           'lambda_1': [1e-8, 1e-6, 1e-4, 1e-2],
                                                           'lambda_2': [1e-8, 1e-6, 1e-4, 1e-2]}},
              'SGDRegressor':                  {'func'  : SGDRegressor(),
                                                'params': {'penalty' : ['elasticnet'],
                                                           'alpha'   : [1e-4, 1e-3, 1e-2],
                                                           'l1_ratio': [0, 0.2, 0.5, 0.8, 1],
                                                           'learning_rate': ['optimal']}},
              'SVR':                           {'func'  : SVR(),
                                                'params': {'C'     : [0.1, 0.5, 1, 1.5, 2],
                                                           'kernel': ['linear', 
                                                                      'poly', 
                                                                      'rbf', 
                                                                      'sigmoid'],
                                                           'degree': [2, 3, 4, 5],
                                                           'coef0' : [0, 0.5, 1]}},
              'AdaBoostRegressor':             {'func'  : AdaBoostRegressor(),
                                                'params': {'n_estimators' : [50, 60 ,70],
                                                           'learning_rate': [1, 1.1, 1.2, 1.3]}},
              'RandomForestRegressor':         {'func'  : RandomForestRegressor(),
                                                'params': {'n_estimators'     : [100, 200, 300],
                                                           'min_samples_split': [2, 3, 4],
                                                           'min_samples_leaf' : [1, 2, 3]}},
              'GradientBoostingRegressor':     {'func'  : GradientBoostingRegressor(),
                                                'params': {'learning_rate'    : [0.005, 0.01, 0.02],
                                                           'n_estimators'     : [100, 150, 200],
                                                           'min_samples_split': [1.0, 2, 3, 4],
                                                           'max_depth'        : [3, 4, 5]}},
              'HistGradientBoostingRegressor': {'func'  : HistGradientBoostingRegressor(),
                                                'params': {'learning_rate'   : [0.005, 0.01, 0.02],
                                                           'min_samples_leaf': [1, 2, 3, 4],
                                                           'max_depth'       : [3, 4, 5],
                                                           'max_iter'        : [100, 150]}}
             }

    
        if models_to_test == 'all':
            self.models_to_test = self.estimators.keys()
        else:
            self.models_to_test = [model for model in models_to_test if model not in models_to_exclude]

        if disable_warnings:
            import warnings
            warnings.filterwarnings("ignore")

        self.score_function    = score_function
        self.greater_is_better = greater_is_better
        self.n_jobs            = n_jobs
        self.verbose           = verbose
        self.log_path          = log_path
        os.mkdir(self.log_path)

    def set_params(self, estimator, params):
        for param, values in params.items():
            self.estimators[estimator]["params"] = values

    def get_params(self, estimator):
        return self.estimators[estimator]["params"]

    def get_estimators(self):
        return {estimator for estimator in self.estimators.keys()}

    def fit(self, X_train, Y_train, X_test, Y_test):
        for estimator in self.models_to_test:
            autoreg = GridSearchCV(estimator=self.estimators[estimator]["func"],
                                   param_grid=self.estimators[estimator]["params"],
                                   scoring=make_scorer(score_func=self.score_function,
                                                       greater_is_better=self.greater_is_better),
                                   verbose=self.verbose,
                                   n_jobs=self.n_jobs)
            autoreg.fit(X_train, Y_train)
            predictions = autoreg.predict(X_test)
            self.estimators[estimator]["score"] = self.score_function(Y_test, predictions)
            dump(autoreg, self.log_path + f"/{estimator}.joblib")
            
    def results(self):
        return {estimator: self.estimators[estimator]["score"] for estimator in self.models_to_test}
            
            
            
        
        
    
