import pandas as pd

from sklearn.utils import all_estimators
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer

import time

ignore_regressors = {"TheilSenRegressor",
                    "ARDRegression",
                    "CCA", 
                    "IsotonicRegression", 
                    "StackingRegressor",
                    "MultiOutputRegressor", 
                    "MultiTaskElasticNet", 
                    "MultiTaskElasticNetCV",  
                    "MultiTaskLasso",
                    "MultiTaskLassoCV",       
                    "PLSCanonical",            
                    "PLSRegression",           
                    "RadiusNeighborsRegressor",
                    "RegressorChain", 
                    "VotingRegressor", 
                    "_SigmoidCalibration", 
                    }
                
class AutoRegressor:
    def __init__(self,
                 models_to_test="all",
                 models_to_exclude=[],
                 scoring=mean_squared_error,
                 param_grid=dict(),
                 disable_warings=True,
                 random_state=42,
                 n_jobs=1):
        
        # Load the regressors
        if models_to_test == "all":
            excluded_regressors = set.union(ignore_regressors, set(models_to_exclude))
            self.regressors = {reg[0]: reg[1] for reg in all_estimators(type_filter="regressor")
                               if reg[0] not in excluded_regressors}
        else:
            self.regressors = {reg[0]: reg[1] for reg in all_estimators(type_filter="regressor")
                               if reg[0] in models_to_test}

        # Disable warnings
        if disable_warings:
            import warnings
            warnings.filterwarnings("ignore")

        # Scoring function
        self.scoring      = scoring
        self.scoring_name = scoring.__name__

        # Final results dataframe
        self.results = pd.DataFrame(columns={"REGRESSOR", self.scoring_name})

        self.param_grid   = param_grid 
        self.random_state = random_state
        self.n_jobs       = n_jobs

    def fit(self, X_train, Y_train, X_val, Y_val):
        total_time = 0
        for name, regressor in self.regressors.items():
            try:
                start = time.time()

                # Fit the current regressor and calculate the score
                model = GridSearchCV(estimator=regressor(),
                                     param_grid=dict(),
                                     scoring=make_scorer(self.scoring),
                                     n_jobs=self.n_jobs)
                model.fit(X_train, Y_train)
                preds = model.predict(X_val)
                score = self.scoring(Y_val, preds)

                end = time.time()

                time_elapsed = end - start
                total_time += time_elapsed

                print("- {}: \n"
                      " Time elapsed: {:.3f} s \n"
                      " {}: {:.3f} \n".format(name, time_elapsed, self.scoring_name, score))

                # Append results for the current regressor
                self.results = self.results.append({"REGRESSOR"      : name,
                                                    self.scoring_name: "{:.3f}".format(score)},
                                                  ignore_index=True)
            except Exception as e:
                print(f"- {name}: \n {e}")
            
        self.results.sort_values(by=[self.scoring_name], inplace=True)

    def get_results(self):
        return results
        
            
        
        
