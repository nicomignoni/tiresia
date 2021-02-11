import pandas as pd

from sklearn.utils import all_estimators
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, make_scorer

from tiresia.utils import to_ignore, print_progress

import time

class AutoPredictor:
    def __init__(self,
                 estimator_type="regressor",
                 models_to_test="all",
                 models_to_exclude=[],
                 disable_warings=True,
                 random_state=42):

        self.estimator_type = estimator_type
        
        # Load the estimators
        if models_to_test == "all":
            excluded_estimators = set.union(to_ignore[self.estimator_type], set(models_to_exclude))
            self.estimators = {est[0]: est[1] for est in all_estimators(type_filter=self.estimator_type)
                               if est[0] not in excluded_estimators}
        else:
            self.estimators = {est[0]: est[1] for est in all_estimators(type_filter=self.estimator_type)
                               if est[0] in models_to_test}
        if disable_warings:
            import warnings
            warnings.filterwarnings("ignore")

        self.random_state = random_state

    def fit(self,
            X_train, Y_train, X_val, Y_val,
            param_grid=dict(),
            scoring=None,
            greater_is_better=False,
            n_jobs=1,
            verbose=1):
        
        total_time = 0

        # Predictions and best params
        self.predictions = dict()
        self.best_params = dict()

        # Scoring function
        if scoring:
            pass
        elif self.estimator_type == "regressor":
            scoring           = mean_squared_error
            greater_is_better = False
        elif self.estimator_type == "classifier":
            scoring           = accuracy_score
            greater_is_better = True

        # Final results dataframe
        self.results = pd.DataFrame(columns={self.estimator_type.upper(), scoring.__name__})
            
        for name, estimator in self.estimators.items():
            try:
                start = time.time()

                # Check if the estimator has a param_grid
                if name in param_grid:
                    estimator_grid = param_grid[name]
                else:
                    estimator_grid = dict()

                # Fit the current estimator and calculate the score
                model = GridSearchCV(estimator=estimator(),
                                     param_grid=estimator_grid,
                                     scoring=make_scorer(scoring, greater_is_better),
                                     n_jobs=n_jobs)
                model.fit(X_train, Y_train)
                preds = model.predict(X_val)
                score = scoring(Y_val, preds)

                end = time.time()

                time_elapsed = end - start
                total_time  += time_elapsed

                self.predictions[name] = preds
                self.best_params[name] = model.best_params_

                # Print the current estimator progresses
                if verbose > 0:
                    print_progress(name, time_elapsed, scoring.__name__, score)

                # Append results for the current regressor
                self.results = self.results.append({self.estimator_type.upper() : name,
                                                    scoring.__name__: "{:.3f}".format(score)},
                                                  ignore_index=True)
            except Exception as e:
                print(f"- {name}: \n {e} \n")

        # Sort the results, depending on the nature of the scoring function
        self.results.sort_values(by=[scoring.__name__],
                                 ascending=not greater_is_better,
                                 inplace=True)
        
        if verbose > 0:
            print("Total time elapsed: {:.3f}".format(total_time))

        
            
        
        
