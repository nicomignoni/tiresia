import pandas as pd

from sklearn.utils import all_estimators
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, make_scorer

from utils import to_ignore, print_progress

import time

class AutoPredictor:
    def __init__(self,
                 estimator_type="regressor",
                 models_to_test="all",
                 models_to_exclude=[],
                 scoring=None,
                 greater_is_better=False,
                 param_grid=dict(),
                 disable_warings=True,
                 random_state=42,
                 verbose=True,
                 n_jobs=1):

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

        # Scoring function
        if scoring:
            self.scoring           = scoring
            self.greater_is_better = greater_is_better
        elif estimator_type == "regressor":
            self.scoring           = mean_squared_error
            self.greater_is_better = False
        elif estimator_type == "classifier":
            self.scoring           = accuracy_score
            self.greater_is_better = True

        # Final results dataframe
        self.results = pd.DataFrame(columns={self.estimator_type.upper(), self.scoring.__name__})

        self.param_grid   = param_grid 
        self.random_state = random_state
        self.n_jobs       = n_jobs
        self.verbose      = verbose

    def fit(self, X_train, Y_train, X_val, Y_val):
        total_time = 0
        for name, estimator in self.estimators.items():
            try:
                start = time.time()

                # Fit the current regressor and calculate the score
                model = GridSearchCV(estimator=estimator(),
                                     param_grid=self.param_grid,
                                     scoring=make_scorer(self.scoring,
                                                         self.greater_is_better),
                                     n_jobs=self.n_jobs)
                model.fit(X_train, Y_train)
                preds = model.predict(X_val)
                score = self.scoring(Y_val, preds)

                end = time.time()

                time_elapsed = end - start
                total_time  += time_elapsed

                # Print the current estimator progresses
                if self.verbose:
                    print_progress(name, time_elapsed, self.scoring.__name__, score)

                # Append results for the current regressor
                self.results = self.results.append({self.estimator_type.upper() : name,
                                                    self.scoring.__name__: "{:.3f}".format(score)},
                                                  ignore_index=True)
            except Exception as e:
                print(f"- {name}: \n {e} \n")

        # Sort the results, depending on the nature of the scoring function
        self.results.sort_values(by=[self.scoring.__name__],
                                 ascending=not self.greater_is_better,
                                 inplace=True)
        
        if self.verbose:
            print("Total time elapsed: {:.3f}".format(total_time))

    def get_results(self):
        return results
        
            
        
        
