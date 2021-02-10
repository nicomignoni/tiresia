to_ignore = {"regressor": {"TheilSenRegressor",
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
                           "_SigmoidCalibration"},
             "classifier": {"CheckingClassifier", 
                            "ClassifierChain",
                            "ComplementNB",
                            "HistGradientBoostingClassifier",
                            "MLPClassifier", 
                            "LogisticRegressionCV", 
                            "MultiOutputClassifier",
                            "MultinomialNB", 
                            "OneVsOneClassifier", 
                            "OneVsRestClassifier",
                            "OutputCodeClassifier", 
                            "RadiusNeighborsClassifier",
                            "VotingClassifier"}
            }

def print_progress(name, time_elapsed, scoring_name, score):
    print("- {}: \n"
          " Time elapsed: {:.3f} s \n"
          " {}: {:.3f} \n".format(name, time_elapsed, scoring_name, score))
