to_ignore = {"regressor": {"TheilSenRegressor",
                           "ARDRegression",
                           "CCA",
                           "GammaRegressor",
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
                            "StackingClassifier",
                            "VotingClassifier"}
            }

def print_progress(name, time_elapsed, scoring_name, score):
    print("- {}: \n"
          " Time elapsed: {:.2f} s \n"
          " {}: {:.3f} \n".format(name, time_elapsed, scoring_name, score))
