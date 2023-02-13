#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:38:24 2023

@author: owen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:20:26 2023

@author: owen
"""

def classifier_selector(classifier_names, random_state = None):
    """
    Function to load regressors only when selected by the user. 
    This prevents the entire function from becoming unuseable when one or multiple of the regressors or not installed
    
    Parameters
    ----------
    regressor_names : list of str, regressor names with the following options:
        'dummy', 'lightgbm', 'xgboost', 'catboost', 'bayesianridge', 'lassolars', 'adaboost', 
        'gradientboost', 'histgradientboost', 'knn', 'sgd', 'bagging', 'svr', 'elasticnet'
    random_state : int, reproduceability state for regressors with randomization options
    
    Returns
    -------
    regressor_dict_none_rem : dict, regressor names as keys corresponding to tuple of regressor function and hyperparameter optimization function
    
    """

    methods = ['dummy', 'lightgbm', 
               # 'xgboost', 'catboost', 'bayesianridge', 'lassolars', 'adaboost', 'gradientboost', 'histgradientboost',
               #    'knn', 'sgd', 'bagging', 'svr', 'elasticnet'
                  ]
    selected_methods = list(set(classifier_names) & set(methods))
    if selected_methods == []:
        return print('no valid regressor names provided')
    
    # -- only load the regressor and import relevant package if provided regressor is in the list of pre-defined regressors
    method_dict = {}
    method_dict['dummy'] = dummy_loader() if 'dummy' in selected_methods else None
    method_dict['lightgbm'] = lightgbm_loader(random_state = random_state) if 'lightgbm' in selected_methods else None
    # method_dict['xgboost'] = xgboost_loader(random_state = random_state) if 'xgboost' in selected_methods else None
    # method_dict['catboost'] = catboost_loader(random_state = random_state) if 'catboost' in selected_methods else None
    # method_dict['bayesianridge'] = bayesianridge_loader() if 'bayesianridge' in selected_methods else None
    # method_dict['lassolars'] = lassolars_loader(random_state = random_state) if 'lassolars' in selected_methods else None
    # method_dict['adaboost'] = adaboost_loader(random_state = random_state) if 'adaboost' in selected_methods else None
    # method_dict['gradientboost'] = gradientboost_loader(random_state = random_state) if 'gradientboost' in selected_methods else None
    # method_dict['histgradientboost'] = histgradientboost_loader(random_state = random_state) if 'histgradientboost' in selected_methods else None
    # method_dict['knn'] = knn_loader() if 'knn' in selected_methods else None
    # method_dict['sgd'] = sgd_loader(random_state = random_state) if 'sgd' in selected_methods else None
    # method_dict['bagging'] = bagging_loader(random_state = random_state) if 'bagging' in selected_methods else None
    # method_dict['svr'] = svr_loader(random_state = random_state) if 'svr' in selected_methods else None
    # method_dict['elasticnet'] = elasticnet_loader(random_state = random_state) if 'elasticnet' in selected_methods else None
    
    # -- remove dictionary elements where regressor was not loaded
    method_dict_none_rem = {k: v for k, v in method_dict.items() if v is not None}
    
    return method_dict_none_rem
    

        

def dummy_loader():

    from sklearn.dummy import DummyClassifier
    def dummyHParams(trial):
        param_dict = {}
        param_dict['strategy'] = trial.suggest_categorical("strategy", ['most_frequent', 'stratified', 'uniform'])
        return param_dict
    
    return (DummyClassifier, dummyHParams)

def lightgbm_loader(random_state):
    
    from lightgbm import LGBMClassifier
    
    def lightgbmHParams(trial):
        param_dict = {}
        param_dict['objective'] = trial.suggest_categorical("objective", ['multiclass'])
        param_dict['max_depth'] = trial.suggest_int('max_depth', 3, 20)
        param_dict['n_estimators'] = trial.suggest_int('n_estimators', 50, 2000, log = True)
        param_dict['min_split_gain'] = trial.suggest_float("min_split_gain", 0, 15)  # boosts speed, decreases performance though
        param_dict['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-8, 10.0, log = True)
        param_dict['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-8, 10.0, log = True)
        param_dict['num_leaves'] = trial.suggest_int('num_leaves', 2, 256)
        param_dict['class_weight'] = trial.suggest_categorical('class_weight', [None, 'balanced'])
        param_dict['min_child_samples'] = trial.suggest_int('min_child_samples', 1, 100)
        param_dict['random_state'] = trial.suggest_categorical("random_state", [random_state])
        param_dict['verbosity'] = trial.suggest_categorical("verbosity", [-1])
        return param_dict
    
    return (LGBMClassifier, lightgbmHParams)

# def xgboost_loader(random_state):
    
#     from xgboost import XGBRegressor
    
#     def xgboostHParams(trial):
#         param_dict = {}
#         param_dict['booster'] = trial.suggest_categorical("booster", ['gbtree', 'gblinear', 'dart'])
#         param_dict['lambda'] = trial.suggest_float("lambda", 1e-8, 10.0, log = True)
#         param_dict['alpha'] = trial.suggest_float("alpha", 1e-8, 10.0, log = True)
#         param_dict['random_state'] = trial.suggest_categorical("random_state", [random_state])
#         param_dict['verbosity'] = trial.suggest_categorical("verbosity", [0])
        
#         if (param_dict['booster'] == 'gbtree') or (param_dict['booster'] == 'dart') :
            
#             param_dict['max_depth'] = trial.suggest_int("max_depth", 1, 14, log = False)  
            
#             # -- prevent the tree from exploding by limiting number of estimators and training size for larger depths
#             if (param_dict['max_depth'] >= 12) :
#                 max_n_estimators = 200; min_eta = 1e-2
#             elif (param_dict['max_depth'] >= 10) :
#                 max_n_estimators = 300; min_eta = 1e-3 # change to if >= 10, elif >= 8, else 
#             else :
#                 max_n_estimators = 400; min_eta = 1e-4
                
#             param_dict['n_estimators'] = trial.suggest_int("n_estimators", 20, max_n_estimators, log=False) 
#             param_dict['eta'] = trial.suggest_float("eta", min_eta, 1.0, log = True)   
#             param_dict['min_child_weight'] = trial.suggest_float("min_child_weight", 0, 10, log = False)
#             param_dict['gamma'] = trial.suggest_float("gamma", 0, 10, log = False)
#             param_dict['subsample'] = trial.suggest_float("subsample", 0.1, 1.0, log = False)
#             param_dict['colsample_bytree'] = trial.suggest_float("colsample_bytree", 0.1, 1.0, log = False)
#             param_dict['max_bin'] = trial.suggest_categorical("max_bin", [64, 128, 256, 512, 1024])    # performance boost when power of 2 (NOT -1)
        
#             if (param_dict['booster'] == 'dart') :
#                 param_dict['sample_type'] = trial.suggest_categorical("sample_type", ['uniform', 'weighted'])
#                 param_dict['normalize_type'] = trial.suggest_categorical("normalize_type", ['tree', 'forest'])
#                 param_dict['rate_drop'] = trial.suggest_float("rate_drop", 0., 1.0, log = False)
#                 param_dict['one_drop'] = trial.suggest_categorical("one_drop", [0, 1])
#         return param_dict
    
#     return (XGBRegressor, xgboostHParams)


# def catboost_loader(random_state):
    
#     from catboost import CatBoostRegressor
    
#     def catboostHParams(trial):
#         param_dict = {}
#         param_dict['depth'] = trial.suggest_int("depth", 1, 10, log = False) # maybe increase?
        
#         # -- prevent the tree from exploding by limiting number of estimators and training size for larger depths
#         if (param_dict['depth'] >= 8) :
#             max_iterations = 300; min_learning_rate = 1e-2
#         elif (param_dict['depth'] >= 6) :
#             max_iterations = 400; min_learning_rate = 5e-3
#         else :
#             max_iterations = 500; min_learning_rate = 1e-3
                
#         param_dict['iterations'] = trial.suggest_int("iterations", 20, max_iterations, log = True)
#         param_dict['learning_rate'] = trial.suggest_float("learning_rate", min_learning_rate, 1e0, log = True)  
#         param_dict['l2_leaf_reg'] = trial.suggest_float("l2_leaf_reg", 1e-2, 1e1, log = True)
#         param_dict['rsm'] = trial.suggest_float("rsm", 1e-2, 1e0, log = False)
#         param_dict['early_stopping_rounds'] =  trial.suggest_categorical("early_stopping_rounds", [5])
#         param_dict['logging_level'] = trial.suggest_categorical("logging_level", ['Silent'])
#         param_dict['random_seed'] = trial.suggest_categorical("random_seed", [random_state])
#         return param_dict
    
#     return (CatBoostRegressor, catboostHParams)

# def bayesianridge_loader():
    
#     from sklearn.linear_model import BayesianRidge
    
#     def bayesianRidgeHParams(trial):
#         param_dict = {}
#         param_dict['n_iter'] = trial.suggest_int("n_iter", 10, 400)
#         param_dict['tol'] = trial.suggest_float("tol", 1e-8, 1e2)
#         param_dict['alpha_1'] =  trial.suggest_float("alpha_1", 1e-8, 1e2, log = True)
#         param_dict['alpha_2'] = trial.suggest_float("alpha_2", 1e-8, 1e2, log = True)
#         param_dict['lambda_1'] = trial.suggest_float("lambda_1", 1e-8, 1e2, log = True)
#         param_dict['lambda_2'] = trial.suggest_float("lambda_2", 1e-8, 1e2, log = True)
#         return param_dict
    
#     return (BayesianRidge, bayesianRidgeHParams)


# def lassolars_loader(random_state):
    
#     from sklearn.linear_model import LassoLars
    
#     def lassoLarsHParams(trial):
#         param_dict = {}
#         param_dict['alpha'] = trial.suggest_float("alpha", 1e-8, 1e2, log = True)
#         param_dict['normalize'] = trial.suggest_categorical("normalize", [False])
#         param_dict['random_state'] =  trial.suggest_categorical("random_state", [random_state])
#         return param_dict
    
#     return (LassoLars, lassoLarsHParams)
    

# def adaboost_loader(random_state):
    
#     from sklearn.ensemble import AdaBoostRegressor
    
#     def adaBoostHParams(trial):
#         param_dict = {}
#         param_dict['n_estimators'] =trial.suggest_int("n_estimators", 10, 200)
#         param_dict['learning_rate'] = trial.suggest_float("learning_rate", 1e-2, 1e0, log = True)
#         param_dict['loss'] = trial.suggest_categorical("loss", ['linear', 'square', 'exponential'])
#         param_dict['random_state'] =  trial.suggest_categorical("random_state", [random_state])
#         return param_dict
    
#     return (AdaBoostRegressor, adaBoostHParams)


# def gradientboost_loader(random_state):
    
#     from sklearn.ensemble import GradientBoostingRegressor
    
#     def gradBoostHParams(trial):
#         param_dict = {}
#         param_dict['n_estimators'] =trial.suggest_int("n_estimators", 10, 300)
#         param_dict['learning_rate'] = trial.suggest_float("learning_rate", 1e-2, 1e0, log = True)
#         param_dict['subsample'] = trial.suggest_float("subsample", 1e-2, 1.0, log = False)
#         param_dict['max_depth'] = trial.suggest_int("max_depth", 1, 10)
#         param_dict['criterion'] = trial.suggest_categorical("criterion", ['friedman_mse', 'squared_error'])
#         param_dict['loss'] = trial.suggest_categorical("loss", ['squared_error', 'absolute_error', 'huber', 'quantile'])
#         param_dict['n_iter_no_change'] = trial.suggest_categorical("n_iter_no_change", [20])
#         param_dict['random_state'] =  trial.suggest_categorical("random_state", [random_state])
#         return param_dict
    
#     return (GradientBoostingRegressor, gradBoostHParams)


# def histgradientboost_loader(random_state):
    
#     from sklearn.ensemble import HistGradientBoostingRegressor
    
#     def histGradBoostHParams(trial):
#         param_dict = {}
#         param_dict['loss'] = trial.suggest_categorical("loss", ['squared_error', 'absolute_error'])
#         param_dict['max_depth'] = trial.suggest_int("max_depth", 1, 20, log = False)
#         param_dict['max_iter'] = trial.suggest_int("max_iter", 10, 500, log = True)
#         param_dict['max_leaf_nodes'] = trial.suggest_int("max_leaf_nodes", 2, 100)
#         param_dict['min_samples_leaf'] = trial.suggest_int("min_samples_leaf", 2, 200)
#         param_dict['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 1.0, log = True)
#         param_dict['n_iter_no_change'] = trial.suggest_categorical("n_iter_no_change", [20]) 
#         param_dict['random_state'] = trial.suggest_categorical("random_state", [random_state])
#         return param_dict
    
#     return (HistGradientBoostingRegressor, histGradBoostHParams)


# def knn_loader():
    
#     from sklearn.neighbors import KNeighborsRegressor
    
#     def KNearNeighboursHParams(trial):
#         param_dict = {}
#         param_dict['n_neighbors'] = trial.suggest_int("n_neighbors", 1, 101, step = 5)
#         param_dict['weights'] = trial.suggest_categorical("weights", ['uniform', 'distance'])
#         return param_dict
    
#     return (KNeighborsRegressor, KNearNeighboursHParams)


# def sgd_loader(random_state):
    
#     from sklearn.neighbors import SGDRegressor
    
#     def sgdHParams(trial):
#         param_dict = {}
#         param_dict['loss'] =  trial.suggest_categorical("loss", ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'])
#         param_dict['penalty'] = trial.suggest_categorical("penalty", ['l2', 'l1', 'elasticnet'])
#         param_dict['alpha'] = trial.suggest_float("alpha", 1e-8, 1e2, log = True)
#         param_dict['random_state'] =  trial.suggest_categorical("random_state", [random_state])
#         return param_dict
    
#     return (SGDRegressor, sgdHParams)


# def bagging_loader(random_state):
    
#     from sklearn.ensemble import BaggingRegressor
    
#     def baggingHParams(trial):
#         param_dict = {}
#         param_dict['n_estimators'] =  trial.suggest_int("n_estimators", 1, 101, step = 5)
#         param_dict['max_features'] = trial.suggest_float("max_features", 1e-1, 1.0, step = 0.1)
#         param_dict['random_state'] =  trial.suggest_categorical("random_state", [random_state])
#         return param_dict
    
#     return (BaggingRegressor, baggingHParams)


# def svr_loader(random_state):
    
#     from sklearn.svm import LinearSVR
    
#     def svrHParams(trial):
#         param_dict = {}
#         param_dict['loss'] = trial.suggest_categorical("loss", ['epsilon_insensitive', 'squared_epsilon_insensitive'])
#         param_dict['C'] = trial.suggest_float("C", 1e-5, 1e2, log = True)
#         param_dict['tol'] = trial.suggest_float("tol", 1e-8, 1e2, log = True)
#         param_dict['random_state'] = trial.suggest_categorical("random_state", [random_state])
#         return param_dict
    
#     return (LinearSVR, svrHParams)
    
    
# def elasticnet_loader(random_state):
    
#     from sklearn.linear_model import ElasticNet
    
#     def elasticnetHParams(trial):
#         param_dict = {}
#         param_dict['alpha'] = trial.suggest_float("alpha", 1e-8, 1e2, log = True)
#         param_dict['l1_ratio'] = trial.suggest_float("l1_ratio", 1e-5, 1.0, log = True)
#         param_dict['random_state'] = trial.suggest_categorical("random_state", [random_state])
#         return param_dict
    
#     return (ElasticNet, elasticnetHParams)
    
