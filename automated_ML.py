#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 16:41:02 2022

@author: owen
"""

#%% Packages

import numpy as np
import pandas as pd 
pd.options.mode.chained_assignment = None 
import os
import joblib
import optuna

from optuna.samplers import TPESampler #, RandomSampler
from sklearn.model_selection import KFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import median_absolute_error, r2_score, make_scorer
from sklearn.pipeline import Pipeline

# -- following imports are only required if selected in list_regressor_hyper, else they can be commented out here ...
# ... and removed the 'methodSelector' functions and `regressors`, `regressors_id` and `regressors_dict` lines
from sklearn.linear_model import LassoLars, Ridge, ElasticNet, BayesianRidge, SGDRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, BaggingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.svm import LinearSVR

# -- requires custom downloads
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

#%%

def automated_regression(y, X, test_frac = 0.2, timeout = 600, n_trial = 100, 
                         cross_validation = None, sampler = None, pruner = None, 
                         poly_value = None, spline_value = None, pca_value = None, 
                         metric_optimise = median_absolute_error, metric_assess = [median_absolute_error, r2_score], optimisation_direction = 'maximize', 
                         write_folder = os.getcwd() + '/auto_regression/', overwrite = False, 
                         list_regressors_hyper = ['lightgbm', 'xgboost', 'catboost', 'bayesianridge', 'lassolars'], list_regressors_training = None, random_state = 42):
    
    """
    ------------------------------------
    Summary:
        1. Takes two dataframes containing validation data (y) and input variables (X).
        2. Splits this data into testing and training subsets 
        3. Performs optuna hyperparameter optimisation per specified regressors on training data
        4. Trains individual regressors using optimized hyperparameters and creates 'stacked' regressor
        5. Assess performance of final stacked regressor
        
    ------------------------------------
    Input:
        y: dataframe, dependent data of size n x 1 with n obervations for valdiation
        X: dataframe, independent data of size n x M with n observations for M parameters
        test_frac: float, fraction of data to use for testing subset
        timeout: int or float, duration in seconds after which hypertuning is terminated per regressor (unless n_trials is reached beforehand). 
        n_trial: int, number of trials to run per regressor. Each trial tests performance using a sampled combination of hyperparameter values (unless timeout is reached beforehand)
        cross_validation: cross valdiator function, sklearn cross_valdiator used in determining cross validated performance of regressors. E.g. sklearn.model_selection.KFold()
        sampler: optuna sampler, used to sample parameter values to test from the multidimensional hyperparameter space. E.g. optuna.samplers.TPESampler(seed = 42), or optuna.samplers.RandomSampler(seed = 42)    
        pruner: optuna pruner, used to prune inauspacious trials. E.g. optuna.pruners.HyperbandPruner(), optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
        poly_value: dict or int/float, optional kwargs for to be fed to poly_chooser-function. E.g. {'degree': 2, 'interaction_only'= False} or 2
        spline_value: dict or tuple/list or int, optional kwargs for to be fed to spline_chooser-function. E.g. {'n_knots': 5, 'degree':3} or (5, 3) or [5, 3]
        pca_value: dict for multiple params or int/str/float for single params, optional kwargs to be fed to pca_chooser-function. E.g. {'n_components': 0.95, 'whiten'=False}
        metric_optimise: function, metric on which to optimise hyperparameters. Takes y_true and y_pred. E.g. sklearn's median_absolute_error, mean_absolute_error, r2_score etc
                         for custom function use lambda e.g. --> metric_optimise = lambda true, prediction : mean_pinball_loss(true, prediction, alpha=0.1)
        metric_assess: list of function(s), metrics with which to assess performance of individual and stacked regressors
        optimisation_direction: str, whether to 'minimize' or 'maximize' evaluation metric
        write_folder: str, name of the folder to be created where to store individually trained regressors and stacked regressor. Directory will be created if does not exist
        overwrite: Boolean, whether to overwrite if files in write_folder already exist 
        list_regressors_hyper: list of str, containing names of regressors for which to apply hyperparameter optimisation. 
            Options are : 'lightgbm', 'xgboost', 'catboost', 'bayesianridge', 'lassolars', 'adaboost', 'gradientboost','knn', 'sgd', 'bagging', 'svr', 'elasticnet'
        list_regressors_training: list of parameters to train and assess on performance. Listed regressors must be present in write_folder
        random_state: int, set random state for reproducibility
        
    ------------------------------------
    Output:
        metric_performance_summary_dict: dict, contains mean and std. performance per metric per regressor in list_regressors_training
        idexes_test_kfold: list of tuples of numpary arrays, contains fold training and test indexes for test dataset
        test_index: array, indexes of test fraction
        train_index: array, indexes of training fraction
        y_pred: array, contains estimates of y predicted on the test dataset
        y_test: array, contains validation values of y
    ------------------------------------
    Example:
        
        import sklearn
        import pandas as pd
        import sklearn
        from sklearn.datasets import fetch_california_housing

        dataset = fetch_california_housing()
        X_full, y_full = dataset.data, dataset.target
        y = pd.DataFrame(y_full)
        X = pd.DataFrame(X_full)

        metric_performance_summary_dict, idexes_test_kfold, test_index, train_index, y_pred, y_test = automated_regression(
            y = y, X = X, test_frac = 0.2, timeout = 600, n_trial = 100, 
            metric_optimise = sklearn.metrics.mean_pinball_loss,  metric_assess = [sklearn.metrics.mean_pinball_loss, sklearn.metrics.mean_squared_error, sklearn.metrics.r2_score],
            optimisation_direction = 'minimize',  overwrite = True, 
            list_regressors_hyper = ['lightgbm', 'bayesianridge'], list_regressors_training = None, 
            random_state = 42)
        
    ------------------------------------
    Note: 
        1. Parameters poly_value, spline_value and pca_value are optional. Even when they are submitted, optimization might favour excluding them. 
           Poly_value and spline_value generally improve linear models but add little to boosted models. pca_value serves as compression.
           When all three are included they are performed on the data in the following order: poly_value --> spline_value --> (optional scaler) --> pca_value 
        2. cross_validation, sampler and pruner are set to the standard versions when None is supplied
    
    """
    # -- split data into traning and testing datasets
    X_train, X_test, y_train, y_test, test_index, train_index = splitTrainTest(X, y, testSize = test_frac, randomState = random_state, continuous = True)
    
    # -- use standard optuna sampler, cross validation, pruner, metric etc, if none are specified
    # -- For the following this is done so that a specified random state is incorporated in the variables
    sampler = TPESampler(seed = random_state) if 'optuna.samplers' not in type(sampler).__module__ else sampler
    cross_validation = KFold(n_splits = 5, shuffle = True, random_state= random_state) if 'split' not in dir(cross_validation) else cross_validation
    # -- for the following it is done to keep the function input uncluttered and to ensure a specific type of input
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3) if 'optuna.pruners' not in type(pruner).__module__ else pruner
    metric_optimise = median_absolute_error if callable(metric_optimise) == False else metric_optimise
    list_regressors_training = list_regressors_hyper if list_regressors_training == None else list_regressors_training
    
    # -- select method(s) of regression to perform
    methods = methodSelector(metric = metric_optimise, random_state = random_state)
    
    # -- create storage location
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    ####################################################
    # -- perform optuna optimisation per regressors -- # 
    ####################################################
    
    regressor_optimise(methods, optimisation_direction, list_regressors_hyper, X_train, y_train, sampler, pruner, metric_optimise, cross_validation, 
                       poly_value, spline_value, pca_value, n_trial,timeout, write_folder, overwrite, random_state)
      
    ########################################################
    # -- fit regressors using optimised hyperparameters -- #
    ########################################################
    
    estimators = regressor_fit(list_regressors_training, write_folder)
    
    ########################################################################
    # -- train regressors (including stacked) and calculate performance -- #
    ########################################################################
    
    performance_dict, idexes_test_kfold, y_pred = regression_assess(X_train, X_test, y_train, y_test, list_regressors_training, estimators, metric_assess, 
                                                                    cross_validation, write_folder, overwrite, random_state)
        
    return performance_dict, idexes_test_kfold, test_index, train_index, y_pred, y_test
        
# regressor selection

# -- create dictionary whose keys call respective regressors
regressors= [DummyRegressor, LGBMRegressor, XGBRegressor, CatBoostRegressor, BayesianRidge, LassoLars, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
             KNeighborsRegressor, SGDRegressor, BaggingRegressor, LinearSVR, ElasticNet]
regressor_id= ['dummy', 'lightgbm', 'xgboost', 'catboost', 'bayesianridge', 'lassolars', 'adaboost', 'gradientboost', 'histgradientboost', 'knn', 'sgd', 'bagging', 'svr', 'elasticnet']
regressor_dict = dict(zip(regressor_id, regressors))

def methodSelector(metric, random_state):

    def dummyHParams(trial):
        param_dict = {}
        param_dict['strategy'] = trial.suggest_categorical("strategy", ['mean', 'median', 'quantile'])
        if param_dict['strategy'] == 'quantile':
            param_dict['quantile'] = trial.suggest_float("quantile", 0., 1., step = 0.01)
        return param_dict
    
    def lightgbmHParams(trial):
        param_dict = {}
        param_dict['objective'] = trial.suggest_categorical("objective", ['regression'])
        param_dict['max_depth'] = trial.suggest_int('max_depth', 3, 20)
        param_dict['n_estimators'] = trial.suggest_int('n_estimators', 50, 2000, log = True)
        param_dict['max_bin'] = trial.suggest_categorical("max_bin", [63, 127, 255, 511, 1023])   # performance boost when power of 2 (-1)
        param_dict['min_gain_to_split'] = trial.suggest_float("min_gain_to_split", 0, 15)  # boosts speed, decreases performance though
        param_dict['lambda_l1'] = trial.suggest_float('lambda_l1', 1e-8, 10.0, log = True)
        param_dict['lambda_l2'] = trial.suggest_float('lambda_l2', 1e-8, 10.0, log = True)
        param_dict['num_leaves'] = trial.suggest_int('num_leaves', 2, 256)
        param_dict['feature_fraction'] = trial.suggest_float('feature_fraction', 0.1, 1.0)
        param_dict['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.1, 1.0)
        param_dict['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 7)
        param_dict['min_child_samples'] = trial.suggest_int('min_child_samples', 1, 100)
        param_dict['random_state'] = trial.suggest_categorical("random_state", [random_state])
        param_dict['verbosity'] = trial.suggest_categorical("verbosity", [0])
        return param_dict
    
    def xgboostHParams(trial):
        param_dict = {}
        param_dict['booster'] = trial.suggest_categorical("booster", ['gbtree', 'gblinear', 'dart'])
        param_dict['lambda'] = trial.suggest_float("lambda", 1e-8, 10.0, log = True)
        param_dict['alpha'] = trial.suggest_float("alpha", 1e-8, 10.0, log = True)
        param_dict['random_state'] = trial.suggest_categorical("random_state", [random_state])
        param_dict['verbosity'] = 0
        
        if (param_dict['booster'] == 'gbtree') or (param_dict['booster'] == 'dart') :
            param_dict['max_depth'] = trial.suggest_int("max_depth", 1, 20, log = False)
            param_dict['n_estimators'] = trial.suggest_int("n_estimators", 20, 400, log=False)
            param_dict['eta'] = trial.suggest_float("eta", 1e-4, 1.0, log = True)
            param_dict['min_child_weight'] = trial.suggest_float("min_child_weight", 0, 10, log = False)
            param_dict['gamma'] = trial.suggest_float("gamma", 0, 10, log = False)
            param_dict['subsample'] = trial.suggest_float("subsample", 0.1, 1.0, log = False)
            param_dict['colsample_bytree'] = trial.suggest_float("colsample_bytree", 0.1, 1.0, log = False)
            param_dict['max_bin'] = trial.suggest_categorical("max_bin", [64, 128, 256, 512, 1024])    # performance boost when power of 2 (NOT -1)
            
            if (param_dict['booster'] == 'dart') :
                param_dict['sample_type'] = trial.suggest_categorical("sample_type", ['uniform', 'weighted'])
                param_dict['normalize_type'] = trial.suggest_categorical("normalize_type", ['tree', 'forest'])
                param_dict['rate_drop'] = trial.suggest_float("rate_drop", 0., 1.0, log = False)
                param_dict['one_drop'] = trial.suggest_categorical("one_drop", [0, 1])
                
        elif (param_dict['booster'] == 'gblinear') :
            param_dict['updater'] = trial.suggest_categorical("updater", ['shotgun', 'coord_descent'])
            param_dict['feature_selector'] = trial.suggest_categorical("feature_selector", ['cyclic', 'shuffle'])
        return param_dict
    
    def catboostHParams(trial):
        param_dict = {}
        param_dict['depth'] = trial.suggest_int("depth", 1, 10, log = False)
        param_dict['iterations'] = trial.suggest_int("iterations", 20, 600, log = True)
        param_dict['l2_leaf_reg'] = trial.suggest_float("l2_leaf_reg", 1e-2, 1e1, log = True)
        param_dict['learning_rate'] = trial.suggest_float("learning_rate", 1e-3, 1e0, log = True)
        param_dict['rsm'] = trial.suggest_float("rsm", 1e-2, 1e0, log = False)
        param_dict['early_stopping_rounds'] =  trial.suggest_categorical("early_stopping_rounds", [20])
        param_dict['logging_level'] = trial.suggest_categorical("logging_level", ['Silent'])
        param_dict['random_seed'] = trial.suggest_categorical("random_seed", [random_state])
        return param_dict
    
    def bayesianRidgeHParams(trial):
        param_dict = {}
        param_dict['n_iter'] = trial.suggest_int("n_iter", 10, 400)
        param_dict['tol'] = trial.suggest_float("tol", 1e-8, 1e2)
        param_dict['alpha_1'] =  trial.suggest_float("alpha_1", 1e-8, 1e2, log = True)
        param_dict['alpha_2'] = trial.suggest_float("alpha_2", 1e-8, 1e2, log = True)
        param_dict['lambda_1'] = trial.suggest_float("lambda_1", 1e-8, 1e2, log = True)
        param_dict['lambda_2'] = trial.suggest_float("lambda_2", 1e-8, 1e2, log = True)
        return param_dict
    
    def lassoLarsHParams(trial):
        param_dict = {}
        param_dict['alpha'] = trial.suggest_float("alpha", 1e-8, 1e2, log = True)
        param_dict['normalize'] = trial.suggest_categorical("normalize", [False])
        param_dict['random_state'] =  trial.suggest_categorical("random_state", [random_state])
        return param_dict
    
    def adaBoostHParams(trial):
        param_dict = {}
        param_dict['n_estimators'] =trial.suggest_int("n_estimators", 10, 200)
        param_dict['learning_rate'] = trial.suggest_float("learning_rate", 1e-2, 1e0, log = True)
        param_dict['loss'] = trial.suggest_categorical("loss", ['linear', 'square', 'exponential'])
        param_dict['random_state'] =  trial.suggest_categorical("random_state", [random_state])
        return param_dict
    
    def gradBoostHParams(trial):
        param_dict = {}
        param_dict['n_estimators'] =trial.suggest_int("n_estimators", 10, 300)
        param_dict['learning_rate'] = trial.suggest_float("learning_rate", 1e-2, 1e0, log = True)
        param_dict['subsample'] = trial.suggest_float("subsample", 1e-2, 1.0, log = False)
        param_dict['max_depth'] = trial.suggest_int("max_depth", 1, 10)
        param_dict['criterion'] = trial.suggest_categorical("criterion", ['friedman_mse', 'squared_error'])
        param_dict['loss'] = trial.suggest_categorical("loss", ['squared_error', 'absolute_error', 'huber', 'quantile'])
        param_dict['n_iter_no_change'] = trial.suggest_categorical("n_iter_no_change", [20])
        param_dict['random_state'] =  trial.suggest_categorical("random_state", [random_state])
        return param_dict
    
    def histGradBoostHParams(trial):
        param_dict = {}
        param_dict['loss'] = trial.suggest_categorical("loss", ['squared_error', 'absolute_error'])
        param_dict['max_depth'] = trial.suggest_int("max_depth", 1, 20, log = False)
        param_dict['max_iter'] = trial.suggest_int("max_iter", 10, 500, log = False)
        param_dict['max_leaf_nodes'] = trial.suggest_int("max_leaf_nodes", 2, 100)
        param_dict['min_samples_leaf'] = trial.suggest_int("min_samples_leaf", 2, 200)
        param_dict['learning_rate'] = trial.suggest_float("learning_rate", 1e-4, 1.0, log = True)
        # param_dict['scoring'] = trial.suggest_categorical("scoring", [make_scorer(metric)])
        param_dict['n_iter_no_change'] = trial.suggest_categorical("n_iter_no_change", [20]) 
        param_dict['random_state'] = trial.suggest_categorical("random_state", [random_state])
        return param_dict
    
    def KNearNeighboursHParams(trial):
        param_dict = {}
        param_dict['n_neighbors'] = trial.suggest_int("n_neighbors", 1, 101, step = 5)
        param_dict['weights'] = trial.suggest_categorical("weights", ['uniform', 'distance'])
        return param_dict
    
    def sgdHParams(trial):
        param_dict = {}
        param_dict['loss'] =  trial.suggest_categorical("loss", ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'])
        param_dict['penalty'] = trial.suggest_categorical("penalty", ['l2', 'l1', 'elasticnet'])
        param_dict['alpha'] = trial.suggest_float("alpha", 1e-8, 1e2, log = True)
        param_dict['random_state'] =  trial.suggest_categorical("random_state", [random_state])
        return param_dict
    
    def baggingHParams(trial):
        param_dict = {}
        param_dict['n_estimators'] =  trial.suggest_int("n_estimators", 1, 101, step = 5)
        param_dict['max_features'] = trial.suggest_float("max_features", 1e-1, 1.0, step = 0.1)
        param_dict['random_state'] =  trial.suggest_categorical("random_state", [random_state])
        return param_dict
    
    def svrHParams(trial):
        param_dict = {}
        param_dict['loss'] = trial.suggest_categorical("loss", ['epsilon_insensitive', 'squared_epsilon_insensitive'])
        param_dict['C'] = trial.suggest_float("C", 1e-5, 1e2, log = True)
        param_dict['tol'] = trial.suggest_float("tol", 1e-8, 1e2, log = True)
        param_dict['random_state'] = trial.suggest_categorical("random_state", [random_state])
        return param_dict
    
    def elasticnetHParams(trial):
        param_dict = {}
        param_dict['alpha'] = trial.suggest_float("alpha", 1e-8, 1e2, log = True)
        param_dict['l1_ratio'] = trial.suggest_float("l1_ratio", 1e-5, 1.0, log = True)
        param_dict['random_state'] = trial.suggest_categorical("random_state", [random_state])
        return param_dict
    
    # possible regressors
    methods = {
        "dummy":( DummyRegressor, dummyHParams),
        "lightgbm": (LGBMRegressor, lightgbmHParams),
        "xgboost": (XGBRegressor, xgboostHParams),
        "catboost": (CatBoostRegressor, catboostHParams),
        "bayesianridge": (BayesianRidge, bayesianRidgeHParams),
        "lassolars": (LassoLars, lassoLarsHParams),
        "adaboost": (AdaBoostRegressor, adaBoostHParams),
        "gradientboost": (GradientBoostingRegressor, gradBoostHParams),
        "histgradientboost": (HistGradientBoostingRegressor, histGradBoostHParams),
        "knn": (KNeighborsRegressor, KNearNeighboursHParams),
        "sgd": (SGDRegressor, sgdHParams),
        "bagging": (BaggingRegressor, baggingHParams),
        "svr": (LinearSVR, svrHParams),
        "elasticnet": (ElasticNet, elasticnetHParams),
        }
    
    return methods


def splitTrainTest(x_data, y_validation, testSize = 0.3, randomState = 42, n_splits = 1, smote = False, equalSizedClasses = False, classToIgnore = None, continuous = False):
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import ShuffleSplit
    
    """
    Equation to split a dataframe into arrays of training and testing values.
    
    Optionally choose:
        smote: Wether to artificially oversample from under represented classes to create equal sized classes (only for categorical data)
        equalSizedClasses: Wether to pick the minimum number of points per class among all classes such that there are equal points per class
            (e.g. if two classes with 20 and 10 points respectively it will select 10 points per class)
        classToIgnore: Int or list of ints with classes to ignore 
        continuous: if False data is split into classes and input ratio of classes is identical in training and testing. 
                    If data is continuous no input ratio is considered (unstratified). 
                    Does not work with 'equalSizedClasses' or 'classToIgnore'
    
    """
    y_validation = pd.DataFrame(y_validation.values, columns = ['val'])
    
    
    if equalSizedClasses == True:
        # take equal number of points from each error class
        min_number_among_classes = y_validation.groupby('val', group_keys = False).apply(lambda x: x.count()).min().values[0]

        # apply random smaple using minimum number to achieve equal sized classes
        y_validation = y_validation.groupby('val', group_keys = False).apply(lambda x: x.sample(n = min_number_among_classes))
        equalSizeIndexes = y_validation.index
        x_data = x_data[x_data.index.isin(equalSizeIndexes)]
    
    # input is dataframe so convert to arrays
    y_validation = np.ravel(y_validation.reset_index(drop=True).values)
    x_data = x_data.reset_index(drop=True).values
    
    
    # determine whether to use a stratified or unstratified classifier
    if continuous == False: # if objective consists of classes and not continuous values ...
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=testSize, random_state=randomState)  #... split the data taking into account classes ()
    elif continuous == True: # if objectvie consists of v=continuous values
        sss = ShuffleSplit(n_splits=n_splits, test_size=testSize, random_state=randomState)
    
    for train_index, test_index in sss.split(x_data , y_validation):
        
        # if specified ignore specific class
        if classToIgnore != None:
            if np.shape(classToIgnore) == ():
                train_index = train_index[y_validation[train_index]!=classToIgnore]
            else: 
                for i in classToIgnore:
                    train_index = train_index[y_validation[train_index]!=i]

        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_validation[train_index], y_validation[test_index]
        
        if smote == True:
            from imblearn.over_sampling import SMOTE
            oversample = SMOTE(random_state=randomState)
            x_train, y_train = oversample.fit_resample(x_train, y_train)
            
        return x_train, x_test, y_train, y_test, test_index, train_index
    
    
def scaler_chooser(scaler_str):
    """
    Function outputs a scaler function corresponding to input string
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    
    if scaler_str == "minmax":
        return MinMaxScaler()
    elif scaler_str == "standard":
        return StandardScaler()
    elif scaler_str == "robust":
        return RobustScaler()
    return None


def pca_chooser(trial = None, **kwargs):
    """
    Function outputs a pca function corresponding to input
    """
    
    if kwargs.get('pca_value') is not None:
        from sklearn.decomposition import PCA
        type_ref = type(kwargs.get('pca_value'))
        
        if type_ref is dict:
            pca = PCA(**kwargs.get('pca_value'))
        elif type_ref is int or type_ref is float or type_ref is str:
            pca = PCA(kwargs.get('pca_value'))
        if trial != None:
            trial.suggest_categorical('pca_value', [pca.get_params()])
    else:
        pca = None
        if trial != None:
            trial.suggest_categorical('pca_value', [None])
        
    return pca


def poly_chooser(trial = None, **kwargs):
    """
    Function to transform input variables using polynomial features
    """
    
    if kwargs.get('poly_value') is not None:
        from sklearn.preprocessing import PolynomialFeatures
        type_ref = type(kwargs.get('poly_value'))
        
        if type_ref is dict:
            poly = PolynomialFeatures(**kwargs.get('poly_value'))
        elif type_ref is int or type_ref is float:
            poly = PolynomialFeatures(degree = kwargs['poly_value'])
        if trial != None:
            trial.suggest_categorical('poly_value', [poly.get_params()])
    else:
        poly = None
        if trial != None:
            trial.suggest_categorical('poly_value', [None])
        
    return poly



def spline_chooser(feature_combo = False, trial = None, **kwargs):
    """
    Function to transform input variables using spline features
    """
    
    if (kwargs.get('spline_value') is not None) :#& (feature_combo == True):
        from sklearn.preprocessing import SplineTransformer
        type_ref = type(kwargs.get('spline_value'))
        
        if type_ref is dict:
            spline = SplineTransformer(**kwargs.get('spline_value'))
        elif type_ref is tuple or type_ref is list:
            spline = SplineTransformer(*kwargs.get('spline_value'))
        elif type_ref is int:
            spline = SplineTransformer(kwargs.get('spline_value'))
        # elif (trial != None) & (type_ref is bool):
        #     n_knots = trial.suggest_categorical("spline_n_knots", [int(i) for i in np.linspace(2, 10, 9)])
        #     degree = trial.suggest_categorical("spline_degree", [int(i) for i in np.linspace(1, 10, 10)])
        #     knots = trial.suggest_categorical('spline_knots', ['uniform', 'quantile'])
        #     spline = SplineTransformer(n_knots = n_knots, degree = degree, knots = knots)
        if (trial != None) & (type_ref is not bool):
            trial.suggest_categorical('spline_value', [spline.get_params()])
            # trial.suggest_categorical("spline_n_knots", [spline.get_params()['n_knots']])
            # trial.suggest_categorical("spline_degree", [spline.get_params()['degree']])
            # trial.suggest_categorical('spline_knots', [spline.get_params()['knots']])
    else:
        spline = None
        if trial != None:
            trial.suggest_categorical('spline_value', [None])
            # trial.suggest_categorical("spline_n_knots", [None])
            # trial.suggest_categorical("spline_degree", [None])
            # trial.suggest_categorical('spline_knots', [None])
            
    # if (kwargs.get('spline_value') is not None) & (feature_combo == True) & (list(set(kwargs.keys()) & set(['spline_n_knots', 'spline_degree', 'spline_knots'])) != []):
    #     spline = SplineTransformer(n_knots = kwargs.get('spline_n_knots'), degree = kwargs.get('spline_degree'), knots = kwargs.get('spline_knots'))
        
    return spline
    

def transformer_chooser(transformer_str, trial = None, n_quantiles = 500, random_state = 42):
    """
    Function outputs a transformer function corresponding to input string
    """
    
    from sklearn.preprocessing import QuantileTransformer
    
    if transformer_str == "none":
        return None
    elif transformer_str == "quantile_trans":
        
        # -- if optuna trial is provided to function determine optimal number of quantiles
        if trial != None:
            n_quantiles = trial.suggest_int('n_quantiles', 100, 4000, step = 100)
            
        return QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", random_state = random_state)

        

def model_performance(trial, X_train, y_train, cross_validation, pipeline, study_name, metric = median_absolute_error, random_state = 42):
    """
    function for splitting, training, assessing and pruning the regressor
    
    1. First the data is split into K-folds. 
    2. Iteratively an increasing fraction of the training and tes folds and test fold is taken
    3. The regressor is trained and assessed iteratively
    4. If performance is for first iterations is poor, regressor is pruned thus preventing training and testing on full dataset
    
    Input:
        trial: optuna trial (machine learning run with hyperparameters selected by optuna)
        X_train: array of N samples with M measurements
        y_train: array of N validation values
        cross_validation: method of cross valdiation for splitting data into folds
        pipeline: pipeline serving as a regressor for which the optuna trial is optimizing
                  i.e. the pipeline is the regressor being tested
                  
    Output:
        MAE: Median Absolute error of regressor
        MAE_std: standard deviation of MAE
        r2: r2 score of truth and regressor estimate
        r2_std: standard deviation r2
    """
    
    # -- turn train and test arrays into temporary dataframes
    df_X_train = pd.DataFrame(X_train)
    df_y_train = pd.DataFrame(y_train)
    
    # -- Retrieve list containing with dataframes for training and testing for each fold
    indexes_train_kfold = list(cross_validation.split(df_X_train))
    
    result_folds_fracs = []
    result_folds_stds = []

    # -- For each fraction value...
    for idx_fraction, partial_fit_frac in enumerate([0.1, 0.2, 0.3, 0.4, 0.6, 1]):
        
        # -- prepare storage lists
        result_folds = []
        
        # -- select the fraction of the fold ...
        for idx_fold, fold in enumerate(indexes_train_kfold):
            
            # ... select a fold 
            fold_X_train = df_X_train.iloc[fold[0]]
            fold_X_test = df_X_train.iloc[fold[1]]
            fold_y_train = df_y_train.iloc[fold[0]]
            fold_y_test = df_y_train.iloc[fold[1]]
            
            # ... retrieve indexes belonging to fraction of the fold 
            idx_partial_fit_train = pd.DataFrame(fold_X_train).sample(frac = partial_fit_frac, random_state= random_state).index
            idx_partial_fit_test = pd.DataFrame(fold_X_test).sample(frac = partial_fit_frac, random_state= random_state).index

            # ... select fraction of fold 
            fold_X_train_frac = fold_X_train.loc[idx_partial_fit_train]
            fold_X_test_frac = fold_X_test.loc[idx_partial_fit_test]
            fold_y_train_frac = fold_y_train.loc[idx_partial_fit_train]
            fold_y_test_frac = fold_y_test.loc[idx_partial_fit_test]
            
            # -- determine if regressor is boosted model
            regressor_is_boosted = bool(set([study_name]) & set(['lightgbm', 'xgboost'])) #catboost ignored, bugs  out
            
            # -- fit training data and add early stopping function if X-iterations did not improve data
            # ... if regressor is boosted ...
            if regressor_is_boosted:
                
                # -- fit transformers to training fold of training data
                fold_X_train_frac_transformed = pipeline[:-1].fit_transform(fold_X_train_frac)
                # -- transform testting fold of training data
                fold_X_test_frac_transformed = pipeline[:-1].transform(fold_X_test_frac)
                
                # fit pipeline using pre-fitted transformers
                pipeline.fit(fold_X_train_frac_transformed, fold_y_train_frac, 
                              regressor__eval_set=[(fold_X_test_frac_transformed, fold_y_test_frac)],
                              regressor__early_stopping_rounds=20)
                
            # ... if regressor is NOT boosted ...
            else:
                # -- fit training data
                pipeline.fit(fold_X_train_frac, fold_y_train_frac)
                
            # ... assess fold performance, sometimes performance is so poor a value error is thrown, therefore insert in 'try' function and return nan's for errors
            try:
                # ... if regressor is boosted ...
                if regressor_is_boosted:
                    # ... make fold prediction on transformed test fraction of training dataset
                    prediction = pipeline.predict(fold_X_test_frac_transformed)
                else:
                    # ... make fold prediction on original test fraction of training dataset
                    prediction = pipeline.predict(fold_X_test_frac)
                    
                    # ... assess prediction with chosen metric
                result_fold = metric(fold_y_test_frac, prediction) 
                pass
            except Exception as e:
                print(e)
                result_fold = np.nan
                pass
            
            # ... store results to assess performance per fraction
            result_folds.append(result_fold)
        
        # -- Calculate mean and std results from all folds per fraction of data
        result_folds_frac = np.mean(result_folds)
        result_folds_std = np.std(result_folds)
        
        # -- Save results
        result_folds_fracs.append(result_folds_frac); result_folds_stds.append(result_folds_std); 
        
        # -- only prune if not applied on fraction containing all datapoints 
        if partial_fit_frac < 1.0:
            
            # -- Report results to decide wether to prune
            trial.report(result_folds_frac, idx_fraction)
            
            # -- Prune the intermediate value if neccessary.
            if trial.should_prune():
                raise optuna.TrialPruned()

    # -- final results are those obtained for last fraction (e.g. fraction of 1/1)
    return result_folds_fracs[-1]



def create_objective(study_name, write_path, regressor_class, create_params, X_training, y_training, study, cross_validation, metric = None, random_state = 42, **kwargs):
    """
    Nested function containing the optuna objective which has to be mini/maximised
    
    input:
        study_name: str, custom name to save study in folder
        write_path: str, path to save study
        regressor_class: regressor loaded from methods
        create_params: function loading the parameters associated with each regressor
        X_training: n x m array, independent training data
        y_training: n x 1 array, dependent training data
        study: optuna optimisation study 
        cross_validation: method of cross validation
        metric: function, optimisation metric used to measure optimasation performance e.g. metric = sklearn.metrics.r2_score
            kwargs: 
                pca_value: int or float, pca compression to apply after scaling of X matrix
                poly_value: int, creates polynomial expension of degree i of X matrix
                spline_value: int, float, list, dict, creates spline expansion with n_knots and of degree i of X matrix
        
    """
    
    def objective(trial):
    
        # save optuna study
        joblib.dump(study, write_path + '/' + study_name + '.pkl')
        
        # -- Instantiate scaler for independents
        scalers = trial.suggest_categorical("scalers", [None, 'minmax', 'standard', 'robust'])
        scaler = scaler_chooser(scalers)
        
        # -- determine if requested feature combinations improve results
        # -- only suggest this to trial of kwargs contain at least one of the relevant parameters
        if any([i in kwargs for i in ['spline_value', 'pca_value','poly_value']]):
            
            # -- suggest either to include feature combination or not
            feature_combo = trial.suggest_categorical("feature_combo", [False, True])
            
            # # # -- instantiate spline transformer if relevant kwargs included
            # spline = spline_chooser(feature_combo = feature_combo, trial = trial, **kwargs)
            
            # -- if trial will try using feature combinations/compression
            if feature_combo == True:
                # -- instantiate pca compression if relevant kwargs included
                pca = pca_chooser(trial = trial, **kwargs)
    
                # -- instantiate spline transformer if relevant kwargs included
                spline = spline_chooser(feature_combo = feature_combo, trial = trial, **kwargs)
                
                # -- instantiate polynomial transformer if relevant kwargs included
                poly = poly_chooser(trial = trial, **kwargs)
            else:
                # pca = poly = None
                pca = spline = poly = None
                
        else:
            pca = spline = poly = None
            
        # -- Instantiate transformer for dependends
        transformers = trial.suggest_categorical("transformers", ['none', 'quantile_trans'])
        transformer = transformer_chooser(transformers, trial = trial, random_state=random_state)

        # -- Tune estimator algorithm
        param = create_params(trial)
    
        # -- Create regressor
        regressor = regressor_class()
        regressor.set_params(**param)
        
        # -- Create transformed regressor
        transformed_regressor = TransformedTargetRegressor(
            regressor = regressor,
            transformer = transformer
            )
        
        # -- Make a pipeline
        pipeline = Pipeline([('poly', poly), ('spline', spline), ('scaler', scaler), ('pca', pca), ('regressor', transformed_regressor)])
        
        # -- Assess model performance using specified cross validation on pipeline with pruning
        result = model_performance(trial, X_training, y_training, cross_validation, pipeline, study_name, metric, random_state)
        return result 
    return objective


    
def regressor_optimise(methods, optimisation_direction, list_regressors_hyper, X_train, y_train, sampler, pruner, metric, cross_validation, 
                       poly_value, spline_value, pca_value, n_trial,timeout, write_folder, overwrite, random_state):
    """
    Function performs the optuna optimisation for filtered list of methods (methods_filt) on training data
    """
    
    # -- only perform optimisation if regressors are given, else proceed with assessing previously optimized regressors
    if bool(list_regressors_hyper):
    
        # -- only select specified regressors
        methods_filt = {regressor_key: methods[regressor_key] for regressor_key in list_regressors_hyper}
        
        for regressor_name, (regressor, create_params) in methods_filt.items():
            study = optuna.create_study(direction = optimisation_direction, sampler = sampler, pruner = pruner)
            
            write_file = write_folder + regressor_name + '.pkl'
            # -- if regressor already trained, throw warning unless overwrite  == True
            if os.path.isfile(write_file):
                if overwrite != True:
                    message = "Regressor already exists in directory but overwrite set to 'False'. Regressor skipped." 
                    print(len(message)*'_' + '\n' + message + '\n' + len(message)*'_')
                    continue
                if overwrite == True:
                    message = "Regressor already exists in directory. Overwrite set to 'TRUE'"
                    print(len(message)*'_' + '\n' + message + '\n' + len(message)*'_')
            
            study.optimize(create_objective(regressor_name, write_folder, regressor, create_params, metric = metric,
                                            X_training = X_train, y_training = y_train, study = study , cross_validation = cross_validation, random_state = random_state,
                                            poly_value = poly_value, spline_value = spline_value, pca_value = pca_value), # < -- optional poly, spline and pca attributes
                           n_trials=n_trial, timeout=timeout)
            
            # the study is saved during each trial to update with the previous trial (this stores the data even if the study does not complete)
            # here the study is saved once more to include the final iteration
            joblib.dump(study, write_file)
         
        return
    
def regressor_fit(list_regressors_training, write_folder):
    """
    Loads hyperoptimisation trials for regressors specified in list_regressors_training, located in write_folder. 
    Selects hyperparameters belonging to best trial and outputs a list of tuples containing optimized regressor name with corresponding pipeline
    
    Intput:
        list_regressors_training: list of str, contains regressor id's of regressors to fit following best trial's hypeperparameters
        write_folder: location with hyperparameter optimisation trials for regressors specified in list_regressors_training
        
    Output:
        estimators: list of tuples containing 
    
    """
    
    estimators = []
    parameter_dict_dict = {}
    for regressor_name in list_regressors_training:
        study = joblib.load(write_folder + regressor_name + '.pkl')
        
        list_params = list(study.best_params)
        
        # parameters 'scalers', 'transformers', 'n_quantiles', 'pca_value', 'spline_value', 'poly_value', 'feature_combo' are not specific to regressors
        # they must be read (if present) and then removed such that they are not fitted to pipeline
        list_params.remove('scalers'); list_params.remove('transformers'); 
        
        # -- n_quantiles is not always selected therefore it can only be removed if present
        if 'n_quantiles' in list_params :
            n_quantiles = study.best_params['n_quantiles']
            list_params.remove('n_quantiles')
        else :
            n_quantiles = None
        
        # -- if combinations of features or feature compression is performed...
        if 'feature_combo' in list_params:
            if study.best_params['feature_combo'] == True:
                # -- instantiate pca compression if relevant kwargs included       
                pca = pca_chooser(**study.best_params)
                # -- instantiate spline transformer if relevant kwargs included
                spline = spline_chooser(**study.best_params)
                # -- instantiate polynomial transformer if relevant kwargs included
                poly = poly_chooser(**study.best_params)
                
                # -- remove unnecessary params
                list_params.remove('pca_value')
                list_params.remove('spline_value')
                list_params.remove('poly_value')
            else:
                pca = spline = poly = None
            list_params.remove('feature_combo')
        else:
            pca = spline = poly = None
    
        parameter_dict = {k: study.best_params[k] for k in study.best_params.keys() & set(list_params)}
        parameter_dict_dict[regressor_name] = parameter_dict
        
        # study must contain parameters "scalers", "transformers", "n_quantiles"
        pipe_single_study = Pipeline([
            ('poly', poly),
            ('spline', spline),
            ('scaler', scaler_chooser(study.best_params['scalers'])),
            ('pca', pca),
            ('model', TransformedTargetRegressor(
                    regressor = regressor_dict[regressor_name](**parameter_dict), 
                    transformer= transformer_chooser(study.best_params['transformers'], n_quantiles = n_quantiles )
                    ))]
            )
        estimators.append((regressor_name, pipe_single_study))
        
    return estimators
        


def regression_assess(X_train, X_test, y_train, y_test, list_regressors_training, estimators, metric_assess, cross_validation, write_folder, overwrite, random_state):
    """
    Loads testing and training data. Train individual regressors and assess performance using specified metrics
    
    Input:
        X_train: array (n, m), selection of X-matrix on which to train
        y_train: array (n, ), selection of y-matrix on which to train
        X_test: array (n, m), selection of X-matrix on which to test
        y_test: array (n, ), selection of y-matrix on which to test
        list_regressors_training: list of str, regressors to assess
        estimators: list of tuples containing name and pipeline per 
        metric_assess: list of function(s), metrics with which to assess performance of individual and stacked regressors
        cross_validation: cross valdiator function, sklearn cross_valdiator used in determining cross validated performance of regressors. E.g. sklearn.model_selection.KFold()
        write_folder: str, name of the folder to be created where to store individually trained regressors and stacked regressor. Directory will be created if does not exist
        overwrite: Boolean, whether to overwrite if files in write_folder already exist 
        random_state: int, set random state for reproducibility
        
    Return:
        metric_performance_summary_dict: dict, contains mean and std. performance per metric per regressor in list_regressors_training
        idexes_test_kfold: list of tuples of numpary arrays, contains fold training and test indexes for test dataset
        y_pred: array, contains estimates of y predicted on the test dataset
    
    """
    
    
    # -- create dataframe just to retrieve fold indexes from K-fold validation for TEST data
    df_X_test = pd.DataFrame(X_test)
    idexes_test_kfold = list(cross_validation.split(df_X_test))
    
    regressors_to_assess = list_regressors_training + ['stacked']
    
    # -- create an empty dictionary to populate with performance while looping over regresors
    metric_performance_summary_dict = dict([(regressor, list()) for regressor in regressors_to_assess])
    
    for i, regressor in enumerate(regressors_to_assess):
    
        estimator_temp = estimators[i:i+1]
        
        # -- if independet X contains a single column it must be reshaped, else estimate.fit() fails
        X_train, X_test = (X_train.reshape(-1, 1), X_test.reshape(-1, 1)) if X_train.ndim == 1 else (X_train, X_test)
        
        # -- the final regressor is the stacked regressor
        if i == len(estimators):
            estimator_temp = estimators
        
            regressor_final = StackingRegressor(estimators=estimator_temp, final_estimator=Ridge(random_state = random_state), cv = cross_validation)
            regressor_final.fit(X_train, y_train)
            
            # -- predict on the whole testing dataset
            y_pred = regressor_final.predict(X_test)
            
            # -- store stacked regressor
            write_file_stacked_regressor = write_folder + "stacked_regressor.joblib"
                
            # -- file does exist, double check whether the user wants to overwrite or not
            if os.path.isfile(write_file_stacked_regressor):
                if overwrite != True:
                    user_input = input("Stacked Regressor already exists in directory but overwrite set to 'False'. Overwrite anyway ? (y/n): ")
                    if user_input != 'y':
                        message = "Stacked Regressor already exists in directory but overwrite set to 'False'. Stacked regressor not saved." 
                        print(len(message)*'_' + '\n' + message + '\n' + len(message)*'_')
                if overwrite == True:
                    user_input = input("Stacked Regressor already exists in directory. Overwrite set to 'TRUE'. Are you certain ? (y/n): ")
                    if user_input != 'n':
                        message = "Stacked Regressor already exists in directory. Overwrite set to 'TRUE'"
                        print(len(message)*'_' + '\n' + message + '\n' + len(message)*'_')
                        joblib.dump(regressor_final, write_folder + "stacked_regressor.joblib")
                        
            # -- if file doesnt exist, write it
            if not os.path.isfile(write_file_stacked_regressor):
                joblib.dump(regressor_final, write_folder + "stacked_regressor.joblib")
            
        else:
            regressor_final = estimator_temp[0][1]
            regressor_final.fit(X_train, y_train)
        
        # -- create dictionary with elements per metric allowing per metric fold performance to be stored
        metric_performance_dict = dict([('metric_' + str(i), [metric, list()] ) for i, metric in enumerate(metric_assess)])
        
        # -- For each TEST data fold...
        for idx_fold, fold in enumerate(idexes_test_kfold):
            
            # -- Select the fold indexes        
            fold_test = fold[1]
    
            # -- Predict on the TEST data fold
            prediction = regressor_final.predict(X_test[fold_test, :])
            
            # -- Assess prediction per metric and store per-fold performance in dictionary
            [metric_performance_dict[key][1].append(metric_performance_dict[key][0](y_test[fold_test], prediction)) for key in metric_performance_dict]
        
        # -- store mean and standard deviation of performance over folds per regressor
        metric_performance_summary_dict[regressor] = [[np.mean(metric_performance_dict[key][1]), np.std(metric_performance_dict[key][1])] for key in metric_performance_dict]
        
    return metric_performance_summary_dict, idexes_test_kfold, y_pred
