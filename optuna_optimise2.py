#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:50:49 2022

@author: owen
"""

# %reset -f

import xarray as xr
import numpy as np
import pandas as pd 
import os

from matplotlib import pyplot as plt 
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from tqdm.auto import tqdm
tqdm.pandas()


path = '/home/owen/Documents/python scripts'
os.chdir(path)
import equations as eq
import importlib
importlib.reload(eq)

#%%
pd.options.mode.chained_assignment = None 

from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import median_absolute_error, mean_absolute_error, r2_score

from sklearn.preprocessing import QuantileTransformer, quantile_transform, FunctionTransformer

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
import time

from sklearn.linear_model import Lasso, LassoLars, Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import shap

import optuna
from optuna.samplers import TPESampler, RandomSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, cross_val_score, KFold, cross_validate
from sklearn.neighbors import LocalOutlierFactor

#%%


def calc_PSD_spatial(KinHeatFlux, Beta, xi, Tv, Alpha = 0.5, Psi = 1, g = 9.81):
    """
    Calculates S(xi) based from kinematic ehat flux and several parameters approximates by constants
        
    """
    Alpha                                   # Kolmogorov constant
    g                                       # Gravitational acceleration, m/s**2
    Psi                                     # dimensionless dissipation rate, 
    Tv                                      # Virtual potential temperature, K
    
    part1 = Alpha * Beta / xi **(5/3)
    part2 = (Psi * KinHeatFlux * g) / (2 * np.pi * Tv)**(2/3)
        
    PSD_spatial = part1 * part2
    
    PSD_spatial = float(np.where(PSD_spatial < 0, np.nan, PSD_spatial))
    
    return pd.Series([PSD_spatial])



def sigma_u_panofsky(L, Zi, u_star):
    """
    From Panofsky et al 1977
    
    """
    sigma_u = u_star * np.sqrt(4 + 0.6 * (-Zi / L)**(2/3))
    return sigma_u

def envelope(df, param_x, param_y, begin, end, steps =25, log = True):
    """
    function to derive the median and quantiles for a pointcloud from a df with two specified parameters
    """
    placeholder = df
    
    if log == True:
        bins = np.logspace(begin, end, steps)
    else:
        bins=np.linspace(begin, end, steps)
        
    placeholder['bins'], bins = pd.cut(abs(placeholder[param_x]), bins=bins, include_lowest=True, retbins=True)
        
    bin_center = (bins[:-1] + bins[1:]) /2
    bin_median = abs(placeholder.groupby('bins')[param_y].agg(np.nanmedian))#.nanmedian())
    bin_count = abs(placeholder.groupby('bins')[param_y].count())
    bin_std = abs(placeholder.groupby('bins')[param_y].agg(np.nanstd)) #.nanstd())
    bin_quantile_a = abs(placeholder.groupby('bins')[param_y].agg(lambda x: np.nanpercentile(x, q = 2.5)))
    bin_quantile_b = abs(placeholder.groupby('bins')[param_y].agg(lambda x: np.nanpercentile(x, q = 16)))
    bin_quantile_c = abs(placeholder.groupby('bins')[param_y].agg(lambda x: np.nanpercentile(x, q = 84)))
    bin_quantile_d = abs(placeholder.groupby('bins')[param_y].agg(lambda x: np.nanpercentile(x, q = 97.5)))
    return bin_center, bin_median, bin_count, bin_std, bin_quantile_a, bin_quantile_b, bin_quantile_c, bin_quantile_d


def model_performance(trial, X_train, y_train, cross_validation, pipeline):
    
    """
    function for splitting, training, assessing and pruning the regressor
    
    1. First the data is split into K-folds. 
    2. Iteratively an increasing fraction of the training folds and test fold is taken
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
    
    # -- Create K-fold split algorithm to apply on Dataframe --> cross_validation
    # --retrieve list containing with dataframes for training and testing for each fold
    indexes_train_kfold = list(cross_validation.split(df_X_train))
    
    r2_fracs = []
    r2_frac_stds = []
    MAE_fracs = []
    MAE_frac_stds = []
    
    # 
    # -- For each fraction value...
    for idx_fraction, partial_fit_frac in enumerate([0.2, 0.4, 0.6, 1]):
        
        r2_folds = []
        MAE_folds = []
        
        # -- select the fraction of the fold ...
        for idx_fold, fold in enumerate(indexes_train_kfold):
            
            # ... select a fold 
            fold_X_train = df_X_train.iloc[fold[0]]
            fold_X_test = df_X_train.iloc[fold[1]]
            fold_y_train = df_y_train.iloc[fold[0]]
            fold_y_test = df_y_train.iloc[fold[1]]
            
            # -- retrieve indexes belonging to fraction of the fold 
            idx_partial_fit_train = pd.DataFrame(fold_X_train).sample(frac = partial_fit_frac, random_state= 42).index
            idx_partial_fit_test = pd.DataFrame(fold_X_test).sample(frac = partial_fit_frac, random_state= 42).index

            # -- select fraction of fold 
            fold_X_train_frac = fold_X_train.loc[idx_partial_fit_train]
            fold_X_test_frac = fold_X_test.loc[idx_partial_fit_test]
            fold_y_train_frac = fold_y_train.loc[idx_partial_fit_train]
            fold_y_test_frac = fold_y_test.loc[idx_partial_fit_test]
            
            # ... fit to the regressor
            # pipeline = make_pipeline(LGBMRegressor())
            pipeline.fit(fold_X_train_frac, fold_y_train_frac)
            
            # ... make fold prediction
            prediction = pipeline.predict(fold_X_test_frac)
            
            # ... assess fold performance
            MAE_fold = -1* median_absolute_error(fold_y_test_frac, prediction)
            r2_fold = r2_score(fold_y_test_frac, prediction)
            
            MAE_folds.append(MAE_fold)
            r2_folds.append(r2_fold)
        
        # -- Calculate mean and std results per fraction of data
        r2_frac = np.mean(r2_folds)
        MAE_frac = np.mean(MAE_folds)
        
        r2_fracs_std = np.std(r2_folds)
        MAE_fracs_std = np.std(MAE_folds)
        
        # -- Save results
        r2_fracs.append(r2_frac); r2_frac_stds.append(r2_fracs_std); 
        MAE_fracs.append(MAE_frac);  MAE_frac_stds.append(MAE_fracs_std)
        
        # -- Report results to decide wether to prune
        trial.report(MAE_frac, idx_fraction)
        
        # Prune the intermediate value if neccessary.
        if trial.should_prune():
            raise optuna.TrialPruned()

    # -- final results are those obtained for last fraction (e.g. fraction of 1/1)
    MAE = MAE_fracs[-1]
    MAE_std = r2_frac_stds[-1]
    r2 = r2_fracs[-1]
    r2_std = r2_frac_stds[-1]
    return MAE, r2, MAE_std, r2_std



def create_objective(study_name, save_id, regressor_class, create_params):
    def objective(trial):
    
        # save optuna study
        joblib.dump(study, '/home/owen/Documents/models/optuna/' + study_name + '_' +save_id + '.pkl')
        
        # -- Instantiate scaler for independents
        scalers = trial.suggest_categorical("scalers", [None, 'minmax', 'standard', 'robust'])
        if scalers == "minmax":
            scaler = MinMaxScaler()
        elif scalers == "standard":
            scaler = StandardScaler()
        elif scalers == "robust":
            scaler = RobustScaler()
        else:
            scaler = None
        
        # -- Instantiate transformer for dependends
        transformers = trial.suggest_categorical("transformers", ['none', 'quantile'])
        if transformers == "none":
            transformer = None
        elif transformers == "quantile":
            transformer = QuantileTransformer(n_quantiles=500, output_distribution="normal")
                    
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
        pipeline = make_pipeline(scaler, transformed_regressor)
    
        # -- Assess model performance using specified cross validation on pipeline with pruning
        MAE, r2, MAE_std, r2_std = model_performance(trial, X_train, y_train, cross_validation, pipeline)
        return MAE
    
    return objective

#%% Load data

file_name = 'wdirwspd_11_05_v25_extra_era5'
file = '/home/owen/Documents/data/'+ file_name +  '.csv'
df_output = pd.read_csv(file)
df = df_output.copy()

#%% filter out specific data 

# remove if classification probability is below 50%
df_val = df.loc[df['prob'] >= 0.5]
outliers_prob = len(df) - len(df_val)
print('# outliers probability: ' + str(outliers_prob))

# remove if era5 claims either positive Obukhov length or sensible heat flux is into the ocean rather into the atmosphere (positive sshf)
df_val = df_val.loc[(df_val['L_era5'] < 0) & (df_val['sshf_era5'] < 0 )]
print('# outliers positive ERA5 L or heat flux into the ocean: ' + str(len(df) - outliers_prob - len(df_val)))

# remove if normalised standard deviation is too low (too pefect slope is presumed wrong)
number = len(df_val)
df_val = df_val[df_val.w_star_normalised_deviation >=0.01]
number2 =  len(df_val)
print('# w* normalised deviation < 0.01: ' + str(number - number2))

# remove if wind direction is exactly range direction  --> unnecessary when using hardcoded ERA5 wind
number = len(df_val)
df_val = df_val.loc[abs(((df_val['mean_ground_heading'] + 360)%360)%360 - df_val['wdir_estimate']) >= 0.5]
df_val = df_val.loc[abs(((df_val['mean_ground_heading'] + 360)%360 + 90)%360 - df_val['wdir_estimate']) >= 0.5]
df_val = df_val.loc[abs(((df_val['mean_ground_heading'] + 360)%360 + 180)%360 - df_val['wdir_estimate']) >= 0.5]
df_val = df_val.loc[abs(((df_val['mean_ground_heading'] + 360)%360 + 270)%360 - df_val['wdir_estimate']) >= 0.5]
number2 =  len(df_val)
print('# estimated wind direction is exactly along range: ' + str(number - number2))

# remove outliers in all observation columns
def outlier(df):
    return np.where((df> np.mean(df) + np.std(df)*4) | (df< np.mean(df) - np.std(df)*4))

keep_after_index = list(df_val.keys()).index('wspd_median') # wspd_median is first observation column
T = df_val.reset_index(drop = True).iloc[:,keep_after_index:].apply(outlier, axis = 0).values[0] # find rows with outliers in any of the estimates parameters
idx_delete = list(set([x for xs in T for x in xs])) # extract
print('# outliers: ' + str(len(idx_delete)))


# # create outlier detector
# outlier_detector = LocalOutlierFactor(n_neighbors=10)
# idx_measurements = list(df_val.keys()).index('wspd_median') # index of measurement start (variabels before are validation or metadata)
# idx_measurements_end = list(df_val.keys()).index('S_normalised_deviation')
# inliers = outlier_detector.fit_predict(df_val.iloc[:, idx_measurements:idx_measurements_end])
# df_val = df_val[inliers==1]
# number3 = len(df_val)
# print('# outliers: ' + str(number2 - number3))
# df_val = df_val.reset_index(drop = True)


df_val = df_val.reset_index(drop = True).drop(index = idx_delete)

##########################################################################################
######## pre filter on df, not df_val (such that shape stays identical )#################
##########################################################################################

# add spectral amplitude of wind field to df_val
Beta = 1   ###### !!!!  This value ought changes depending on cells or rolls (1 and 4/3 respectively)
spatial_freq = 1/600
# temperature for estimate is 293 because that was the approximation used, i.e. now the approximation is undone 
df_val['PSD_spatial']  = df_val.progress_apply(lambda x: calc_PSD_spatial( x['B'], Beta,spatial_freq, 293), axis=1)

# remove meta data and era5 but keep latitude and CNN prediction prob
df_obs = df_val.iloc[:,keep_after_index:] 
df_obs['prob'] = df_val['prob'] #### df_obs['prob'] = df_val['prob_1'] 
df_obs['lat'] = df_val['lat']
df_obs['lon'] = df_val['lon']

# add wdir error to validation
Error_wdir = df_val.wdir_estimate.values - df_val.wdir_era5.values
Error_wdir = np.where(Error_wdir>=270, Error_wdir - 360, Error_wdir)
Error_wdir = np.where(Error_wdir>=90, Error_wdir - 180, Error_wdir)
Error_wdir = np.where(Error_wdir<=-270, Error_wdir + 360, Error_wdir)
Error_wdir = np.where(Error_wdir<=-90, Error_wdir + 180, Error_wdir)
df_val['Error_wdir'] = Error_wdir
RSE_wdir = np.sqrt(Error_wdir**2)
df_val['RSE_wdir'] = RSE_wdir

# add wind direction in range to validation
step_1 = (360 - ((df_val.mean_ground_heading.values -90 + 360 ) % 360 - df_val.wdir_era5 ) % 360) % 180
df_val['wdir_range_era5'] = np.where(step_1 > 90, 90 - step_1 % 90, step_1)

# add Obukhov length calculated using the flux equations and era5 friction velocity rather than inputting the fluxes and windspeed into coare4
g = 9.8
kappa = 0.4
C_p = 1005
kinematic_heat_flux = -df_val.sshf_era5 / (df_val.rhoa_era5 * C_p) # sshf * -1 such that positive heat flux is into the atmosphere
df_val['L_era5_only'] = -(df_val.tair_era5 + 273.15) * df_val.friction_velocity_era5**3 / (g*kappa*kinematic_heat_flux)

L_param = 'L_estimate'

# add Obukhov length errors classes to validation
df_val['L_rel_error_era5'] = (df_val[L_param] - df_val.L_era5) / df_val.L_era5
df_val['L_rel_error_era5_only'] = (df_val[L_param] - df_val.L_era5_only) / df_val.L_era5_only

df_val['L_rel_error_era5_nosign'] = abs((df_val[L_param] - df_val.L_era5) / df_val.L_era5)
df_val['L_rel_error_era5_only_nosign'] = abs((df_val[L_param] - df_val.L_era5_only) / df_val.L_era5_only)

df_val['L_rel_era5'] = abs(df_val.L_era5) / abs(df_val[L_param])
df_val['L_rel_era5_only'] = abs(df_val.L_era5_only) / abs(df_val[L_param])

# add Obukhov length error classes to validation
df_val['sigma_u_era5'] = df_val.progress_apply(lambda x: sigma_u_panofsky( x['L_era5'], x['pblh_era5'], x['usr_era5']), axis=1)   # 'usr_era5' is friction velocity (u*) from coare4 using era5
df_val['sigma_u_era5_only'] = df_val.progress_apply(lambda x: sigma_u_panofsky( x['L_era5'], x['pblh_era5'], x['friction_velocity_era5']), axis=1)

# Errors w.r.t buoy
df_val['L_rel_error_buoy'] = (df_val[L_param] - df_val.L_buoy) / df_val.L_buoy
df_val['L_rel_error_buoy_nosign'] = abs((df_val[L_param] - df_val.L_buoy) / df_val.L_buoy)
df_val['L_rel_buoy'] = abs(df_val.L_buoy) / abs(df_val[L_param])

# add Obukhov length error classes to validation
df_val['sigma_u_buoy'] = df_val.progress_apply(lambda x: sigma_u_panofsky( x['L_buoy'], x['pblh_era5'], x['usr_era5']), axis=1)   # 'usr_era5' is friction velocity (u*) from coare4 using era5

#%% split data and prepare functions

############ All components #############
X = df_obs   # 

############ Components correlating with error (retrieved by kendall correlation and VIF) #################
# X = df_obs[['incidence_avg', 'var_highpass', 'var_lowpass', 'L', 'S', 'PSD_spatial']].iloc[:, :]

############ adding components from validation ##############
# X = pd.concat([df_obs, df_val['wdir_range_era5']], axis = 1).values

############ When applying PCA  #############
 # 1. specify number of pca components (or n_components = 0.95 such that 95% of explained variance is included)
 # 2. Uncomment 'scaler' and 'pca' in pipeline

############ Validation, transform optional ############
# y = np.log10(abs(df_val['L_era5_only'].iloc[:]))
y = np.log10(abs(df_val['L_buoy'].iloc[:]))
# y = np.log10(abs(df_val['L_era5_only'].iloc[:])) - np.log10(abs(df_val['L'].iloc[:]))

angleBinLow, angleBinHigh = 0, 91
X = X[(df_val['wdir_range_era5'] < angleBinHigh) & (df_val['wdir_range_era5'] >= angleBinLow)]
y = y[(df_val['wdir_range_era5'] < angleBinHigh) & (df_val['wdir_range_era5'] >= angleBinLow)]

parameters = X.keys()

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2)
X_train, X_test, y_train, y_test, test_index, train_index = eq.splitTrainTest(X, y, 
                                                                              testSize = 0.2, 
                                                                              randomState = 42, 
                                                                              n_splits = 1, 
                                                                              smote = False, 
                                                                              equalSizedClasses = False, 
                                                                              classToIgnore = None, 
                                                                              continuous = True)


#%%

timeout = 1000
n_trial = 200
sampler = TPESampler(42)# TPESampler(seed=42) # RandomSampler(seed=42)
cross_validation = KFold(n_splits = 10, shuffle = False, random_state= None)   #  KFold(n_splits=5)

# according to: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
# the best sampler for TPE is hyperband whereas for random it is median
# pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1)
# pruner = optuna.pruners.PercentilePruner(percentile = 25.0, n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3)
save_id = 'test4'





methods = {
    "lightgbm": (
        LGBMRegressor,
        lambda trial: {
            'objective': trial.suggest_categorical("objective", ['regression']),
            'metric': trial.suggest_categorical("metric", ['mean_squared_error']),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log = True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log = True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'random_state': 42  
        },
    ),
    "xgboost": (
        XGBRegressor,
        lambda trial: {
            'max_depth': trial.suggest_int("max_depth", 1, 20, log = False),
            'n_estimators': trial.suggest_int("n_estimators", 20, 400, log=False),
            'eta': trial.suggest_float("eta", 1e-4, 1.0, log = True),
            'subsample': trial.suggest_float("subsample", 0.1, 1.0, log = False),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1.0, log = False),
            'random_state': 42
        },
    ),
    "catboost": (
        CatBoostRegressor,
        lambda trial: {
            'depth': trial.suggest_int("depth", 1, 10),
            'iterations': trial.suggest_int("iterations", 50, 300),
            'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 0.1, 3),
            'learning_rate': trial.suggest_float("learning_rate", 1e-2, 1e0),
        },
    ),
    "ridge": (
        Ridge,
        lambda trial: {
            'alpha': trial.suggest_float("alpha", 1e-8, 1e2, log = True),
            'solver': trial.suggest_categorical("solver", ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'])
        },
    ),
    "bayesianridge": (
        BayesianRidge,
        lambda trial: {
            'n_iter': trial.suggest_int("n_iter", 10, 400),
            'tol': trial.suggest_float("tol", 1e-8, 1e2),
            'alpha_1': trial.suggest_float("alpha_1", 1e-8, 1e2, log = True),
            'alpha_2': trial.suggest_float("alpha_2", 1e-8, 1e2, log = True),
            'lambda_1': trial.suggest_float("lambda_1", 1e-8, 1e2, log = True),
            'lambda_2': trial.suggest_float("lambda_2", 1e-8, 1e2, log = True),
        },
    ),
    "adaboost": (
        AdaBoostRegressor,
        lambda trial: {
            'n_estimators': trial.suggest_int("n_estimators", 10, 100),
            'learning_rate': trial.suggest_float("learning_rate", 1e-2, 1e0, log = True),
            'loss': trial.suggest_categorical("loss", ['linear', 'square', 'exponential']),
            'random_state': 42,
        },
    ),
    "gradientboost": (
        GradientBoostingRegressor,
        lambda trial: {
            'n_estimators': trial.suggest_int("n_estimators", 10, 300),
            'learning_rate': trial.suggest_float("learning_rate", 1e-2, 1e0, log = True),
            'subsample': trial.suggest_float("subsample", 1e-2, 1.0, log = False),
            'max_depth': trial.suggest_int("max_depth", 1, 10),
            'criterion': trial.suggest_categorical("criterion", ['friedman_mse', 'squared_error']),
            'loss': trial.suggest_categorical("loss", ['squared_error', 'absolute_error', 'huber', 'quantile']),
            'random_state': 42,
        },
    ),
    "knn": (
        KNeighborsRegressor,
        lambda trial: {
            'n_neighbors': trial.suggest_int("n_neighbors", 1, 101, step = 5),
            'weights': trial.suggest_categorical("weights", ['uniform', 'distance']),
        },
    ),
    "sgd": (
        SGDRegressor,
        lambda trial: {
            'loss': trial.suggest_categorical("loss", ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
            'penalty': trial.suggest_categorical("penalty", ['l2', 'l1', 'elasticnet']),
            'alpha': trial.suggest_float("alpha", 1e-8, 1e2, log = True),
            'random_state': 42,
        },
    ),
    "bagging": (
        BaggingRegressor,
        lambda trial: {
            'n_estimators': trial.suggest_int("n_estimators", 1, 101, step = 5),
            'max_features': trial.suggest_float("max_features", 1e-1, 1.0, step = 0.1),
            'random_state': 42,
        },
    ),
    "svr": (
        LinearSVR,
        lambda trial: {
            'loss': trial.suggest_categorical("loss", ['epsilon_insensitive', 'squared_epsilon_insensitive']),
            'C': trial.suggest_float("C", 1e-5, 1e2, log = True),
            'tol': trial.suggest_float("tol", 1e-8, 1e2, log = True),
            'random_state': 42,
        },
    ),
    "elasticnet": (
        ElasticNet,
        lambda trial: {
            'alpha': trial.suggest_float("alpha", 1e-8, 1e2, log = True),
            'l1_ratio': trial.suggest_float("C", 1e-5, 1.0, log = True),
            'random_state': 42,
        },
    ),
    "lassolars": (
        LassoLars,
        lambda trial: {
            'alpha': trial.suggest_float("alpha", 1e-8, 1e2, log = True),
            'normalize': trial.suggest_categorical("normalize", [False]),
            'random_state': 42,
        },
    ),
}


methods = {
    "lightgbm": (
        LGBMRegressor,
        lambda trial: {
            'objective': trial.suggest_categorical("objective", ['regression']),
            'metric': trial.suggest_categorical("metric", ['mean_squared_error']),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log = True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log = True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'random_state': 42  
        },
    ),
}




studies = {}
for study_name, (regressor, create_params) in methods.items():
    study = optuna.create_study(direction = 'maximize', sampler=sampler, pruner=pruner)
    study.optimize(create_objective(study_name, save_id, regressor, create_params), n_trials=n_trial, timeout=timeout)
    
    # 
    joblib.dump(study, '/home/owen/Documents/models/optuna/' + study_name + '_'  + save_id + '.pkl')
    studies[study_name] = study

test = LGBMRegressor(**study.best_params)