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
import os, glob

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


def outlier_detector(df, column_key_start, column_key_end, pca_comp = 0.80, neighbours = 100, plot_PCA = False):
    """
    Function using SKlearn's 'LocalOutlierFactor' to detec outliers within a specififed range of the input df's columns
    
    Input:
        df: dataframe from which to select sub-dataframe
        column_key_start: name of column to start 
        column_key_end: nume of column end including column itself
        pca_comp: number of Principal components to include (int for number, float for fraction explained variance)
        neighbours: number of neighbours to conisder in localOutlier
        plot_PCA: if true will plot first two principal componenets and colour outliers
    """
    
    # -- Select specified columns
    idx_measurements = list(df_val.keys()).index(column_key_start)
    idx_measurements_end = list(df_val.keys()).index(column_key_end) + 1  # plus 1 because [5:10] does not include idx 10
    data = df.iloc[:, idx_measurements : idx_measurements_end]
    
    # -- change processing depending on whether PCA should be invoked or not
    if (type(pca_comp) == float) | (type(pca_comp) == int) :

        # -- apply standard scaler (outliers will remain)
        x = StandardScaler().fit_transform(data)
        
        # -- select fraction or number of Principal componenets and create PCA
        pca = PCA(n_components = pca_comp)
        
        # -- apply PCA
        X = pca.fit_transform(x)
        
    else:
        X = data
        
    # -- create outlier detector
    outlier_detector = LocalOutlierFactor(n_neighbors=neighbours)
    
    # -- apply detector on data
    inliers = outlier_detector.fit_predict(X)
    
    # -- create df with inliers only
    df_outliers_removed = df[inliers==1] # inliers = 1
    
    if plot_PCA == True:
        plt.scatter(X[:, 0], X[:, 1], c = inliers)
        plt.xlabel(r"Principal Component 1"); plt.ylabel(r"Principal Component 2")
        plt.title("Outliers")
        
    return df_outliers_removed



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
    
    # -- Retrieve list containing with dataframes for training and testing for each fold
    indexes_train_kfold = list(cross_validation.split(df_X_train))
    
    r2_fracs = []
    r2_frac_stds = []
    MAE_fracs = []
    MAE_frac_stds = []
    
    # -- For each fraction value...
    for idx_fraction, partial_fit_frac in enumerate([0.4, 1]):
        
        # -- prepare storage lists
        r2_folds = []
        MAE_folds = []
        
        # -- select the fraction of the fold ...
        for idx_fold, fold in enumerate(indexes_train_kfold):
            
            # ... select a fold 
            fold_X_train = df_X_train.iloc[fold[0]]
            fold_X_test = df_X_train.iloc[fold[1]]
            fold_y_train = df_y_train.iloc[fold[0]]
            fold_y_test = df_y_train.iloc[fold[1]]
            
            # ... retrieve indexes belonging to fraction of the fold 
            idx_partial_fit_train = pd.DataFrame(fold_X_train).sample(frac = partial_fit_frac, random_state= 42).index
            idx_partial_fit_test = pd.DataFrame(fold_X_test).sample(frac = partial_fit_frac, random_state= 42).index

            # ... select fraction of fold 
            fold_X_train_frac = fold_X_train.loc[idx_partial_fit_train]
            fold_X_test_frac = fold_X_test.loc[idx_partial_fit_test]
            fold_y_train_frac = fold_y_train.loc[idx_partial_fit_train]
            fold_y_test_frac = fold_y_test.loc[idx_partial_fit_test]
            
            # ... fit to the regressor
            pipeline.fit(fold_X_train_frac, fold_y_train_frac)
            
            # ... make fold prediction
            prediction = pipeline.predict(fold_X_test_frac)
            
            # ... assess fold performance
            MAE_fold = -1* median_absolute_error(fold_y_test_frac, prediction)
            r2_fold = r2_score(fold_y_test_frac, prediction)
            
            # ... store results to assess performance per fraction
            MAE_folds.append(MAE_fold)
            r2_folds.append(r2_fold)
        
        # -- Calculate mean and std results from all folds per fraction of data
        r2_frac = np.mean(r2_folds)
        MAE_frac = np.mean(MAE_folds)
        r2_fracs_std = np.std(r2_folds)
        MAE_fracs_std = np.std(MAE_folds)
        
        # -- Save results
        r2_fracs.append(r2_frac); r2_frac_stds.append(r2_fracs_std); 
        MAE_fracs.append(MAE_frac);  MAE_frac_stds.append(MAE_fracs_std)
        
        # -- Report results to decide wether to prune
        trial.report(MAE_frac, idx_fraction)
        
        # -- Prune the intermediate value if neccessary.
        if trial.should_prune():
            raise optuna.TrialPruned()

    # -- final results are those obtained for last fraction (e.g. fraction of 1/1)
    MAE = MAE_fracs[-1]
    MAE_std = r2_frac_stds[-1]
    r2 = r2_fracs[-1]
    r2_std = r2_frac_stds[-1]
    return MAE, r2, MAE_std, r2_std



def create_objective(study_name, write_path, regressor_class, create_params, X_training, y_training):
    """
    Nested function containing the optuna objective which has to be mini/maximised
    """
    
    def objective(trial):
    
        # save optuna study
        joblib.dump(study, write_path + '/' + study_name + '.pkl')
        
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
            
        # -- Optionally apply PCA in independent variables such that 99% of variance can still be explained
        # pca = PCA(n_components = 0.99)
        
        # -- Make a pipeline
        # pipeline = make_pipeline(scaler, pca, transformed_regressor)
        pipeline = make_pipeline(scaler, transformed_regressor)
    
        # -- Assess model performance using specified cross validation on pipeline with pruning
        MAE, r2, MAE_std, r2_std = model_performance(trial, X_training, y_training, cross_validation, pipeline)
        return MAE
    
    return objective


def scaler_chooser(scaler_str):
    """
    Function outputs a scaler function corresponding to input string
    """
    
    if scaler_str == "minmax":
        scaler = MinMaxScaler()
    elif scaler_str == "standard":
        scaler = StandardScaler()
    elif scaler_str == "robust":
        scaler = RobustScaler()
    else:
        scaler = None
        
    return scaler

def transformer_chooser(transformer_str):
    """
    Function outputs a transformer function corresponding to input string
    """
    
    if transformer_str == "none":
        transformer = None
    elif transformer_str == "quantile":
        transformer = QuantileTransformer(n_quantiles=500, output_distribution="normal")
        
    return transformer



#%% Load data

file_name = 'wdirwspd_11_05_v26_extra_era5'
file = '/home/owen/Documents/data/'+ file_name +  '.csv'
df_output = pd.read_csv(file)
df = df_output.copy()

#%% filter out specific data 

# remove if classification probability is below 50%
# df_val = df.loc[df['prob_1'] >= 0.5]
df_val = df.loc[df['prob'] >= 50]
outliers_prob = len(df) - len(df_val)
print('# outliers probability: ' + str(outliers_prob))

# remove if era5 claims either positive Obukhov length or sensible heat flux is into the ocean rather into the atmosphere (positive sshf)
# df_val = df_val.loc[(df_val['L_era5'] < 0)]
df_val = df_val.loc[(df_val['L_era5'] < 0) & (df_val['L_buoy'] < 0 )]
print('# outliers positive validation L: ' + str(len(df) - outliers_prob - len(df_val)))

# remove if normalised standard deviation is too low (too pefect slope is presumed wrong)
number = len(df_val)
df_val = df_val[df_val.w_star_normalised_deviation >= 0.01]
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

keep_after_index = list(df_val.keys()).index('wspd_median') # wspd_median is first observation column
keep_before_index = list(df_val.keys()).index('S_normalised_deviation') +1 # S_normalised_deviation is the last measured param

# # remove outliers in all observation columns
# def outlier(df):
#     return np.where((df> np.mean(df) + np.std(df)*4) | (df< np.mean(df) - np.std(df)*4))

# T = df_val.reset_index(drop = True).iloc[:,keep_after_index:].apply(outlier, axis = 0).values[0] # find rows with outliers in any of the estimates parameters
# idx_delete = list(set([x for xs in T for x in xs])) # extract
# print('# outliers: ' + str(len(idx_delete)))
# df_val = df_val.reset_index(drop = True).drop(index = idx_delete)

df_val = outlier_detector(df_val, 'window_effect', 'S_normalised_deviation', pca_comp=0.80, neighbours =100, plot_PCA=True)   
df_val = df_val.reset_index(drop = True)  
number3 = len(df_val)
print('# outliers: ' + str(number2 - number3))
    
##########################################################################################
######## pre filter on df, not df_val (such that shape stays identical )#################
##########################################################################################

# add spectral amplitude of wind field to df_val
Beta = 1   ###### !!!!  This value ought changes depending on cells or rolls (1 and 4/3 respectively)
spatial_freq = 1/600
# temperature for estimate is 293 because that was the approximation used, i.e. now the approximation is undone 
df_val['PSD_spatial']  = df_val.progress_apply(lambda x: calc_PSD_spatial( x['B'], Beta,spatial_freq, 293), axis=1)

# remove meta data and era5 but keep lat and lon from image centroid, and CNN prediction prob and PSD spatial
df_obs = df_val.iloc[:,keep_after_index:keep_before_index] 
df_obs['prob'] = df_val['prob'] #### df_obs['prob'] = df_val['prob_1'] 
df_obs['lat_centroid'] = df_val['lat_centroid']
df_obs['lon_centroid'] = df_val['lon_centroid']
# df_obs['prob'] = df_val['prob_1']
# df_obs['lat'] = df_val['lat']
# df_obs['lon'] = df_val['lon']

df_obs['PSD_spatial'] = df_val['PSD_spatial']

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
# L_param = 'L'

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

# ################### # Errors w.r.t buoy #################
df_val['L_rel_error_buoy'] = (df_val[L_param] - df_val.L_buoy) / df_val.L_buoy
df_val['L_rel_error_buoy_nosign'] = abs((df_val[L_param] - df_val.L_buoy) / df_val.L_buoy)
df_val['L_rel_buoy'] = abs(df_val.L_buoy) / abs(df_val[L_param])

# add Obukhov length error classes to validation
df_val['sigma_u_buoy'] = df_val.progress_apply(lambda x: sigma_u_panofsky( x['L_buoy'], x['pblh_era5'], x['usr_era5']), axis=1)   # 'usr_era5' is friction velocity (u*) from coare4 using era5


#%% split data and prepare functions

############ All components #############
X = df_obs
X = pd.concat([
    df_val.iloc[:, list(df_val.keys()).index('tair_era5'):list(df_val.keys()).index('relh_era5')+1],
    df_val.iloc[:, list(df_val.keys()).index('tau_era5'):list(df_val.keys()).index('wdir100_era5')+1],
    df_val.iloc[:, list(df_val.keys()).index('L_era5_only')]],
    axis = 1,
    )  # 

############ Components correlating with error (retrieved by kendall correlation and VIF) #################
# X = df_obs[['incidence_avg', 'var_highpass', 'var_lowpass', 'L', 'S', 'PSD_spatial']].iloc[:, :]

############ adding components from validation ##############
# X = pd.concat([df_obs, df_val['wdir_range_era5']], axis = 1).values

############ Validation, transform optional ############
# y = np.log10(abs(df_val['L_era5_only'].iloc[:]))
y = np.log10(abs(df_val['L_buoy'].iloc[:]))
# y = np.log10(abs(df_val['L_era5_only'].iloc[:])) - np.log10(abs(df_val['L'].iloc[:]))

############ California houseing price test ############
# from sklearn.datasets import fetch_california_housing
# test = fetch_california_housing()
# X = pd.DataFrame(test.data)
# y = pd.DataFrame(test.target)

############ Subsample data ############
# angleBinLow, angleBinHigh = 0, 91
# X = X[(df_val['wdir_range_era5'] < angleBinHigh) & (df_val['wdir_range_era5'] >= angleBinLow)]
# y = y[(df_val['wdir_range_era5'] < angleBinHigh) & (df_val['wdir_range_era5'] >= angleBinLow)]

# X = X.iloc[:1560,:]
# y = y.iloc[:1560]

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2)
X_train, X_test, y_train, y_test, test_index, train_index = eq.splitTrainTest(X, y, 
                                                                              testSize = 0.2, 
                                                                              randomState = 42, 
                                                                              n_splits = 1, 
                                                                              smote = False, 
                                                                              equalSizedClasses = False, 
                                                                              classToIgnore = None, 
                                                                              continuous = True)

# X_train_hyper, X_train_fit, y_train_hyper, y_train_fit, test_index, train_index =  eq.splitTrainTest(pd.DataFrame(X_train), pd.DataFrame(y_train), 
#                                                                                                        testSize = 0.8, 
#                                                                                                        randomState = 42, 
#                                                                                                        n_splits = 1, 
#                                                                                                        smote = False, 
#                                                                                                        equalSizedClasses = False, 
#                                                                                                        classToIgnore = None, 
#                                                                                                        continuous = True)

#%%

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
    # "ridge": (
    #     Ridge,
    #     lambda trial: {
    #         'alpha': trial.suggest_float("alpha", 1e-8, 1e2, log = True),
    #         'solver': trial.suggest_categorical("solver", ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'])
    #     },
    # ),
    # "bayesianridge": (
    #     BayesianRidge,
    #     lambda trial: {
    #         'n_iter': trial.suggest_int("n_iter", 10, 400),
    #         'tol': trial.suggest_float("tol", 1e-8, 1e2),
    #         'alpha_1': trial.suggest_float("alpha_1", 1e-8, 1e2, log = True),
    #         'alpha_2': trial.suggest_float("alpha_2", 1e-8, 1e2, log = True),
    #         'lambda_1': trial.suggest_float("lambda_1", 1e-8, 1e2, log = True),
    #         'lambda_2': trial.suggest_float("lambda_2", 1e-8, 1e2, log = True),
    #     },
    # ),
    # "adaboost": (
    #     AdaBoostRegressor,
    #     lambda trial: {
    #         'n_estimators': trial.suggest_int("n_estimators", 10, 100),
    #         'learning_rate': trial.suggest_float("learning_rate", 1e-2, 1e0, log = True),
    #         'loss': trial.suggest_categorical("loss", ['linear', 'square', 'exponential']),
    #         'random_state': 42,
    #     },
    # ),
    # "gradientboost": (
    #     GradientBoostingRegressor,
    #     lambda trial: {
    #         'n_estimators': trial.suggest_int("n_estimators", 10, 300),
    #         'learning_rate': trial.suggest_float("learning_rate", 1e-2, 1e0, log = True),
    #         'subsample': trial.suggest_float("subsample", 1e-2, 1.0, log = False),
    #         'max_depth': trial.suggest_int("max_depth", 1, 10),
    #         'criterion': trial.suggest_categorical("criterion", ['friedman_mse', 'squared_error']),
    #         'loss': trial.suggest_categorical("loss", ['squared_error', 'absolute_error', 'huber', 'quantile']),
    #         'random_state': 42,
    #     },
    # ),
    # "knn": (
    #     KNeighborsRegressor,
    #     lambda trial: {
    #         'n_neighbors': trial.suggest_int("n_neighbors", 1, 101, step = 5),
    #         'weights': trial.suggest_categorical("weights", ['uniform', 'distance']),
    #     },
    # ),
    # "sgd": (
    #     SGDRegressor,
    #     lambda trial: {
    #         'loss': trial.suggest_categorical("loss", ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
    #         'penalty': trial.suggest_categorical("penalty", ['l2', 'l1', 'elasticnet']),
    #         'alpha': trial.suggest_float("alpha", 1e-8, 1e2, log = True),
    #         'random_state': 42,
    #     },
    # ),
    # "bagging": (
    #     BaggingRegressor,
    #     lambda trial: {
    #         'n_estimators': trial.suggest_int("n_estimators", 1, 101, step = 5),
    #         'max_features': trial.suggest_float("max_features", 1e-1, 1.0, step = 0.1),
    #         'random_state': 42,
    #     },
    # ),
    # "svr": (
    #     LinearSVR,
    #     lambda trial: {
    #         'loss': trial.suggest_categorical("loss", ['epsilon_insensitive', 'squared_epsilon_insensitive']),
    #         'C': trial.suggest_float("C", 1e-5, 1e2, log = True),
    #         'tol': trial.suggest_float("tol", 1e-8, 1e2, log = True),
    #         'random_state': 42,
    #     },
    # ),
    # "elasticnet": (
    #     ElasticNet,
    #     lambda trial: {
    #         'alpha': trial.suggest_float("alpha", 1e-8, 1e2, log = True),
    #         'l1_ratio': trial.suggest_float("l1_ratio", 1e-5, 1.0, log = True),
    #         'random_state': 42,
    #     },
    # ),
    # "lassolars": (
    #     LassoLars,
    #     lambda trial: {
    #         'alpha': trial.suggest_float("alpha", 1e-8, 1e2, log = True),
    #         'normalize': trial.suggest_categorical("normalize", [False]),
    #         'random_state': 42,
    #     },
    # ),
}



timeout = 1200
n_trial = 200
sampler = TPESampler(seed = 42)
# sampler = RandomSampler(seed =  42)
cross_validation = KFold(n_splits = 5, shuffle = False, random_state= None)   #  KFold(n_splits=5)

# according to: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
# the best sampler for TPE is hyperband whereas for random it is median
# pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1) # warmup steps starts at n=0, 
# pruner = optuna.pruners.PercentilePruner(percentile = 25.0, n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3)

base_directory = '/home/owen/Documents/models/optuna/cells/'
# 'buoy_filtOnPca080Neighbours100_randSampMedPrune'  'buoy_filtOnPca080Neighbours100'
# 'buoy_filtOnPca080Neighbours100_RandSampMedPrune5Models' 'california_TPE_hyper'
folder = 'era5_to_buoy2'   

write_path = base_directory + folder

if not os.path.exists(write_path):
    os.makedirs(write_path)

studies = {}
for study_name, (regressor, create_params) in methods.items():
    study = optuna.create_study(direction = 'maximize', sampler=sampler, pruner=pruner)
    study.optimize(create_objective(study_name, write_path, regressor, create_params, 
                                    X_training = X_train, y_training = y_train), n_trials=n_trial, timeout=timeout)
    
    # the study is saved during each trial to update with the previous trial (this stores the data even if the study does not complete)
    # here the study is saved once more to include the final iteration
    joblib.dump(study, write_path + '/' + study_name + '.pkl')
    studies[study_name] = study


#%% retrieve regressors from stored location

regressor_names = list(methods.keys())
# directory = '/home/owen/Documents/models/optuna/rolls/era5_filtOnPca080Neighbours100/'
# directory = '/home/owen/Documents/models/optuna/rolls/buoy_filtOnPca080Neighbours100_randSampMedPrune/'
# directory = '/home/owen/Documents/models/optuna/rolls/buoy_filtOnPca080Neighbours100_TPESampHyperPrune5Models/'
# directory = '/home/owen/Documents/models/optuna/rolls/buoy_filtOnPca080Neighbours100_RandSampMedPrune5Models/'
# directory = '/home/owen/Documents/models/optuna/rolls/buoy_filtOnPca080Neighbours100_TPESampHyperPrune5Models_TrainSplitHyperFit/'
# directory = '/home/owen/Documents/models/optuna/cells/buoy_filtOnPca080Neighbours100_TPESampHyperPrune5Models/'
# directory = '/home/owen/Documents/models/optuna/cells/buoy_filtOnPca080Neighbours100_TPESampHyperPrune5Models_2fracs/'
# directory = '/home/owen/Documents/models/optuna/cells/era5_wdirwspd24/'
# directory = '/home/owen/Documents/models/optuna/cells/era5_wdirwspd24_PCA99/'
# directory = '/home/owen/Documents/models/optuna/rolls/era5_wdirwspd27/'
# directory = '/home/owen/Documents/models/optuna/rolls/era5_to_buoy/'
# directory = '/home/owen/Documents/models/optuna/rolls/era5_to_buoy2/'
# directory = '/home/owen/Documents/models/optuna/rolls/era5_to_buoy2_subsample/'
# directory = '/home/owen/Documents/models/optuna/cells/era5_to_buoy/'
directory = '/home/owen/Documents/models/optuna/rolls/era5_to_buoy2/'

# directory = '/home/owen/Documents/models/optuna/cells/era5_filtOnPca080Neighbours100_TPESampHyperPrune5Models_2fracs/'

# directory = '/home/owen/Documents/models/optuna/rolls/california_TPE_hyper/'


estimators = []
for regressor_name in regressor_names:
    file_name = glob.glob(directory + regressor_name + '*.pkl')[0]

    study = joblib.load(file_name)
    
    list_params = list(study.best_params)
    list_params.remove('scalers'); list_params.remove('transformers')
    
    parameter_dict = {k: study.best_params[k] for k in study.best_params.keys() & set(list_params)}
    
    pipe_single_study = Pipeline([
        ('scaler', scaler_chooser(study.best_params['scalers'])),
        # ('pca', PCA(n_components = 0.99)),
        ('model', TransformedTargetRegressor(
                regressor = methods[regressor_name][0](**parameter_dict), 
                transformer= transformer_chooser(study.best_params['transformers'])
                ))]
        )
    estimators.append((regressor_name, pipe_single_study))
    
##########################################################################
### plot estimated Obukhov lengths before and after stacked correction ###
##########################################################################

X_fit = X_train # X_train   X_train_fit
y_fit = y_train # y_train    y_train_fit

stacking_regressor = StackingRegressor(estimators= estimators, final_estimator=Ridge(), cv = 5)
stacking_regressor.fit(X_fit, y_fit)

y_pred_stacked = stacking_regressor.predict(X_test)
y_pred_train_stacked = stacking_regressor.predict(X_fit)

print(r2_score(y_test, y_pred_stacked))
print(median_absolute_error(y_test, y_pred_stacked))
print(r2_score(y_fit, y_pred_train_stacked))
print(median_absolute_error(y_fit, y_pred_train_stacked))

# plt.scatter(y_test, y_pred_stacked, alpha = 0.5, s=1, c = 'r'); plt.plot([0.5, 3], [0.5, 3], '--k')
# plt.scatter(y_test, np.log10(abs(X_test[:, list(X.keys()).index('L_estimate')])), alpha = 0.5, s=1, c = 'k'); plt.plot([0.5, 3], [0.5, 3], '--k')
    
#%%

########################################################################
### calculate performance per regressor and stacked, including std's ###
########################################################################

r2_train = []
r2_test = []
r2_test_std = []
MAE_train = []
MAE_test = []
MAE_test_std = []
kf = KFold(n_splits = 5, shuffle = False, random_state= None)

for i, regressor in enumerate(regressor_names + ['stacked']):

    estimator_temp = estimators[i:i+1]
    
    if i == len(estimators):
        estimator_temp = estimators
    
        regressor_final = StackingRegressor(estimators=estimator_temp, final_estimator=Ridge(), cv = 5)
        regressor_final.fit(X_fit, y_fit)
        
    else:
        regressor_final = estimator_temp[0][1]
        regressor_final.fit(X_fit, y_fit)
    
    # -- create dataframe just to retrieve fold indexes from K-fold validation for TEST data
    df_X_test = pd.DataFrame(X_test)
    idexes_trest_kfold = list(kf.split(df_X_test))
    
    # -- prepare storage of per-fold prediction
    r2_fold = []
    MAE_fold= []
    
    # -- For each TEST data fold...
    for idx_fold, fold in enumerate(idexes_trest_kfold):
        
        # -- Select the fold indexes        
        fold_test = fold[1]

        # -- Predict on the TEST data fold
        prediction = regressor_final.predict(X_test[fold_test, :])
    
        # -- Assess prediction
        intermediate_value_MAE = median_absolute_error(y_test[fold_test], prediction)
        intermediate_value_r2 = r2_score(y_test[fold_test], prediction)
        
        # -- Store prediction
        r2_fold.append(intermediate_value_r2)
        MAE_fold.append(intermediate_value_MAE)
        
    
    # -- calculate means and std's per fold
    r2_test.append(np.mean(r2_fold))
    r2_test_std.append(np.std(r2_fold))
    MAE_test.append(abs(np.mean(MAE_fold)))
    MAE_test_std.append(abs(np.std(MAE_fold)))
 

######################################################
### plot performance per regressor and for stacked ###
######################################################    

fig,ax = plt.subplots(figsize=(8,5))
nr_ticks = len(regressor_names + ['stacked (ridge)'])
x_tick_loc = np.linspace(1, nr_ticks, nr_ticks)
x_tick_label = regressor_names + ['stacked (ridge)']


plt_err = ax.errorbar(x_tick_loc, r2_test, yerr = r2_test_std, linestyle = '-', marker="o", c = 'k', barsabove = True, capsize = 5)
plt_err[-1][0].set_linestyle('--')
# ax.set_xlabel("ERA5 wind direction in range", fontsize = 14)
ax.set_ylabel(r"$R^2$", color="k",fontsize=14)
ax.set_xticks(x_tick_loc, labels = x_tick_label, rotation = 60)
ax.set_ylim([0.2, 0.80])
ax.grid(axis='y')
ax2=ax.twinx()
plt_err = ax2.errorbar(x_tick_loc, np.array(MAE_test), yerr = np.array(MAE_test_std), linestyle = '-', marker="o", c = 'r', barsabove = True, capsize = 5)
plt_err[-1][0].set_linestyle('--')
ax2.set_ylim([0.10, 0.25])
ax2.set_ylabel("Median Absolute Error",color="r",fontsize=14)
ax2.set_title(r"Regressor performance on $\mathrm{log_{10}}(|\mathrm{L}|)$ from $\mathrm{log_{10}}(|\mathrm{L_{\mathrm{ERA5}}}|)$ for Cells (5-Fold)")
# ax2.grid()

plt.show()
    
######################
### plot envelopes ###
######################

# L_param = 'L_estimate'
# L_param = 'L'
L_param = 'L_era5'

df_plot = pd.DataFrame()
df_plot['y_test'] = 10**y_test
df_plot['y_pred'] = abs(X_test[:, list(X.keys()).index(L_param)])
df_plot['y_ML'] = 10**y_pred_stacked

hist_steps = 30
title = 'Correction using stacked regression from ERA5 $L$ (Cells)'
x_axis_title = r"|Obukhov length| validation (ERA5)"
_ = eq.plot_envelope(df_plot, hist_steps, title = title, x_axis_title = x_axis_title, alpha = 0.5)


# L_val = 'L_era5'
L_val = 'L_buoy'

r2_score(np.log10(abs(df_val[L_val])), np.log10(abs(df_val[L_param]))) 
mean_absolute_error(np.log10(abs(df_val[L_val])), np.log10(abs(df_val[L_param]))) 


#%% plot SHAP

model = methods[regressor_name][0](**parameter_dict)
model.fit(X_fit, y_fit)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_fit)

shap.summary_plot(shap_values, X_fit, feature_names = list(X.keys()))


#%% plot results per buoy

df_test = df_val.iloc[test_index]
df_test['error_rem'] = y_test - y_pred_stacked

df_grouped = df_test.groupby(df_test['buoy'])['error_rem']
func = [(r'$\overline{(U_1 - U_2)^2}$', lambda x: np.median(abs(x)))]
func2 = [(r'$Count$',lambda x: x.count())]

fig = plt.figure( figsize = (20, 6)) # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.4

df_grouped.agg(func).plot(kind = 'bar', ax = ax, stacked=False, width=width, color = 'red', position=1, rot=90, grid = True)
df_grouped.count().plot(kind = 'bar', ax = ax2, stacked=False, width=width, color = 'black', position=0, rot=90, grid = True)

ax.set_ylabel('Median Aboslute Error')
ax2.set_ylabel('Observation count')

plt.show()

