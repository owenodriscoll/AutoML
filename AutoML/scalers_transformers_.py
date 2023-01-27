#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:11:20 2023

@author: owen
"""

class chooser:
    
    def __init__(self, arg, *, trial = None):
        self.arg = arg
        self.trial = trial
        
    # @type_decorrator           # first decorate the getter method
    # def get_type(self): # This getter method name is *the* name
    #     return type(arg)
        
        
class scalerChooser(chooser):
    
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    
    options = {"minmax": MinMaxScaler(), "standard": StandardScaler(), "robust": RobustScaler()}
    
    return None
    
        
        
test = chooser('str')   
def type_decorrator(func):
    


  
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

