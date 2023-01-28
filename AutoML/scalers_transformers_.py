#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:11:20 2023

@author: owen
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer


class FuncHelper:
    @staticmethod
    def run_with_argument(func, arg):

        if isinstance(arg, (int, float)):
            return func(arg)
        if isinstance(arg, dict):
            return func(**arg)
        elif isinstance(arg, type(None)):
            return None
        else:
            print("Input argument not of type: int, float or dict")


def input_type_fit(f):
    def wrapper(args):

        assert all([variable in dir(args) for variable in ['func', 'arg']]), "Required variables not provided to class"

        if isinstance(args.arg, (int, float)):
            return args.func(args.arg)
        if isinstance(args.arg, dict):
            return args.func(**args.arg)
        elif isinstance(args.arg, type(None)):
            return None
        else:
            print("Input argument not of type: int, float or dict")

    return wrapper


def input_type_report(f):
    def wrapper(args):

        assert 'func_fit' in dir(args), "Required variable not provided to class"
        if isinstance(args.func_fit, type(None)):
            return None
        else:
            return f

    return wrapper


class chooser:
    def __init__(self, arg: any, func: callable, transformer, trial=None):  # change to args
        self.func_fitted = None
        self.arg = arg
        self.func = func
        self.transformer = transformer
        self.trial = trial

    def fit(self):
        self.func_fitted = FuncHelper.run_with_argument(self.func, self.arg)

    @input_type_report
    def _report_trial(self):
        # return self.trial.suggest_categorical(self.transformer, [self.func_fit.get_params()])
        return print([self.func_fitted.get_params()])

    def fit_report_trial(self) -> callable:
        self.fit()
        self._report_trial()
        return self.func_fitted


class pcaChooser(chooser):

    def __init__(self, arg, trial=None):  # change to args
        super().__init__(arg=arg, func=PCA, transformer='pca_value')  # change to args


class polyChooser(chooser):

    def __init__(self, arg, trial=None):
        super().__init__(arg=arg, func=PolynomialFeatures, transformer='poly_value')


class splineChooser(chooser):

    def __init__(self, arg, trial=None):
        super().__init__(arg=arg, func=SplineTransformer, transformer='spline_value')


# %%

pca = pcaChooser(3)
pca.fit()
pca._report_trial()
test = pcaChooser(3)

spline = splineChooser(3).fit_report_trial()

spline = splineChooser(None).fit_report_trial()


# %%


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


def transformer_chooser(transformer_str, trial=None, n_quantiles=500, random_state=42):
    """
    Function outputs a transformer function corresponding to input string
    """

    from sklearn.preprocessing import QuantileTransformer

    if transformer_str == "none":
        return None
    elif transformer_str == "quantile_trans":

        # -- if optuna trial is provided to function determine optimal number of quantiles
        if trial != None:
            n_quantiles = trial.suggest_int('n_quantiles', 100, 4000, step=100)

        return QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", random_state=random_state)
