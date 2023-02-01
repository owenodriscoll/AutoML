#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 16:11:20 2023

@author: owen
"""
from typing import Optional, Any

from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer


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


def decorator_report(variable):
    def wrap_function(f):
        def wrap_arguments(args):
            if isinstance(getattr(args, variable), type(None)):
                return None
            else:
                return f(args)

        return wrap_arguments

    return wrap_function


class Chooser:
    def __init__(self, arg: any, func: callable, transformer: str, trial=None):  # change to args
        self.arg = arg
        self.func = func
        self.transformer = transformer
        self.trial = trial
        self.func_fitted = None

    def fit(self):
        self.func_fitted = FuncHelper.run_with_argument(self.func, self.arg)

    @decorator_report("func_fitted")
    def _report_trial(self):
        # self.trial.suggest_categorical(self.transformer, [self.func_fit.get_params()])
        print(self.func_fitted.get_params())
        pass

    def fit_report_trial(self) -> callable:
        self.fit()
        self._report_trial()
        return self.func_fitted


class PcaChooser(Chooser):
    def __init__(self, arg, trial=None):  # change to args
        super().__init__(arg=arg, func=PCA, transformer='pca_value', trial=trial)  # change to args


class PolyChooser(Chooser):
    def __init__(self, arg, trial=None):
        super().__init__(arg=arg, func=PolynomialFeatures, transformer='poly_value', trial=trial)


class SplineChooser(Chooser):
    def __init__(self, arg, trial=None):
        super().__init__(arg=arg, func=SplineTransformer, transformer='spline_value', trial=trial)


pca = PcaChooser(3)
pca.fit()
pca._report_trial()
PcaChooser(3).fit_report_trial()
test = PcaChooser(3)
spline = SplineChooser(3).fit_report_trial()
spline_2 = SplineChooser(None).fit_report_trial()


class ScalerChooser(Chooser):
    def __init__(self, arg: str = '', transformer: str = 'scaler', trial=None):
        self.arg = arg
        self.transformer = transformer
        self.trial = trial
        self.func = None

    def suggest_trial(self):
        # self.arg = self.trial.suggest_categorical(self.transformer, [None, "minmax", "standard", "robust"])
        self.arg = "minmax"
        return self

    def string_to_func(self):
        if self.arg == "minmax":
            self.func = MinMaxScaler
        elif self.arg == "standard":
            self.func = StandardScaler
        elif self.arg == "robust":
            self.func = RobustScaler
        return self

# ScalerChooser().suggest_trial().string_to_func().func()

class TransformerChooser(Chooser):

    def __init__(self, trial=None, random_state: int = 42):
        self.random_state = random_state
        self.trial = trial
        self.func = QuantileTransformer



    def suggest_trial(self):
        #transform_type = trial.suggest_categorical("transformers", ['none', 'quantile_trans'])
        transform_type = 'quantile_trans'
        if not transform_type == 'none':
            # self.n_quantiles = trial.suggest_int('n_quantiles', 100, 4000, step=100)
            self.n_quantiles = 400
            #self.func = QuantileTransformer(n_quantiles=self.n_quantiles, output_distribution="normal", random_state=self.random_state)
        return self

    def fit(self):
        self.func_fitted = FuncHelper.run_with_argument(self.func.set_params, self.n_quantiles)
        return self



def transformer_chooser(transformer_str, trial=None, n_quantiles=500, random_state=42):
    """
    Function outputs a transformer function corresponding to input string
    """

    from sklearn.preprocessing import QuantileTransformer

    if transformer_str == "none":
        return None
    elif transformer_str == "quantile_trans":

        # -- if optuna trial is provided to function determine optimal number of quantiles
        if not trial is None:
            n_quantiles = trial.suggest_int('n_quantiles', 100, 4000, step=100)

        return QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", random_state=random_state)
