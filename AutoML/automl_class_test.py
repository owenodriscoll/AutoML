

import pandas as pd
from typing import Callable, Union
import os
from sklearn.metrics import median_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler #, RandomSampler
from sklearn.model_selection import KFold
from sklearn.compose import TransformedTargetRegressor

from sklearn.pipeline import Pipeline

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

from regressors_ import regressor_selector


class AutomatedRegression:
    def __init__(self, y: pd.DataFrame, X: pd.DataFrame, test_frac: float=0.2, timeout: int = 600, n_trial: int = 100, cross_validation: callable = None,
                 sampler: callable = None, poly_value: Union[int, float, dict] = None, spline_value: Union[int, float, dict]=None,
                 pca_value: Union[int, float, dict] =None, metric_optimise: Callable= median_absolute_error,
                 metric_assess: list[Callable]=[median_absolute_error, r2_score],
                 optimisation_direction: str='maximize', write_folder: str=os.getcwd() + '/auto_regression/', overwrite: bool=False,
                 list_regressors_optimise: list[str]=['lightgbm', 'xgboost', 'catboost', 'bayesianridge', 'lassolars'],
                 list_regressors_assess: list[str]=None, fit_frac: list[float]=[0.1, 0.2, 0.3, 0.4, 0.6, 1],
                 random_state: int=42, warning_verbosity: str='ignore'):
        """


        """
        self.y = y
        self.X = X
        self.test_frac = test_frac
        self.timeout = timeout
        self.n_trial = n_trial
        self.cross_validation = cross_validation
        self.sampler = sampler
        self.poly_value = poly_value
        self.spline_value = spline_value
        self.pca_value = pca_value
        self.metric_optimise = metric_optimise
        self.metric_assess = metric_assess
        self.optimisation_direction = optimisation_direction
        self.write_folder = write_folder
        self.overwrite = overwrite
        self.list_regressors_optimise = list_regressors_optimise
        self.list_regressors_assess = list_regressors_assess
        self.fit_frac = fit_frac
        self.random_state = random_state
        self.warning_verbosity = warning_verbosity

        def splitTrainTest(self, n_splits=1):
            from sklearn.model_selection import ShuffleSplit


            y_val = pd.DataFrame(self.y.values, columns=['val'])
            # input is dataframe so convert to arrays
            y_val = np.ravel(y_val.reset_index(drop=True).values)
            x_data = self.X.reset_index(drop=True).values

            sss = ShuffleSplit(n_splits=n_splits, test_size=self.testSize, random_state=self.random_state)

            for train_index, test_index in sss.split(x_data, y_validation):
                x_train, x_test = x_data[train_index], x_data[test_index]
                y_train, y_test = y_validation[train_index], y_validation[test_index]




