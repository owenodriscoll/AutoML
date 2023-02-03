


import warnings
import optuna
import joblib
import os, sys
import pandas as pd
import numpy as np
from typing import Callable, Union
from sklearn.metrics import median_absolute_error, r2_score

from optuna.samplers import TPESampler #, RandomSampler
from sklearn.model_selection import KFold
from sklearn.compose import TransformedTargetRegressor

from sklearn.pipeline import Pipeline

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from AutoML.AutoML.regressors_ import regressor_selector


def warning_catcher(f):
    def wrap_arguments(args):
        warnings.simplefilter(args.warning_verbosity, UserWarning)
        old_stdout = sys.stdout
        if args.warning_verbosity == 'ignore':
            sys.stdout = open(os.devnull, "w")
        else:
            sys.stdout = old_stdout

        f(args)

        warnings.simplefilter('default', UserWarning)
        sys.stdout = old_stdout

    return wrap_arguments

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
def pca_chooser(trial=None, **kwargs):
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
def poly_chooser(trial=None, **kwargs):
    """
    Function to transform input variables using polynomial features
    """

    if kwargs.get('poly_value') is not None:
        from sklearn.preprocessing import PolynomialFeatures
        type_ref = type(kwargs.get('poly_value'))

        if type_ref is dict:
            poly = PolynomialFeatures(**kwargs.get('poly_value'))
        elif type_ref is int or type_ref is float:
            poly = PolynomialFeatures(degree=kwargs['poly_value'])
        if trial != None:
            trial.suggest_categorical('poly_value', [poly.get_params()])
    else:
        poly = None
        if trial != None:
            trial.suggest_categorical('poly_value', [None])

    return poly
def spline_chooser(feature_combo=False, trial=None, **kwargs):
    """
    Function to transform input variables using spline features
    """

    if (kwargs.get('spline_value') is not None):  # & (feature_combo == True):
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





class AutomatedRegression:
    def __init__(self, y: pd.DataFrame, X: pd.DataFrame, test_frac: float=0.2, timeout: int = 600, n_trial: int = 100, cross_validation: callable = None,
                 sampler: callable = None, pruner: callable = None, poly_value: Union[int, float, dict] = None, spline_value: Union[int, float, dict] = None,
                 pca_value: Union[int, float, dict] = None, metric_optimise: Callable = median_absolute_error,
                 metric_assess: list[Callable]=[median_absolute_error, r2_score],
                 optimisation_direction: str='maximize', write_folder: str=os.getcwd() + '/auto_regression/', overwrite: bool = False,
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
        self.cross_validation = cross_validation if 'split' in dir(cross_validation) else KFold(n_splits=5, shuffle=True, random_state=random_state)
        self.sampler = sampler if 'optuna.samplers' in type(sampler).__module__ else TPESampler(seed=random_state)
        self.pruner = pruner if 'optuna.pruners' in type(pruner).__module__ else optuna.pruners.HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3)
        self.poly_value = poly_value
        self.spline_value = spline_value
        self.pca_value = pca_value
        self.metric_optimise = metric_optimise
        self.metric_assess = metric_assess
        self.optimisation_direction = optimisation_direction
        self.write_folder = write_folder
        self.overwrite = overwrite
        self.list_regressors_optimise = list_regressors_optimise
        self.list_regressors_assess = list_regressors_optimise if list_regressors_assess == None else list_regressors_assess
        self.fit_frac = fit_frac
        self.random_state = random_state
        self.regressors_2_optimise = regressor_selector(regressor_names=self.list_regressors_optimise,
                                                      random_state=self.random_state)
        self.regressors_2_assess = regressor_selector(regressor_names=self.list_regressors_optimise,
                                                    random_state=self.random_state)
        self.warning_verbosity = warning_verbosity
        self.create_dir()

    # turn list of regressors into dictionary containing regressor functions

    def create_dir(self):
        if not os.path.exists(self.write_folder):
            os.makedirs(self.write_folder)
            print('would have made a new directory now')

    def split_train_test(self, shuffle: bool = True):
        from sklearn.model_selection import train_test_split
        df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(self.X, self.y, test_size=self.test_frac,
                                                                        random_state=self.random_state, shuffle=shuffle)

        self.X_train = df_X_train.values
        self.X_test = df_X_test.values
        self.y_train = np.ravel(df_y_train.values)
        self.y_test = np.ravel(df_y_test.values)
        self.train_index = df_X_train.index.values
        self.test_index = df_X_test.index.values
        return self

    #@warning_catcher
    def regressor_optimise(self):
        """
        Function performs the optuna optimisation for filtered list of methods (methods_filt) on training data
        """
        def _optimise():

            # -- if catboost is loaded prepare special catch for common catboost errors
            if 'catboost' in list(self.regressors_2_optimise.keys()):
                import catboost
                catch = (catboost.CatBoostError,)
            else:
                catch = ( )

            for regressor_name, (regressor, create_params) in self.regressors_2_optimise.items():
                study = optuna.create_study(direction=self.optimisation_direction, sampler=self.sampler, pruner=self.pruner)
                self._study = study
                self._create_params = create_params
                self._regressor = regressor
                self._regressor_name = regressor_name
                self._write_file = self.write_folder + regressor_name + '.pkl'

                # -- if regressor already trained, throw warning unless overwrite  == True
                if os.path.isfile(self._write_file):
                    if self.overwrite != True:
                        message = "Regressor already exists in directory but overwrite set to 'False'. Regressor skipped."
                        print(len(message) * '_' + '\n' + message + '\n' + len(message) * '_')
                        continue
                    if self.overwrite == True:
                        message = "Regressor already exists in directory. Overwrite set to 'TRUE'"
                        print(len(message) * '_' + '\n' + message + '\n' + len(message) * '_')

                self._study.optimize(_create_objective(),
                               n_trials=self.n_trial, timeout=self.timeout, catch=catch)
                #
                # # -- save final study iteration
                joblib.dump(self._study, self._write_file)
            return

        def _create_objective():
            def _objective(trial):
                self._trial = trial

                # save optuna study
                joblib.dump(self._study, self._write_file)

                # -- Instantiate scaler for independents
                #scalers = trial.suggest_categorical("scalers", [None, 'minmax', 'standard', 'robust'])
                scaler = None #scaler_chooser(scalers)

                # -- determine if requested feature combinations improve results
                # -- only suggest this to trial of kwargs contain at least one of the relevant parameters
                if any([bool(i) for i in [self.spline_value, self.poly_value]]):

                    # -- suggest either to include feature combination or not
                    feature_combo = trial.suggest_categorical("feature_combo", [False, True])

                    # -- if trial will try using feature combinations/compression
                    if feature_combo == True:

                        # -- instantiate spline transformer if relevant kwargs included
                        spline = None

                        # -- instantiate polynomial transformer if relevant kwargs included
                        poly = None
                    else:
                        spline = poly = None

                    # -- instantiate pca compression if relevant kwargs included
                else:
                    spline = poly = None

                pca = None

                # -- Instantiate transformer for dependents
                # transformers = trial.suggest_categorical("transformers", ['none', 'quantile_trans'])
                transformer = None

                # -- Tune estimator algorithm
                param = self._create_params(trial)

                # -- Create regressor
                regressor = self._regressor()
                regressor.set_params(**param)

                # -- Create transformed regressor
                transformed_regressor = TransformedTargetRegressor(
                    regressor= regressor,
                    transformer=transformer
                )

                # -- Make a pipeline
                self._pipeline = pipeline = Pipeline([('poly', poly), ('spline', spline), ('scaler', scaler), ('pca', pca),
                                     ('regressor', transformed_regressor)])

                # -- Assess model performance using specified cross validation on pipeline with pruning
                result = _model_performance()
                # result = 0.15
                return result

            return _objective

        def _model_performance():
            """
            function for splitting, training, assessing and pruning the regressor
            1. First the data is split into K-folds.
            2. Iteratively an increasing fraction of the training and tes folds and test fold is taken
            3. The regressor is trained and assessed iteratively
            4. If performance is for first iterations is poor, regressor is pruned thus preventing training and testing on full dataset

            """
            # -- turn train and test arrays into temporary dataframes
            df_X_train = pd.DataFrame(self.X_train)
            df_y_train = pd.DataFrame(self.y_train)

            # -- Retrieve list containing with dataframes for training and testing for each fold
            indexes_train_kfold = list(self.cross_validation.split(df_X_train))

            result_folds_fracs = []
            result_folds_stds = []

            # -- For each fraction value...
            for idx_fraction, partial_fit_frac in enumerate(self.fit_frac):

                # -- when too few samples are available for assessment (less than 20 are used as the test fraction --> prun)
                min_samples = int(np.ceil(len(self.X_train) * partial_fit_frac * (1) / self.cross_validation.n_splits))
                if min_samples < 20:
                    # !!! add error condition/print when partial_fit_Frac = 1 while samples are still too low
                    continue

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
                    idx_partial_fit_train = pd.DataFrame(fold_X_train).sample(frac=partial_fit_frac,
                                                                              random_state=self.random_state).index
                    idx_partial_fit_test = pd.DataFrame(fold_X_test).sample(frac=partial_fit_frac,
                                                                            random_state=self.random_state).index

                    # ... select fraction of fold
                    fold_X_train_frac = fold_X_train.loc[idx_partial_fit_train]
                    fold_X_test_frac = fold_X_test.loc[idx_partial_fit_test]
                    fold_y_train_frac = fold_y_train.loc[idx_partial_fit_train]
                    fold_y_test_frac = fold_y_test.loc[idx_partial_fit_test]

                    # -- determine if regressor is lightgbm boosted model
                    regressor_is_boosted = bool(
                        set([self._regressor_name]) & set(['lightgbm']))  # xgboost and catboost ignored, bugs  out

                    # -- fit training data and add early stopping function if X-iterations did not improve data
                    # ... if regressor is boosted ...
                    if regressor_is_boosted:

                        # -- fit transformers to training fold of training data
                        fold_X_train_frac_transformed = self._pipeline[:-1].fit_transform(fold_X_train_frac)

                        # -- transform testting fold of training data
                        fold_X_test_frac_transformed = self._pipeline[:-1].transform(fold_X_test_frac)

                        # fit pipeline using pre-fitted transformers
                        self._pipeline.fit(fold_X_train_frac_transformed, fold_y_train_frac,
                                     regressor__eval_set=[(fold_X_test_frac_transformed, fold_y_test_frac)],
                                     regressor__early_stopping_rounds=20)

                    # ... if regressor is NOT boosted ...
                    else:
                        # -- fit training data
                        self._pipeline.fit(fold_X_train_frac, fold_y_train_frac)

                    # ... assess fold performance, sometimes performance is so poor a value error is thrown, therefore insert in 'try' function and return nan's for errors
                    try:
                        # ... if regressor is boosted ...
                        if regressor_is_boosted:
                            # ... make fold prediction on transformed test fraction of training dataset
                            prediction = self._pipeline.predict(fold_X_test_frac_transformed)
                        else:
                            # ... make fold prediction on original test fraction of training dataset
                            prediction = self._pipeline.predict(fold_X_test_frac)

                            # ... assess prediction with chosen metric
                        result_fold = self.metric_optimise(fold_y_test_frac, prediction)
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
                result_folds_fracs.append(result_folds_frac);
                result_folds_stds.append(result_folds_std);

                # -- only prune if not applied on fraction containing all datapoints
                if partial_fit_frac < 1.0:

                    # -- Report results to decide wether to prune
                    self._trial.report(result_folds_frac, idx_fraction)

                    # -- Prune the intermediate value if neccessary.
                    if self._trial.should_prune():
                        raise optuna.TrialPruned()

            # -- final results are those obtained for last fraction (e.g. fraction of 1/1)
            return result_folds_fracs[-1]

        if bool(self.regressors_2_optimise):
            _optimise()
            return self





# test = AutomatedRegression(y=pd.DataFrame([1,2,3,4]), X=pd.DataFrame([1,2,3,4]))
# AutomatedRegression(y=pd.DataFrame([1,2,3,4]), X=pd.DataFrame([1,2,3,4])).split_train_test().regressor_optimise()
# test = AutomatedRegression(y=pd.DataFrame([1,2,3,4]), X=pd.DataFrame([1,2,3,4])).split_train_test().regressor_optimise()
# test._write_file

from sklearn.datasets import make_regression
X, y = make_regression(n_samples=2000, n_features=10, n_informative=5)

# test2 = AutomatedRegression(y=pd.DataFrame(y), X=pd.DataFrame(X), pca_value=None, spline_value=None, poly_value=None, n_trial=2, overwrite=True)
# test2.split_train_test().regressor_optimise()

