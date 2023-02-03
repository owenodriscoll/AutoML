


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
from AutoML.AutoML._regressors import regressor_selector
from AutoML.AutoML._scalers_transformers import PcaChooser, PolyChooser, SplineChooser, ScalerChooser, TransformerChooser

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

class AutomatedRegression:
    def __init__(self, y: pd.DataFrame, X: pd.DataFrame, test_frac: float = 0.2, timeout: int = 600, n_trial: int = 100, cross_validation: callable = None,
                 sampler: callable = None, pruner: callable = None, poly_value: Union[int, float, dict] = None, spline_value: Union[int, float, dict] = None,
                 pca_value: Union[int, float, dict] = None, metric_optimise: Callable = median_absolute_error,
                 metric_assess: list[Callable] = [median_absolute_error, r2_score],
                 optimisation_direction: str = 'minimize', write_folder: str = os.getcwd() + '/auto_regression/', overwrite: bool = False,
                 list_regressors_optimise: list[str] = ['lightgbm', 'xgboost', 'catboost', 'bayesianridge', 'lassolars'],
                 list_regressors_assess: list[str] = None, fit_frac: list[float] = [0.1, 0.2, 0.3, 0.4, 0.6, 1],
                 random_state: int = 42, warning_verbosity: str = 'ignore'):
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

    def create_dir(self):
        if not os.path.exists(self.write_folder):
            os.makedirs(self.write_folder)

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

    @warning_catcher
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

                # -- save final study iteration
                joblib.dump(self._study, self._write_file)
            return


        def _create_objective():
            def _objective(trial):
                self._trial = trial

                # save optuna study
                joblib.dump(self._study, self._write_file)

                # -- Instantiate scaler for independents
                scaler = ScalerChooser(trial=trial).suggest_fit() #None

                # -- determine if requested feature combinations improve results
                # -- only suggest this to trial of kwargs contain at least one of the relevant parameters
                optionals_included = any([bool(i) for i in [self.spline_value, self.poly_value]])
                feature_combo = trial.suggest_categorical("feature_combo", [False, True])
                if all([optionals_included, feature_combo]):
                    # -- if trial will try using spline or polynomial expansion
                    if feature_combo == True:
                        spline = SplineChooser(spline_value=self.spline_value, trial=trial).fit_report_trial()
                        poly = PolyChooser(poly_value=self.poly_value, trial=trial).fit_report_trial()
                else:
                    spline = SplineChooser(spline_value=None, trial=trial).fit_report_trial()
                    poly = PolyChooser(poly_value=None, trial=trial).fit_report_trial()

                pca = PcaChooser(pca_value=self.pca_value, trial=trial).fit_report_trial()

                # -- Instantiate transformer for dependents
                transformer = TransformerChooser(random_state=self.random_state, trial=trial).suggest_and_fit()

                # -- Tune estimator algorithm
                param = self._create_params(trial)

                # -- Create regressor
                regressor = self._regressor()
                regressor.set_params(**param)

                # -- Create transformed regressor
                transformed_regressor = TransformedTargetRegressor(
                    regressor=regressor,
                    transformer=transformer
                )

                # -- Make a pipeline
                self._pipeline = Pipeline([('poly', poly), ('spline', spline), ('scaler', scaler), ('pca', pca),
                                     ('regressor', transformed_regressor)])

                return _model_performance()

            return _objective

        def _model_performance() -> float:
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

    def regressor_fit(self):

        """

        """

        estimators = []
        parameter_dict_dict = {}
        for regressor_name in self.list_regressors_optimise:
            study = joblib.load(self.write_folder + regressor_name + '.pkl')
            self._study = study

            spline = SplineChooser(spline_value=study.best_params.get('spline_value')).fit()
            poly = PolyChooser(poly_value=study.best_params.get('poly_value')).fit()
            pca = PcaChooser(pca_value=study.best_params.get('pca_value')).fit()
            scaler = ScalerChooser(arg=study.best_params.get('scaler')).string_to_func()
            transformer = TransformerChooser(study.best_params.get('n_quantiles'), self.random_state).fit()

            list_params = list(study.best_params)
            list_params_NOT_regressor = ['scaler', 'pca_value', 'spline_value', 'poly_value', 'feature_combo', 'transformers', 'n_quantiles']
            list_params_regressor = set(list_params).difference(set(list_params_NOT_regressor))

            parameter_dict = {k: study.best_params[k] for k in study.best_params.keys() & set(list_params_regressor)}
            parameter_dict_dict[regressor_name] = parameter_dict

            # study must contain parameters "scalers", "transformers", "n_quantiles"
            pipe_single_study = Pipeline([
                ('poly', poly),
                ('spline', spline),
                ('scaler', scaler),
                ('pca', pca),
                ('model', TransformedTargetRegressor(
                    # index 0 is the regressor, index 1 is hyperoptimization function
                    regressor=self.regressors_2_assess[regressor_name][0](**parameter_dict),
                    transformer=transformer
                ))]
            )
            estimators.append((regressor_name, pipe_single_study))
        self.estimators = estimators

        return self


# test = AutomatedRegression(y=pd.DataFrame([1,2,3,4]), X=pd.DataFrame([1,2,3,4]))
# AutomatedRegression(y=pd.DataFrame([1,2,3,4]), X=pd.DataFrame([1,2,3,4])).split_train_test().regressor_optimise()
# test = AutomatedRegression(y=pd.DataFrame([1,2,3,4]), X=pd.DataFrame([1,2,3,4])).split_train_test().regressor_optimise()
# test._write_file

from sklearn.datasets import make_regression
X, y = make_regression(n_samples=2000, n_features=10, n_informative=5)

test2 = AutomatedRegression(y=pd.DataFrame(y),
                            X=pd.DataFrame(X),
                            pca_value=2,
                            spline_value={'n_knots':4, 'degree':2},
                            poly_value=None,
                            n_trial=5,
                            overwrite=True,
                            list_regressors_optimise = ['lightgbm', 'lassolars'])
test2.split_train_test().regressor_optimise()
# test2.regressor_fit()

TransformerChooser(test2._study.best_params.get('n_quantiles'), test2.random_state).fit()