from __future__ import annotations
import optuna
import joblib
import os
import pandas as pd
import numpy as np
from typing import Callable, Union, List
from optuna.samplers import TPESampler
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from AutoML.AutoML.scalers_transformers import PcaChooser, PolyChooser, SplineChooser, ScalerChooser, \
    TransformerChooser
from AutoML.AutoML.regressors import regressor_selector
from AutoML.AutoML.function_helper import FuncHelper

# include categorical feature support
# try polynomial features with interactions_only = True, include_bias = False
# make feature combo only occur if polyval or splineval included


class AutomatedRegression:
    def __init__(self,
                 y: pd.DataFrame,
                 X: pd.DataFrame,
                 test_frac: float = 0.2,
                 timeout: int = 600,
                 n_trial: int = 100,
                 cross_validation: callable = None,
                 sampler: callable = None,
                 pruner: callable = None,
                 poly_value: Union[int, float, dict, type(None)] = None,
                 spline_value: Union[int, float, dict, type(None)] = None,
                 pca_value: Union[int, float, dict, type(None)] = None,
                 metric_optimise: Callable = median_absolute_error,
                 metric_assess: List[Callable] = None,
                 optimisation_direction: str = 'minimize',
                 write_folder: str = os.getcwd() + '/auto_regression/',
                 overwrite: bool = False,
                 list_regressors_optimise: List[str] = None,
                 list_regressors_assess: List[str] = None,
                 fit_frac: List[float] = None,
                 random_state: Union[int, type(None)] = 42,
                 warning_verbosity: str = 'ignore'):
        """
        A class for automated regression, which optimizes hyperparameters and select best performing regressor(s).

        Parameters:
        -----------
        y: pandas.DataFrame
            Target values of shape (n_samples, 1).
        X: pandas.DataFrame
            Features of shape (n_samples, n_features).
        test_frac: float, optional (default=0.2)
            Fraction of the data to use as test data.
        timeout: int, optional (default=600)
            Timeout in seconds for optimization of hyperparameters.
        n_trial: int, optional (default=100)
            Number of trials for optimization of hyperparameters.
        cross_validation: callable, optional (default=KFold with 5 splits and shuffling, random_state=42)
            The cross-validation object to use for evaluation of models.
        sampler: callable, optional (default=TPESampler with seed=random_state)
            The sampler object to use for optimization of hyperparameters.
        pruner: callable, optional (default=HyperbandPruner with min_resource=1, max_resource='auto', reduction_factor=3)
            The pruner object to use for optimization of hyperparameters.
        poly_value: int, float, dict, optional (default=None)
            The polynomial transformation to apply to the data, if any. E.g. {'degree': 2, 'interaction_only'= False} or 2
        spline_value: int, float, dict, optional (default=None)
            The spline transformation to apply to the data, if any. {'n_knots': 5, 'degree':3} or or 5
        pca_value: int, float, dict, optional (default=None). 
            The PCA transformation to apply to the data, if any. E.g. {'n_components': 0.95, 'whiten'=False}
        metric_optimise: callable, optional (default=median_absolute_error)
            The metric to use for optimization of hyperparameters.
        metric_assess: list of callables, optional (default=[median_absolute_error, r2_score])
            The metrics to use for assessment of models.
        optimisation_direction: str, optional (default='minimize')
            The direction to optimize the hyperparameters, either 'minimize' or 'maximize'.
        write_folder: str, optional (default='/auto_regression/' in the current working directory)
            The folder where to write the results and models.
        overwrite: bool, optional (default=False)
            Whether to overwrite the existing files in the write_folder.
        list_regressors_optimise: list of str, optional (default=['lightgbm', 'xgboost', 'catboost', 'bayesianridge', 'lassolars'])
            The list of names of regressors to optimize, options: 'lightgbm', 'xgboost', 'catboost', 'bayesianridge', 'lassolars', 
            'adaboost', 'gradientboost','knn', 'sgd', 'bagging', 'svr', 'elasticnet'
        list_regressors_assess: list of str, optional (default=None)
            The list of names of regressors to assess. If None, uses the same as `list_regressors_optimise`.
        fit_frac: list of float, optional (default=[0.1, 0.2, 0.3, 0.4, 0.6, 1])
            The list of fractions of the data to use for fitting the models.
        random_state: int
            The random seed to use, default is 42.
        warning_verbosity: str
            The warning verbosity to use, default is 'ignore'.

        Methods
        -------
        regression_hyperoptimise:
            Performs hyperparameter optimization using the Optuna library. The method contains several
            nested functions and follows a pipeline for training and evaluating a regressor. The method starts by
            preparing the study for hyperparameter optimization and loops through each regressor in the list
            "regressors_2_optimise", optimizes its hyperparameters, and saves the final study iteration as a pickle file.

        regression_select_best:
            This method is used to create estimator pipelines for all the regressors specified in list_regressors_assess
            attribute and store them in the estimators attribute of the class instance.

        regression_evaluate:

        apply:
            applies in correct order the 'split_train_test', 'regression_hyperoptimise', 'regression_select_best' and
            'regression_evaluate' methods.

        Returns
        -------
        None

        """
        self.y = y
        self.X = X
        self.test_frac = test_frac
        self.timeout = timeout
        self.n_trial = n_trial
        self.cross_validation = cross_validation if 'split' in dir(cross_validation) else \
            KFold(n_splits=5, shuffle=True, random_state=random_state)
        self.sampler = sampler if 'optuna.samplers' in type(sampler).__module__ else TPESampler(seed=random_state)
        self.pruner = pruner if 'optuna.pruners' in type(pruner).__module__ else \
            optuna.pruners.HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3)
        self.poly_value = poly_value
        self.spline_value = spline_value
        self.pca_value = pca_value
        self.metric_optimise = metric_optimise
        self.metric_assess = [median_absolute_error, r2_score] if metric_assess is None else metric_assess
        self.optimisation_direction = optimisation_direction
        self.write_folder = write_folder
        self.overwrite = overwrite
        self.list_regressors_optimise = ['lightgbm', 'xgboost', 'catboost', 'bayesianridge', 'lassolars'] if \
            list_regressors_optimise is None else list_regressors_optimise
        self.list_regressors_assess = list_regressors_optimise if list_regressors_assess is None else \
            list_regressors_assess
        self.fit_frac = [0.1, 0.2, 0.3, 0.4, 0.6, 1] if fit_frac is None else fit_frac
        self.random_state = random_state
        self.regressors_2_optimise = regressor_selector(regressor_names=self.list_regressors_optimise,
                                                        random_state=self.random_state)
        self.regressors_2_assess = regressor_selector(regressor_names=self.list_regressors_assess,
                                                      random_state=self.random_state)
        self.warning_verbosity = warning_verbosity
        self.create_dir()

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_index = None
        self.test_index = None
        self.estimators = None
        self.y_pred = None
        self.summary = None

    def create_dir(self):
        if not os.path.exists(self.write_folder):
            os.makedirs(self.write_folder)
        return self

    def split_train_test(self, shuffle: bool = True):
        """
        Split the data into training and test sets.

        Parameters
        ----------
        shuffle : bool, optional
            Whether to shuffle the data before splitting, by default True

        Returns
        -------
        None

        The data is split and stored in class attributes.
        """
        df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(self.X, self.y, test_size=self.test_frac,
                                                                        random_state=self.random_state, shuffle=shuffle)

        # -- if independent X contains a single column it must be reshaped, else estimate.fit() fails
        if df_X_train.values.ndim == 1:
            X_train, X_test = (df_X_train.values.reshape(-1, 1), df_X_test.values.reshape(-1, 1))
        else:
            X_train, X_test = (df_X_train.values, df_X_test.values)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = np.ravel(df_y_train.values)
        self.y_test = np.ravel(df_y_test.values)
        self.train_index = df_X_train.index.values
        self.test_index = df_X_test.index.values
        return self

    @FuncHelper.method_warning_catcher
    def regression_hyperoptimise(self) -> AutomatedRegression:
        """
        Performs hyperparameter optimization on the regression models specified in `self.regressors_2_optimise` using Optuna.
        The optimization is performed on the training data and the final study is saved to disk.
        
        Returns:
            AutomatedRegression: The instance of the class with the updated study information.
            
        Raises:
            CatBoostError: If `catboost` is one of the regressors in `self.regressors_2_optimise`, the optimization process
            may raise this error if there is an issue with the `catboost` library.
        """

        def _optimise():
            """
            Optimizes the regressors specified in the `self.regressors_2_optimise` dictionary using Optuna.
            The study direction, sampler, and pruner are specified in the `self.optimisation_direction`, `self.sampler`, 
            and `self.pruner` attributes respectively. 
            
            The method uses the `_create_objective` function to create the objective function that is optimized by Optuna.
            The final study iteration is saved using joblib.
            """

            # -- if catboost is loaded prepare special catch for common catboost errors
            if 'catboost' in list(self.regressors_2_optimise.keys()):
                import catboost
                catch = (catboost.CatBoostError,)
            else:
                catch = ( )

            for regressor_name, (regressor, create_params) in self.regressors_2_optimise.items():
                study = optuna.create_study(direction=self.optimisation_direction, sampler=self.sampler,
                                            pruner=self.pruner)

                write_file = self.write_folder + regressor_name + '.pkl'

                # -- if regressor already trained, throw warning unless overwrite  == True
                if os.path.isfile(write_file):
                    if not self.overwrite:
                        message = "Regressor already exists in directory but overwrite set to 'False'. Regressor skipped."
                        print(len(message) * '_' + '\n' + message + '\n' + len(message) * '_')
                        continue
                    if self.overwrite:
                        message = "Regressor already exists in directory. Overwrite set to 'TRUE'"
                        print(len(message) * '_' + '\n' + message + '\n' + len(message) * '_')

                study.optimize(_create_objective(study, create_params, regressor, regressor_name, write_file),
                                     n_trials=self.n_trial, timeout=self.timeout, catch=catch)

                # -- save final study iteration
                joblib.dump(study, write_file)
            return

        def _create_objective(study, create_params, regressor, regressor_name, write_file):
            """
            Method creates the objective function that is optimized by Optuna. The objective function first saves
            the Optuna study and instantiates the scaler for the independent variables. Then, it determines if the
            feature combinations improve the results, and if so, fits the SplineChooser and PolyChooser. Next, it
            instantiates PCA compression and the transformer for the dependent variables. Finally, the method tunes
            the estimator algorithm and creates the regressor.
            """

            def _objective(trial):
                # save optuna study
                joblib.dump(study, write_file)

                # -- Instantiate scaler for independents
                scaler = ScalerChooser(trial=trial).suggest_fit()

                # -- determine if requested feature combinations improve results
                # -- only suggest this to trial of kwargs contain at least one of the relevant parameters
                optionals_included = any([bool(i) for i in [self.spline_value, self.poly_value]])

                spline_input = None
                poly_input = None
                
                if optionals_included:
                    feature_combo = trial.suggest_categorical("feature_combo", [False, True])
                    if feature_combo:
                        spline_input = self.spline_value
                        poly_input = self.poly_value

                spline = SplineChooser(spline_value=spline_input, trial=trial).fit_report_trial()
                poly = PolyChooser(poly_value=poly_input, trial=trial).fit_report_trial()

                # -- Instantiate PCA compression
                pca = PcaChooser(pca_value=self.pca_value, trial=trial).fit_report_trial()

                # -- Instantiate transformer for dependents
                transformer = TransformerChooser(random_state=self.random_state, trial=trial).suggest_and_fit()

                # -- Tune estimator algorithm
                param = create_params(trial)

                # -- Create regressor
                regresser_with_parameters = regressor().set_params(**param)

                # -- Create transformed regressor
                transformed_regressor = TransformedTargetRegressor(
                    regressor=regresser_with_parameters,
                    transformer=transformer
                )

                # -- Make a pipeline
                pipeline = Pipeline([('poly', poly), ('spline', spline), ('scaler', scaler), ('pca', pca),
                                           ('regressor', transformed_regressor)])

                return _model_performance(trial, regressor_name, pipeline)

            return _objective

        def _model_performance(trial, regressor_name, pipeline) -> float:
            """
            Evaluates the performance of the `pipeline` regressor. The performance is evaluated by splitting the data into 
            K-folds and iteratively training and assessing the regressor using an increasing fraction of the training and 
            test folds. If the performance for the first iterations is poor, the regressor is pruned.
            """
            
            # -- turn train and test arrays into temporary dataframes
            df_X_train = pd.DataFrame(self.X_train)
            df_y_train = pd.DataFrame(self.y_train)

            # -- retrieve list containing with dataframes for training and testing for each fold
            indexes_train_kfold = list(self.cross_validation.split(df_X_train))

            # -- the first trial does not require pruning, go straight to last fit fraction
            fractions = [self.fit_frac[-1]] if trial.number == 0 else self.fit_frac

            # -- prepare storage
            result_folds_fracs = []
            result_folds_stds = []
            
            # -- for each fraction value...
            for idx_fraction, partial_fit_frac in enumerate(fractions):

                # -- when too few samples are available for assessment proceed to next fraction
                min_samples = int(np.ceil(len(self.X_train) * partial_fit_frac * 1 / self.cross_validation.n_splits))
                if min_samples < 20:
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
                        set([regressor_name]) & set(['lightgbm']))  # xgboost and catboost ignored, bugs  out

                    # -- fit training data and add early stopping function if X-iterations did not improve data
                    # ... if regressor is boosted ...
                    if regressor_is_boosted:

                        # -- fit transformers to training fold of training data
                        fold_X_train_frac_transformed = pipeline[:-1].fit_transform(fold_X_train_frac)

                        # -- transform testing fold of training data
                        fold_X_test_frac_transformed = pipeline[:-1].transform(fold_X_test_frac)

                        # fit pipeline using pre-fitted transformers
                        pipeline.fit(fold_X_train_frac_transformed, fold_y_train_frac,
                                           regressor__eval_set=[(fold_X_test_frac_transformed, fold_y_test_frac)],
                                           regressor__early_stopping_rounds=20)

                    # ... if regressor is NOT boosted ...
                    else:
                        # -- fit training data
                        pipeline.fit(fold_X_train_frac, fold_y_train_frac)

                    # ... assess fold performance, sometimes performance is so poor a value error is thrown, therefore
                    # insert in 'try' function and return nan's for errors
                    try:
                        # ... if regressor is boosted ...
                        if regressor_is_boosted:
                            # ... make fold prediction on transformed test fraction of training dataset
                            prediction = pipeline.predict(fold_X_test_frac_transformed)
                        else:
                            # ... make fold prediction on original test fraction of training dataset
                            prediction = pipeline.predict(fold_X_test_frac)

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
                result_folds_fracs.append(result_folds_frac)
                result_folds_stds.append(result_folds_std)
                
                # -- only prune if not applied on fraction containing all datapoints
                if partial_fit_frac < 1.0:

                    # -- Report results to decide whether to prune
                    trial.report(result_folds_frac, idx_fraction)

                    # -- Prune the intermediate value if necessary.
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            # -- final performance are those obtained for last fraction (e.g. fraction of 1/1)
            performance = result_folds_fracs[-1]
            
            return performance

        if bool(self.regressors_2_optimise):
            _optimise()
            return self

    def regression_select_best(self) -> AutomatedRegression:
        """
        This method is used to create estimator pipelines for all the regressors specified in list_regressors_assess
        attribute and store them in the estimators attribute of the class instance.

        The method loads the study result for each regressor from the file with name "{regressor_name}.pkl" in
        write_folder directory. Then it instantiates objects of SplineChooser, PolyChooser, PcaChooser, ScalerChooser
        and TransformerChooser classes using the best parameters obtained from the study result. Next, it creates a
        pipeline using the Pipeline class from scikit-learn library. Each pipeline per regressor is added to a list of
        pipelines, which is then assigned to the estimators attribute of the class instance.

        Returns
        -------
        class instance.
        """

        estimators = []
        for regressor_name in self.list_regressors_assess:
            study = joblib.load(self.write_folder + regressor_name + '.pkl')

            spline = SplineChooser(spline_value=study.best_params.get('spline_value')).fit()
            poly = PolyChooser(poly_value=study.best_params.get('poly_value')).fit()
            pca = PcaChooser(pca_value=study.best_params.get('pca_value')).fit()
            scaler = ScalerChooser(arg=study.best_params.get('scaler')).string_to_func()
            transformer = TransformerChooser(study.best_params.get('n_quantiles'), self.random_state).fit()

            list_params = list(study.best_params)
            list_params_not_regressor = ['scaler', 'pca_value', 'spline_value', 'poly_value', 'feature_combo',
                                         'transformers', 'n_quantiles']
            list_params_regressor = set(list_params).difference(set(list_params_not_regressor))

            parameter_dict = {k: study.best_params[k] for k in study.best_params.keys() & set(list_params_regressor)}

            pipe_single_study = Pipeline([
                ('poly', poly),
                ('spline', spline),
                ('scaler', scaler),
                ('pca', pca),
                ('model', TransformedTargetRegressor(
                    # index 0 is the regressor, index 1 is hyper-optimization function
                    regressor=self.regressors_2_assess[regressor_name][0](**parameter_dict),
                    transformer=transformer
                ))]
            )
            estimators.append((regressor_name, pipe_single_study))
        self.estimators = estimators

        return self

    def regression_evaluate(self) -> AutomatedRegression:
        """
        Regression evaluation method of an estimator.

        This method will evaluate the regression performance of the estimators specified in 'list_regressors_assess' by
        splitting the test data into folds according to the cross-validation specified, training the estimators on the
        training data and evaluating the predictions on the test data. The performance will be stored in a dictionary
        per metric per estimator. If the estimator is the stacked regressor, it will be saved to disk.

        Returns
        -------
        class instance
        """

        # -- split data according to cross validation for assessment
        indexes_test_cv = list(self.cross_validation.split(self.X_test))

        # -- determine names of regressors to assess
        regressors_to_assess = self.list_regressors_assess + ['stacked']

        # -- create an empty dictionary to populate with performance while looping over regressors
        summary = dict([(regressor, list()) for regressor in regressors_to_assess])

        for i, regressor in enumerate(regressors_to_assess):
            estimator_temp = self.estimators[i:i + 1]

            # -- the final regressor is the stacked regressor
            if i == len(self.estimators):
                estimator_temp = self.estimators

                # -- fit stacked regressor while catching warnings
                regressor_final = StackingRegressor(estimators=estimator_temp,
                                                    final_estimator=Ridge(random_state=self.random_state),
                                                    cv=self.cross_validation)
                FuncHelper.function_warning_catcher(regressor_final.fit, [self.X_train, self.y_train],
                                                    self.warning_verbosity)

                # -- predict on the whole testing dataset
                self.y_pred = regressor_final.predict(self.X_test)

                # -- store stacked regressor, if file does exist, double check whether the user wants to overwrite
                write_file_stacked_regressor = self.write_folder + "stacked_regressor.joblib"
                if os.path.isfile(write_file_stacked_regressor):
                    if not self.overwrite:
                        question = "Stacked Regressor already exists in directory but overwrite set to 'False'. " \
                                   "Overwrite anyway ? (y/n): \n"
                        user_input =  input(len(question) * '_' + '\n' + question + '\n' + len(question) * '_')
                        if user_input != 'y':
                            response = "Stacked regressor not saved"
                            print(len(response) * '_' + '\n' + response + '\n' + len(response) * '_')
                    if self.overwrite:
                        question = "Stacked Regressor already exists in directory. Overwrite set to 'TRUE'. Are you " \
                                  "certain ? (y/n): \n"
                        user_input = input(len(question) * '_' + '\n' + question + '\n' + len(question) * '_')
                        if user_input != 'n':
                            response = "Stacked Regressor overwritten"
                            print(len(response) * '_' + '\n' + response + '\n' + len(response) * '_')
                            joblib.dump(regressor_final, write_file_stacked_regressor)

                # -- if file doesn't exist, write it
                if not os.path.isfile(write_file_stacked_regressor):
                    joblib.dump(regressor_final, write_file_stacked_regressor)

            else:
                regressor_final = estimator_temp[0][1]
                FuncHelper.function_warning_catcher(regressor_final.fit, [self.X_train, self.y_train],
                                                    self.warning_verbosity)

            # -- create dictionary with elements per metric allowing per metric fold performance to be stored
            metric_performance_dict = dict(
                [('metric_' + str(i), [metric, list()]) for i, metric in enumerate(self.metric_assess)])

            # -- For each TEST data fold...
            for idx_fold, fold in enumerate(indexes_test_cv):
                # -- Select the fold indexes
                fold_test = fold[1]

                # -- Predict on the TEST data fold
                prediction = regressor_final.predict(self.X_test[fold_test, :])

                # -- Assess prediction per metric and store per-fold performance in dictionary
                [metric_performance_dict[key][1].append(
                    metric_performance_dict[key][0](self.y_test[fold_test], prediction))
                 for key in metric_performance_dict]

            # -- store mean and standard deviation of performance over folds per regressor
            summary[regressor] = [
                [np.mean(metric_performance_dict[key][1]), np.std(metric_performance_dict[key][1])] for key in
                metric_performance_dict]

            self.summary = summary

        return self

    def apply(self):
        self.split_train_test()
        self.regression_hyperoptimise()
        self.regression_select_best()
        self.regression_evaluate()
        return self
