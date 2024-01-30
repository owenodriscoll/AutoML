from __future__ import annotations
import os
import sys
import pickle
import random
import optuna
import joblib
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable, Union, List, Dict, Any
from sqlalchemy import create_engine
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.compose import TransformedTargetRegressor
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.model_selection import train_test_split

from .scalers_transformers import PcaChooser, PolyChooser, SplineChooser, ScalerChooser, \
    TransformerChooser, CategoricalChooser#, FourrierExpansion
from .utils.function_helper import FuncHelper

# --------------- TODO LIST ---------------
# FIXME add encoding for clustering of feature importance 

# TODO try polynomial features with interactions_only = True, include_bias = False
# TODO add option to overwrite study instead of only coninuing previous available studies
# TODO add time constraint to reloading
# TODO boosted regression design trees using fixed loss function, e.g. RMSE, set loss function to training metric
# TODO several classification models accept class weights, implement class weight support


@dataclass
class AutomatedML:
    """
    A class for automated machine learning, which optimizes hyperparameters and select best performing model(s).

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
    n_weak_models: int, optional (default=0)
        Number of models to train stacked model on in addition to best model. For each specified
        model the best performing and randomly selected n_weak_models are used for stacking.
        E.g. if n_weak_models = 2 for 'lightgbm', the best performing 'lightgbm' model is used for stacking
        in addition to 2 other 'lightgbm' models. Setting this parameter to non-zero allows the stacked model
        to include (unique) additional information from the additional models, despite them performing worse
        independly than the best model
    n_jobs: int, optional (default=1)
        number of simoultaneous threads to run optimisation on. Not recommended to choose values greater than
        number of CPU cores. Simoultaneous threads can result in asynchronous optimisation, e.g. Trial 10
        may complete before Trial 4 and therefore Trial 10 cannot incorporate information of Trial 4 in the
        selection of its hyperparameters.         
    cross_validation: callable, optional (default=KFold with 5 splits and shuffling, random_state=42)
        The cross-validation object to use for evaluation of models.
    sampler: callable, optional (default=TPESampler with seed=random_state)
        The sampler object to use for optimization of hyperparameters.
    pruner: callable, optional (default=HyperbandPruner with min_resource=1, max_resource='auto', reduction_factor=3)
        The pruner object to prune unpromising training trials.
    poly_value: int, float, dict, optional (default=None)
        The polynomial transformation to apply to the data, if any. E.g. {'degree': 2, 'interaction_only'= False} or 2
    spline_value: int, float, dict, optional (default=None)
        The spline transformation to apply to the data, if any. {'n_knots': 5, 'degree':3} or 5
    fourrier_value: int,
        DEPRECIATED
    pca_value: int, float, dict, optional (default=None).
        The PCA transformation to apply to the data, if any. E.g. {'n_components': 0.95, 'whiten'=False}
    metric_optimise: callable, optional (default=median_absolute_error for regression, accuracy_score for classification)
        The metric to use for optimization of hyperparameters. 
    metric_assess: list of callables, optional (default=[median_absolute_error, r2_score])
        The metrics to use for assessment of models.
    optimisation_direction: str, optional (default='minimize')
        The direction to optimize the hyperparameters, either 'minimize' or 'maximize'.
    write_folder: str, optional (default='/AUTOML/' in the current working directory)
        The folder where to write the results and models.
    reload_study: bool, optional (default=False)
        Whether to continue study if previous study exists in write_folder.
    reload_trial_cap: bool, optional (default=False)
        Upper bound on number of trials if new trials are permitted on reloaded study. E.g. if n_trials = 50 and reloaded
        study already performed 40 trials, the new study will at most perform 10 additional trials
    models_to_optimize: list of str, optional (default=['lightgbm', 'xgboost', 'catboost', 'bayesianridge', 'lassolars'])
        The list of names of models to optimize, varies depending on whether objective is regression or classification.
        Check documentation of AutomatedML children classes for details
    models_to_assess: list of str, optional (default=None)
        The list of names of models to assess. If None, uses the same as `list_optimise`.
    boosted_early_stopping_rounds: int, optional (default=None)
        Number of early stopping rounds for 'lightgbm', 'xgboost' and 'catboost'. Lower values may be faster but yield
        less complex (and therefore perhaps worse) tuned models. Higher values generally results in longer optimization time
        per model but more models pruned. Early stopping not yet included for sklearn's GradientBoost and HistGradientBoost
    nominal_columns: list of Union[int, float, string)]
        Column headers of input DataFrame. These columns will be treated as containing nominal categorical columns
        Nominal columns contain unranked categories e.g. classes of weather type
    ordinal_columns: list of Union[int, float, string)]
        Column headers of input DataFrame. These columns will be treated as containing ordinal categorical columns.
        Ordinal columns contain ranked categories e.g. hours of the day
    fit_frac: list of float, optional (default=[0.1, 0.2, 0.3, 0.4, 0.6, 1])
        The list of fractions of the data to use for fitting the models.
    random_state: int
        The random seed to use, default is 42.
    warning_verbosity: str
        The warning verbosity to use, default is 'ignore'.

    Methods
    -------
    model_hyperoptimise:
        Performs hyperparameter optimization using the Optuna library. The method contains several
        nested functions and follows a pipeline for training and evaluating a regressor. The method starts by
        preparing the study for hyperparameter optimization and loops through each regressor in the list
        "regressors_2_optimise", optimizes its hyperparameters, and saves the final study iteration as a pickle file.

    model_select_best:
        This method is used to create estimator pipelines for all the regressors specified in models_to_assess
        attribute and store them in the estimators attribute of the class instance.

    model_evaluate:
        Evaluates performance of selected models. I first trains the models on the training dataset and
        then stacks the models and assesses performance on test fraction of dataset. 
        
    model_feature_importance:
        Explains feature importance of stacked model using SHAP.  

    apply:
        applies in correct order 'model_hyperoptimize', 'model_select_best' and 'model_evaluate' methods.

    Returns
    -------
    None

    """

    y: pd.DataFrame
    X: pd.DataFrame
    test_frac: float = 0.2
    timeout: int = 600
    n_trial: int = 100
    n_weak_models: int = 0
    n_jobs: int = 1
    cross_validation: callable = None
    sampler: callable = None
    pruner: callable = None
    poly_value: Union[int, float, dict, type(None)] = None
    spline_value: Union[int, float, dict, type(None)] = None
    pca_value: Union[int, float, dict, type(None)] = None
    fourrier_value: int = None
    metric_optimise: Callable = None
    metric_assess: List[Callable] = None
    optimisation_direction: str = 'maximize'
    write_folder: str = os.getcwd() + '/AUTOML/'
    reload_study: bool = False
    reload_trial_cap: bool = False
    overwrite: bool = False
    boosted_early_stopping_rounds: int = None
    nominal_columns: Union[List[str], type(None)] = None
    ordinal_columns: Union[List[str], type(None)] = None
    fit_frac: List[float] = None
    random_state: Union[int, type(None)] = 42
    warning_verbosity: str = 'ignore'
    X_train: pd.DataFrame = None
    X_test: pd.DataFrame = None
    y_train: pd.DataFrame = None
    y_test: pd.DataFrame = None
    train_index: Any = None
    test_index: Any = None
    estimators: List[Callable] = None
    y_pred: Any = None
    summary: Dict[str, List[float]] = None

    _models_optimize: List[Callable] = None
    _models_assess: List[Callable] = None
    _ml_objective: str = None
    _shuffle: bool = True
    _stratify: pd.DataFrame = None
    _model_final = None


    # -- conditionally mutate __init__ and call initialization functions
    def __post_init__(self):

        self.cross_validation = self.cross_validation if 'split' in dir(self.cross_validation) else \
            KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        self.sampler = self.sampler if 'optuna.samplers' in type(self.sampler).__module__ else \
            TPESampler(seed=self.random_state)

        self.pruner = self.pruner if 'optuna.pruners' in type(self.pruner).__module__ else \
            optuna.pruners.HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3)

        self.fit_frac = [0.1, 0.2, 0.3, 0.4, 0.6, 1] if self.fit_frac is None else self.fit_frac

        self.create_dir()
        self.split_train_test(shuffle=self._shuffle, stratify=self._stratify)


    def create_dir(self):
        if self.write_folder[-1] != "/": self.write_folder = self.write_folder + "/"

        self.write_folder_sampler = self.write_folder+"samplers/"

        # -- create storage folder for database files and models
        if not os.path.exists(self.write_folder):
            os.makedirs(self.write_folder)

        # -- create storage sub folder for specific samplers in case want to restart
        if not os.path.exists(self.write_folder_sampler):
            os.makedirs(self.write_folder_sampler)
        return self


    def split_train_test(self, shuffle: bool = True, stratify: pd.DataFrame = None ):
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

        # -- ensure input contains correct format
        if type(self.y) == np.ndarray: self.y = pd.DataFrame(self.y)
        if type(self.X) == np.ndarray: self.X = pd.DataFrame(self.X)
        if type(self.y) == pd.core.series.Series: self.y = self.y.to_frame()
        if type(self.X) == pd.core.series.Series: self.X = self.X.to_frame()
        self.y.columns = self.y.columns.astype(str)
        self.X.columns = self.X.columns.astype(str)

        # -- find and warn if non-numeric columns match
        non_numeric_columns = (~self.X.map(np.isreal).any(axis = 0))
        non_numeric_column_names = non_numeric_columns.index[non_numeric_columns].to_list()

        if type(self.nominal_columns) == type(self.ordinal_columns) == list:
            submitted_non_numeric = set(self.nominal_columns + self.ordinal_columns)
        elif type(self.nominal_columns) == type(self.ordinal_columns) == type(None):
            submitted_non_numeric = set([])
        elif type(self.nominal_columns) == type(None):
            submitted_non_numeric = set(self.ordinal_columns)
        elif type(self.ordinal_columns) == type(None):
            submitted_non_numeric = set(self.nominal_columns)

        non_numeric_difference = list(set(non_numeric_column_names) ^ submitted_non_numeric)
        if non_numeric_difference != []:
            print(f"Possible ordinal or nominal columns not specified as either: {non_numeric_difference})")

        # -- split dataframes
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_frac,
                                                                        random_state=self.random_state, shuffle=shuffle)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_index = X_train.index.values
        self.test_index = X_test.index.values

        return self


    @FuncHelper.method_warning_catcher
    def model_hyperoptimise(self) -> AutomatedML:
        """
        Performs hyperparameter optimization on the models specified in `self.models_to_optimize` using Optuna.
        The optimization is performed on the training data and the final study is saved to disk.
        
        Returns:
            AutomatedML: The instance of the class with the updated study information.
            
        Raises:
            CatBoostError: If `catboost` is one of the models in `self.models_to_optimize`, the optimization process
            may raise this error if there is an issue with the `catboost` library.
        """

        def _optimise():
            """
            Optimizes the models specified in the `self.models_to_optimize` dictionary using Optuna.
            The study direction, sampler, and pruner are specified in the `self.optimisation_direction`, `self.sampler`, 
            and `self.pruner` attributes respectively. 
            
            The method uses the `_create_objective` function to create the objective function that is optimized by Optuna.
            The final study iteration is saved using joblib.
            """

            # -- if catboost is loaded prepare special catch for common catboost errors
            if 'catboost' in list(self._models_optimize.keys()):
                import catboost
                catch = (catboost.CatBoostError,)
            else:
                catch = ( )

            for model_name, (model, create_params) in self._models_optimize.items():

                # -- create SQL database to store study results
                dir_study_db = f"{self.write_folder}{model_name}.db"
                dir_study_db_url = f"sqlite:///{dir_study_db}"
                dir_sampler = f"{self.write_folder_sampler}{model_name}_sampler.pkl"

                # -- check whether database already exists in which case should ...
                # ... use previous instance of sampler. Not necessary for pruner (#!!! really?)
                if os.path.exists(dir_sampler):
                    study_sampler = pickle.load(open(dir_sampler, "rb"))

                    # -- skip model if database already exists but reloading not permitted
                    if not self.reload_study:
                        message = [f"Study `{self._ml_objective}_{model_name}` already exists but `reload_study == False` -- > " +
                              "model skipped. \nSet `reload_study = True` to continue on existing study."]

                        # -- temporarily revert printing permission to notify of skipped model
                        FuncHelper.function_warning_catcher(
                            lambda x: print(x, flush=True), # flush = True prevents buffering of print statement
                            message,
                            new_warning_verbosity = 'default',
                            old_warning_verbosity = 'ignore',
                            new_std_error = sys.__stdout__
                            )
                        continue

                else:
                    study_sampler = self.sampler
                    create_engine(dir_study_db_url)

                # -- create study or reload previous
                study = optuna.create_study(study_name=f"{self._ml_objective}_{model_name}",
                                            direction=self.optimisation_direction,
                                            sampler=study_sampler,
                                            pruner=self.pruner,
                                            storage = dir_study_db_url,
                                            load_if_exists = self.reload_study)

                # -- prevent running more trials than cap
                if (self.reload_study) & (self.reload_trial_cap):
                    n_trials = self.n_trial - len(study.trials)
                    if n_trials <= 0:
                        continue
                else:
                    n_trials = self.n_trial

                study.optimize(_create_objective(study, create_params, model, model_name, dir_sampler),
                                      n_trials=n_trials, timeout=self.timeout, catch=catch, n_jobs=self.n_jobs)

            return


        def _create_objective(study, create_params, model, model_name, dir_sampler):
            """
            Method creates the objective function that is optimized by Optuna. The objective function first saves
            the Optuna study and instantiates the scaler for the independent variables. Then, it determines if the
            feature combinations improve the results, and if so, fits the SplineChooser and PolyChooser. Next, it
            instantiates PCA compression and the transformer for the dependent variables. Finally, the method tunes
            the estimator algorithm and creates the model.
            """

            def _objective(trial):
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

                # -- create fourrier expansion
                # fourrier = FourrierExpansion(fourrier_value=self.fourrier_value).fit() # FIXME

                # -- Instantiate PCA compression
                pca = PcaChooser(pca_value=self.pca_value, trial=trial).fit_report_trial()

                # -- Tune estimator algorithm
                param = create_params(trial)

                # -- Create model
                model_with_parameters = model().set_params(**param)

                # -- ordinal and nominal encoding
                categorical = CategoricalChooser(self.ordinal_columns, self.nominal_columns).fit()

                # -- allow for transformed regressor
                if self._ml_objective == 'regression':
                    # -- Instantiate transformer for dependents
                    transformer = TransformerChooser(random_state=self.random_state, trial=trial).suggest_and_fit()

                    model_final = TransformedTargetRegressor(
                        regressor=model_with_parameters,
                        transformer=transformer
                    )
                elif self._ml_objective == 'classification':
                    model_final = model_with_parameters

                # -- Make a pipeline
                pipeline = Pipeline([
                    ('categorical', categorical),
                    ('poly', poly),
                    ('spline', spline),
                    ('scaler', scaler),
                    # ('fourrier', fourrier), # FIXME
                    ('pca', pca),
                    ('model', model_final)
                    ])

                performance = _model_performance(trial, model_name, pipeline)

                # -- re-save the sampler after calculating each performance
                with open(dir_sampler, "wb") as sampler_state:
                    pickle.dump(study.sampler, sampler_state)

                return performance

            return _objective

        def _model_performance(trial, model_name, pipeline) -> float:
            """
            Evaluates the performance of the `pipeline` model. The performance is evaluated by splitting the data into
            K-folds and iteratively training and assessing the model using an increasing fraction of the training and
            test folds. If the performance for the first iterations is poor, the model is pruned.
            """

            # -- retrieve list containing with dataframes for training and testing for each fold
            indexes_train_kfold = list(self.cross_validation.split(self.X_train))

            # -- the first trial does not require pruning, go straight to last fit fraction
            fractions = [self.fit_frac[-1]] if trial.number == 0 else self.fit_frac

            # -- prepare storage
            result_folds_fracs = []
            result_folds_stds = []
            
            # -- performance is assessed per fraction so that pruning may early on remove unpromising trials
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
                    fold_X_train = self.X_train.iloc[fold[0]]
                    fold_X_test = self.X_train.iloc[fold[1]]
                    fold_y_train = self.y_train.iloc[fold[0]]
                    fold_y_test = self.y_train.iloc[fold[1]]

                    # -- retrieve indexes belonging to fraction of the fold
                    idx_partial_fit_train = fold_X_train.sample(frac=partial_fit_frac,
                                                                              random_state=self.random_state).index
                    idx_partial_fit_test = fold_X_test.sample(frac=partial_fit_frac,
                                                                            random_state=self.random_state).index

                    # -- select fraction of fold
                    fold_X_train_frac = fold_X_train.loc[idx_partial_fit_train]
                    fold_X_test_frac = fold_X_test.loc[idx_partial_fit_test]
                    fold_y_train_frac = fold_y_train.loc[idx_partial_fit_train]
                    fold_y_test_frac = fold_y_test.loc[idx_partial_fit_test]

                    # -- determine if model is  boosted model
                    early_stopping_permitted = bool(
                        set([model_name]) & set(['xgboost', 'catboost'])) # TODO add early stopping for LightGBM using a callback

                    if early_stopping_permitted: 
                        # -- During early stopping we assess the training performance of the model per round
                        # on the test fold of the training dataset. The model testing is performed during
                        # the last step of the pipeline. Therefore we must first apply all previous
                        # transformations on the test fold. To accomplish this we first fit all the
                        # previous pipeline steps. Next we transform the test fold (still of the training
                        # data). By doing so we have in effect created a transformed X matrix as it would
                        # be in the pipeline after all but the last pipeline steps. The last step, applying
                        # the model with early stopping, is therefore applied on an already transformed
                        # X-matrix.

                        # -- fit transformations in pipeline (all but the last step) for later use
                        pipeline[:-1].fit_transform(fold_X_train_frac)

                        # -- transform testing fold of training data
                        fold_X_test_frac_transformed = pipeline[:-1].transform(fold_X_test_frac)

                        # -- fit complete pipeline using properly transformed testing fold
                        pipeline.fit(fold_X_train_frac, fold_y_train_frac,
                                      model__eval_set=[(fold_X_test_frac_transformed, fold_y_test_frac)],
                                      model__early_stopping_rounds = self.boosted_early_stopping_rounds)

                    else:
                        # -- fit training data
                        pipeline.fit(fold_X_train_frac, fold_y_train_frac)

                    # -- assess fold performance
                    # -- in 'try' function as really poor performance can error out
                    try:
                        # -- make fold prediction on original test fraction of training dataset
                        prediction = pipeline.predict(fold_X_test_frac)

                        # -- assess prediction with chosen metric
                        result_fold = self.metric_optimise(fold_y_test_frac, prediction)
                        pass

                    except Exception as e:
                        print(e)
                        result_fold = np.nan
                        pass

                    # -- store results to assess performance per fraction
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

        if bool(self._models_optimize):
            _optimise()

            return 

    def model_select_best(self, random_state_model_selection=None) -> AutomatedML:
        """
        This method is used to create estimator pipelines for all the models specified in models_to_assess
        attribute and store them in the estimators attribute of the class instance.

        The method loads the study result for each model from the file with name "{model_name}.pkl" in
        write_folder directory. Then it instantiates objects of SplineChooser, PolyChooser, PcaChooser, ScalerChooser
        and TransformerChooser classes using the best parameters obtained from the study result. Next, it creates a
        pipeline using the Pipeline class from scikit-learn library. Each pipeline per model is added to a list of
        pipelines, which is then assigned to the estimators attribute of the class instance.

        Returns
        -------
        class instance.
        """

        # -- prepare all estimators for stacking
        estimators = []
        for model_name in list(self._models_assess.keys()):

            # -- set randomness parameters for randomly selecting models (if self.n_weak_models > 0)
            if type(random_state_model_selection) == type(None):
                random_state_model_selection = self.random_state
            random.seed(random_state_model_selection)

            # -- reload relevant study. Sampler not reloaded here as no additional studies are performed
            study = optuna.create_study(
                study_name=f"{self._ml_objective}_{model_name}",
                direction=self.optimisation_direction,
                storage=f"sqlite:///{self.write_folder}{model_name}.db",
                load_if_exists=True)

            # -- select parameters corresponding to model
            list_params = list(study.best_params)
            list_params_not_model = ['scaler', 'pca_value', 'spline_value', 'poly_value', 'feature_combo',
                                         'transformers', 'n_quantiles']
            list_params_model = set(list_params).difference(set(list_params_not_model))

            # -- select all trials associated with model
            df_trials = study.trials_dataframe()
            df_trials_non_pruned = df_trials[df_trials.state == 'COMPLETE']

            # -- ensure that selected number of weak models does not exceed `total completed trials` - `best trial`
            n_weak_models = self.n_weak_models
            if self.n_weak_models > len(df_trials_non_pruned) -1:

                message = ["Number of unique weak models less than requested number of weak models: " +
                           f"{len(df_trials_non_pruned) -1} < {self.n_weak_models} \n" +
                           "n_weak_models set to total number of weak models instead."]
                print(message[0], flush=True)

                n_weak_models = len(df_trials_non_pruned) -1

            # -- select best
            if self.optimisation_direction == 'maximize':
                idx_best = df_trials_non_pruned.index[df_trials_non_pruned.value.argmax()]
            elif self.optimisation_direction == 'minimize':
                idx_best = df_trials_non_pruned.index[df_trials_non_pruned.value.argmin()]

            # -- add additional models
            idx_remaining = df_trials_non_pruned.number.values.tolist()
            idx_remaining.remove(idx_best)
            idx_models = [idx_best] + random.sample(idx_remaining, n_weak_models)

            # -- name best and weaker models
            selected_models = [model_name+'_best']  + [model_name+'_'+str(i) for i in idx_models[1:]]

            # -- create estimator for best and additional weaker models
            for i, idx_model in enumerate(idx_models):

                model_params = study.trials[idx_model].params
                parameter_dict = {k: model_params[k] for k in model_params.keys() & set(list_params_model)}

                # -- select all the pipeline steps corresponding to input settings or best trial
                categorical = CategoricalChooser(self.ordinal_columns, self.nominal_columns).fit()
                spline = SplineChooser(spline_value=model_params.get('spline_value')).fit()
                poly = PolyChooser(poly_value=model_params.get('poly_value')).fit()
                pca = PcaChooser(pca_value=model_params.get('pca_value')).fit()
                scaler = ScalerChooser(arg=model_params.get('scaler')).string_to_func()

                model_with_parameters = self._models_assess[model_name][0](**parameter_dict)

                # -- Create transformed regressor
                if self._ml_objective == 'regression':
                    transformer = TransformerChooser(model_params.get('n_quantiles'), self.random_state).fit()

                    model_final = TransformedTargetRegressor(
                        regressor=model_with_parameters,
                        transformer=transformer
                    )
                # -- ... or normal classification model
                elif self._ml_objective == 'classification':
                    model_final = model_with_parameters

                pipe_single_study = Pipeline([
                    ('categorical', categorical),
                    ('poly', poly),
                    ('spline', spline),
                    ('scaler', scaler),
                    ('pca', pca),
                    ('model', model_final)]
                )
                estimators.append((selected_models[i], pipe_single_study))

        self.estimators = estimators
        self.list_all_models_assess = [estimator[0] for estimator in estimators]

        return


    def model_evaluate(self) -> AutomatedML:
        """
        Model evaluation method of an estimator.

        This method will evaluate the model performance of the estimators specified in 'models_to_assess' by
        splitting the test data into folds according to the cross-validation specified, training the estimators on the
        training data and evaluating the predictions on the test data. The performance will be stored in a dictionary
        per metric per estimator. If the estimator is the stacked model, it will be saved to disk.

        Returns
        -------
        class instance
        """
        
        # -- check whether split_train_test method has been performed, else perform it
        if getattr(self, 'X_train') is None: self.split_train_test()

        # -- split data according to cross validation for assessment
        indexes_test_cv = list(self.cross_validation.split(self.X_test))

        # -- determine names of models to assess
        models_to_assess = self.list_all_models_assess + ['stacked']

        # -- create an empty dictionary to populate with performance while looping over models
        summary = dict([(model, list()) for model in models_to_assess])

        for i, model in enumerate(models_to_assess):
            estimator_temp = self.estimators[i:i + 1]

            # -- the final model is the stacked model
            if i == len(self.estimators):
                estimator_temp = self.estimators

                # -- create a scorer compatible with Cross Validated Ridge
                greater_is_better = self.optimisation_direction == 'maximize'
                scoring = make_scorer(
                    self.metric_optimise, 
                    greater_is_better = greater_is_better
                    )

                # -- fit stacked model while catching warnings
                if self._ml_objective == 'regression':
                    self._model_final = StackingRegressor(
                        estimators=estimator_temp,
                        final_estimator=RidgeCV(scoring=scoring),
                        cv=self.cross_validation
                        )

                elif self._ml_objective == 'classification':
                    self._model_final = StackingClassifier(
                        estimators=estimator_temp,
                        final_estimator=RidgeClassifierCV(scoring=scoring),
                        cv=self.cross_validation
                        )

                FuncHelper.function_warning_catcher(self._model_final.fit, [self.X_train, self.y_train],
                                                    self.warning_verbosity)

                # -- predict on the whole testing dataset
                self.y_pred = self._model_final.predict(self.X_test)

                # -- store stacked model, if file already exists, confirm overwrite
                write_file_stacked_model = self.write_folder + "stacked_model.joblib"

                if os.path.isfile(write_file_stacked_model):
                    question = "Stacked model already exists in directory. Overwrite ? (y/n):"
                    user_input = input(len(question) * '_' + '\n' + question + '\n' + len(question) * '_' + '\n')

                    if user_input != 'n':
                        response = "Stacked model overwritten"
                        joblib.dump(self._model_final, write_file_stacked_model)
                    else:
                        response = "Stacked model not saved"

                    print(len(response) * '_' + '\n' + response + '\n' + len(response) * '_'  + '\n')

                # -- if file doesn't exist, write it
                if not os.path.isfile(write_file_stacked_model):
                    joblib.dump(self._model_final, write_file_stacked_model)

            else:
                self._model_final = estimator_temp[0][1]
                FuncHelper.function_warning_catcher(self._model_final.fit, [self.X_train, self.y_train],
                                                    self.warning_verbosity)
                

            # -- create dictionary with elements per metric allowing per metric fold performance to be stored
            metric_performance_dict = dict(
                [('metric_' + str(i), [metric, list()]) for i, metric in enumerate(self.metric_assess)])

            # -- For each TEST data fold...
            for idx_fold, fold in enumerate(indexes_test_cv):
                # -- Select the fold indexes
                fold_test = fold[1]

                # -- Predict on the TEST data fold
                prediction = self._model_final.predict(self.X_test.iloc[fold_test, :])

                # -- Assess prediction per metric and store per-fold performance in dictionary
                [metric_performance_dict[key][1].append(
                    metric_performance_dict[key][0](self.y_test.iloc[fold_test], prediction))
                 for key in metric_performance_dict]

            # -- store mean and standard deviation of performance over folds per model
            summary[model] = [
                [np.mean(metric_performance_dict[key][1]), np.std(metric_performance_dict[key][1])] for key in
                metric_performance_dict]

            self.summary = summary

        return
    

    def apply(self):
            self.model_hyperoptimise()
            self.model_select_best()
            self.model_evaluate()

            return

    
    def model_feature_importance(self, n_train_points = 200, n_test_points = 200, cluster = True):
        """
        NOTE DOES NOT WORK WITH NON-NUMERIC DATA
        NOTE requires installation of the shap package
            python3 -m pip install pyautoml[shap]

        Evaluates feature importance using shapely values. The SHAP kernel explainer is trained on 
        the training data (or on the cluster thereof). Then the explainer calculates for the test 
        data how parameters affect model performance.

        Parameters
        ----------
        n_train_points: int, default=200
            number of training observations (or clusters) for to use in explaining the model
        n_test_points: int, default=200
            number of test observations for which to assess feature importance
        cluster: bool, default=True
            whether to cluster the training data. If not individual points are chosen to create explainer 
            
        Returns
        -------
        shap_values:
            Shapely values calculated for data
        data:
            Subset of test data on which shapely values are calculated
        
        
        """

        import shap
        
        # -- reload the final model if it exists
        if type(self._model_final) is type(None):
            try:
                self._model_final = joblib.load(f"{self.write_folder}stacked_model.joblib")
            except:
                raise Exception(f"No trained model available in write_folder: {self.write_folder}")

        # -- cluster the training data to speed up, otherwise randomly sample training data
        if cluster:
            print('Clustering...')
            kmeans = KMeans(n_clusters = n_train_points, n_init = 10).fit(self.X_train)
            X_train_summary = kmeans.cluster_centers_
            
        else:
            X_train_summary = self.X_train.sample(n = n_train_points, random_state = self.random_state)
            
        
        # -- create explainer based on clustered training data
        ex = FuncHelper.function_warning_catcher(shap.KernelExplainer, [self._model_final.predict, X_train_summary],
                                            self.warning_verbosity)
        
        # -- select subset of test data 
        data = self.X_test.sample(n = n_test_points, random_state = self.random_state)
        feature_names = list(self.X_test.keys())

        # -- calculate SHAP values
        print('Calculating Shapely values...', flush=True)
        shap_values = FuncHelper.function_warning_catcher(ex.shap_values, [data],
                                                          self.warning_verbosity)

        # -- create summary plot
        shap.summary_plot(shap_values, features = data, feature_names = feature_names)
        
        # 
        # most_important_feature = feature_names[np.argmax(shap_values.var(axis = 1))]
        # shap.dependence_plot(most_important_feature, shap_values, features = data, feature_names = feature_names)

        return pd.DataFrame(data = shap_values, columns = feature_names), data
        

