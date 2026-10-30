"""AutomatedML end to end: optimise, select the best models, stack and evaluate."""

import os
import pickle
import time
import warnings

import cloudpickle
import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, LeaveOneGroupOut, LeavePGroupsOut, ShuffleSplit, \
    StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer

from automl import AutomatedClassification, AutomatedRegression, CustomLoss, LossFallbackWarning
from automl.loss import TargetCenterer, squared_error_grad_hess
from automl.multi_output import MultiOutputStackingRegressor
from automl.stacking import LossStackingClassifier, LossStackingRegressor

from helpers import BOOSTED_MODELS, completed_trials, n_trials, std_difference, study_of
from test_model_selectors import CLASSIFIERS, REGRESSORS


def final_model(automl, name=None):
    pipeline = dict(automl.estimators)[name] if name else automl.estimators[0][1]
    return pipeline.named_steps['model']


def step(automl, name):
    return automl.estimators[0][1].named_steps[name]


# ------------------------------------------------- optimisation -------------------------------------------------
def test_optimisation_runs_requested_trials_per_model(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars', 'elasticnet'], n_trial=3))
    regression.model_hyperoptimise()
    assert n_trials(regression, 'lassolars') == n_trials(regression, 'elasticnet') == 3
    assert os.path.isfile(f"{regression.write_folder}samplers/lassolars_sampler.pkl")


def test_study_timeout_ends_optimisation(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], n_trial=100000, timeout_study=1))
    regression.model_hyperoptimise()
    assert 0 < n_trials(regression, 'lassolars') < 100000


@pytest.mark.parametrize('direction, expected_sign', [('maximize', 1), ('minimize', -1)])
def test_optimisation_direction_is_passed_to_study(regression_data, automl_kwargs, direction, expected_sign):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], optimisation_direction=direction, n_trial=5))
    regression.model_hyperoptimise()
    study = study_of(regression, 'lassolars')
    values = [t.value for t in study.trials]
    assert study.best_value == (max(values) if expected_sign == 1 else min(values))


def test_trials_are_scored_with_the_loss_on_cross_validation(regression_data, automl_kwargs):
    X, y = regression_data
    calls = []

    def recording_r2(y_true, y_pred):
        calls.append(len(y_true))
        return r2_score(y_true, y_pred)

    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], metric_optimise=recording_r2,
                                                               n_trial=1, cross_validation=KFold(4)))
    regression.model_hyperoptimise()
    assert calls == [60] * 4


def test_optimisation_does_not_see_test_data(regression_data, automl_kwargs):
    X, y = regression_data
    seen = []

    def recording_r2(y_true, y_pred):
        seen.extend(np.ravel(y_true))
        return r2_score(y_true, y_pred)

    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(
        models_to_optimize=['lightgbm'], metric_optimise=recording_r2, boosted_early_stopping_rounds=3,
        transformer_kwargs={'skip': True}))
    regression.model_hyperoptimise()

    rounded = lambda values: set(np.round(np.ravel(values), 3))  # -- LightGBM stores labels as float32
    assert seen
    assert rounded(seen) <= rounded(regression.y_train)
    assert not rounded(seen) - rounded(regression.y_train) & rounded(regression.y_test)


def test_first_trial_skips_partial_fits_and_later_trials_report_them(automl_kwargs):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(1000, 3)), columns=list('abc'))
    y = pd.Series(X.a + rng.normal(size=1000), name='target')
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], fit_frac=[.25, .5, 1], n_trial=3,
                                                               pruner=optuna.pruners.NopPruner()))
    regression.model_hyperoptimise()
    trials = study_of(regression, 'lassolars').trials
    assert trials[0].intermediate_values == {}
    assert sorted(trials[1].intermediate_values) == [0, 1]


def test_partial_fit_below_minimum_number_of_samples_is_skipped(automl_kwargs):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 3)), columns=list('abc'))
    y = pd.Series(X.a + rng.normal(size=500), name='target')
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], fit_frac=[.05, .5, 1], n_trial=2,
                                                               pruner=optuna.pruners.NopPruner()))
    regression.model_hyperoptimise()
    assert sorted(study_of(regression, 'lassolars').trials[1].intermediate_values) == [1]


def test_unpromising_trials_are_pruned(automl_kwargs):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(1000, 3)), columns=list('abc'))
    y = pd.Series(X.a + rng.normal(size=1000), name='target')
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], fit_frac=[.5, 1], n_trial=3,
                                                               pruner=optuna.pruners.ThresholdPruner(lower=2.)))
    regression.model_hyperoptimise()
    states = [t.state for t in study_of(regression, 'lassolars').trials]
    assert states == [optuna.trial.TrialState.COMPLETE] + [optuna.trial.TrialState.PRUNED] * 2


class Slow(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        time.sleep(30)
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_trial_timeout_prunes_slow_trials(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], timeout_trial=1, n_trial=1))
    regression._models_optimize = {'slow': (Slow, lambda trial: {})}
    regression.model_hyperoptimise()
    assert study_of(regression, 'slow').trials[0].state == optuna.trial.TrialState.PRUNED


def test_multithreaded_optimisation(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lightgbm'], n_trial=4, n_jobs=2,
                                                               boosted_early_stopping_rounds=3))
    regression.apply()
    assert n_trials(regression, 'lightgbm') == 4
    assert np.isfinite(regression.summary['stacked'][-1][0])


# ---------------------------------------------- errors during optimisation ----------------------------------------------
class Crashing(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        raise ValueError('crashed during fit')

    def predict(self, X):
        return np.zeros(len(X))


def test_failure_during_optimisation_is_reported_when_warnings_are_enabled(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], warning_verbosity='always'))
    regression._models_optimize = {'crashing': (Crashing, lambda trial: {})}
    with pytest.warns(UserWarning, match='crashed during fit'):
        regression.model_hyperoptimise()


@pytest.mark.xfail(reason="BUG: with the default warning_verbosity='ignore' any exception in model_hyperoptimise is "
                          "discarded without a trace, the user only sees a later unrelated failure or a missing model")
def test_failure_during_optimisation_is_visible_by_default(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars']))
    regression._models_optimize = {'crashing': (Crashing, lambda trial: {})}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            regression.model_hyperoptimise()
        except ValueError:
            return
    assert any('crashed during fit' in str(warning.message) for warning in caught)


@pytest.mark.xfail(reason="BUG: a model failing in one trial aborts the study loop, later models are silently not optimised")
def test_failing_model_does_not_stop_optimisation_of_other_models(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars', 'elasticnet']))
    lassolars = regression._models_optimize['lassolars']
    regression._models_optimize = {'crashing': (Crashing, lambda trial: {}), 'lassolars': lassolars}
    regression.model_hyperoptimise()
    assert os.path.isfile(f"{regression.write_folder}lassolars.db")


def test_library_errors_fail_a_trial_rather_than_the_study(regression_data, automl_kwargs):
    import lightgbm

    class LibraryError(RegressorMixin, BaseEstimator):
        def __init__(self, objective=None):  # -- set from the loss, as for any boosted model
            self.objective = objective

        def fit(self, X, y):
            raise lightgbm.basic.LightGBMError('no usable features')

        def predict(self, X):
            return np.zeros(len(X))

    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lightgbm'], n_trial=2))
    regression._models_optimize = {'lightgbm': (LibraryError, lambda trial: {})}
    regression.model_hyperoptimise()
    assert [t.state for t in study_of(regression, 'lightgbm').trials] == [optuna.trial.TrialState.FAIL] * 2


@pytest.mark.xfail(raises=RuntimeError, reason="BUG: with under 100 training rows (5 folds) every fraction is skipped for "
                                               "having fewer than 20 samples, the trial raises IndexError on the empty results "
                                               "and evaluation ends in an opaque 'Error in function warning catcher'")
def test_small_dataset(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y.iloc[:60], X=X.iloc[:60], **automl_kwargs(models_to_optimize=['lassolars']))
    regression.apply()
    assert 'stacked' in regression.summary


# --------------------------------------------------- reloading ---------------------------------------------------
@pytest.fixture
def run(regression_data, automl_kwargs):
    X, y = regression_data

    def execute(**kwargs):
        regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], **kwargs))
        regression.model_hyperoptimise()
        return regression

    return execute


def test_existing_study_is_skipped_without_reload(run, capsys):
    run(n_trial=2)
    regression = run(n_trial=2)
    assert n_trials(regression, 'lassolars') == 2
    assert "already exists" in capsys.readouterr().out


def test_reload_continues_study(run):
    run(n_trial=2)
    assert n_trials(run(n_trial=2, reload_study=True), 'lassolars') == 4


def test_reload_continues_from_saved_sampler(run):
    regression = run(n_trial=2)
    with open(f"{regression.write_folder}samplers/lassolars_sampler.pkl", 'rb') as f:
        assert isinstance(pickle.load(f), optuna.samplers.TPESampler)


def test_trial_cap_limits_total_trials(run):
    run(n_trial=3)
    assert n_trials(run(n_trial=5, reload_study=True, reload_trial_cap=True), 'lassolars') == 5


def test_trial_cap_already_reached_runs_nothing(run):
    run(n_trial=3)
    assert n_trials(run(n_trial=2, reload_study=True, reload_trial_cap=True), 'lassolars') == 3


def test_reloaded_study_can_be_selected_without_optimising(run, regression_data, automl_kwargs):
    run(n_trial=2)
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], reload_study=True))
    regression.model_select_best()
    regression.model_evaluate()
    assert 'stacked' in regression.summary


# ------------------------------------------------ model selection ------------------------------------------------
@pytest.mark.parametrize('direction', ['maximize', 'minimize'])
def test_best_trial_is_selected(regression_data, automl_kwargs, direction):
    X, y = regression_data
    metric = r2_score if direction == 'maximize' else mean_squared_error
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], n_trial=8, metric_optimise=metric,
                                                               optimisation_direction=direction))
    regression.model_hyperoptimise()
    regression.model_select_best()
    best = study_of(regression, 'lassolars').best_trial
    assert regression.list_all_models_assess == ['lassolars_best']
    assert final_model(regression).regressor.alpha == best.params['alpha']


def test_selected_pipeline_reproduces_best_trial_score(regression_data, automl_kwargs):
    """Rebuilding the pipeline from the stored parameters has to yield the score of the trial."""
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['histgradientboost'], n_trial=4,
                                                               pca_value=3, poly_value=2))
    regression.model_hyperoptimise()
    regression.model_select_best()

    scores = []
    for train, test in regression.cross_validation.split(regression.X_train):
        pipeline = regression.estimators[0][1].fit(regression.X_train.iloc[train], regression.y_train.iloc[train])
        scores.append(r2_score(regression.y_train.iloc[test], pipeline.predict(regression.X_train.iloc[test])))
    assert np.mean(scores) == pytest.approx(study_of(regression, 'histgradientboost').best_value)


def test_weak_models_are_added(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], n_trial=6, n_weak_models=2))
    regression.apply()
    names = regression.list_all_models_assess
    assert len(names) == 3 and names[0] == 'lassolars_best'
    assert len(set(names)) == 3 and set(regression.summary) == {*names, 'stacked'}


def test_number_of_weak_models_is_capped_by_available_trials(regression_data, automl_kwargs, capsys):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], n_trial=3, n_weak_models=10))
    regression.model_hyperoptimise()
    regression.model_select_best()
    assert len(regression.estimators) == 3
    assert "n_weak_models set to total" in capsys.readouterr().out


def test_weak_model_selection_is_seeded(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], n_trial=10, n_weak_models=2))
    regression.model_hyperoptimise()

    regression.model_select_best()
    default = regression.list_all_models_assess
    regression.model_select_best()
    assert regression.list_all_models_assess == default

    selections = set()
    for seed in range(8):
        regression.model_select_best(random_state_model_selection=seed)
        selections.add(tuple(regression.list_all_models_assess))
    assert len(selections) > 1


@pytest.mark.xfail(reason="BUG: model_select_best filters every trial on the parameter names of the best trial, so weak "
                          "models lose the parameters their booster has but the best trial's does not (and get defaults)")
def test_weak_models_keep_their_own_hyperparameters(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['dummy'], n_trial=20, n_weak_models=19))
    regression.model_hyperoptimise()
    regression.model_select_best()
    trials = study_of(regression, 'dummy').trials

    # -- 'quantile' is only sampled for strategy='quantile', which is never the best on this symmetric target
    assert 'quantile' not in study_of(regression, 'dummy').best_params
    weak_quantile_models = {name: trials[int(name.split('_')[-1])] for name, _ in regression.estimators[1:]
                            if 'quantile' in trials[int(name.split('_')[-1])].params}
    assert weak_quantile_models
    for name, trial in weak_quantile_models.items():
        assert dict(regression.estimators)[name].named_steps['model'].regressor.quantile == trial.params['quantile']


def test_models_to_assess_may_be_subset_of_optimised_models(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars', 'elasticnet'], models_to_assess=['elasticnet']))
    regression.apply()
    assert set(regression.summary) == {'elasticnet_best', 'stacked'}


def test_model_without_trials_is_skipped(regression_data, automl_kwargs, capsys):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], models_to_assess=['lassolars', 'elasticnet']))
    regression.apply()
    assert "No trials for elasticnet" in capsys.readouterr().out
    assert set(regression.summary) == {'lassolars_best', 'stacked'}


def test_model_with_only_failed_trials_is_skipped(regression_data, automl_kwargs, capsys):
    def always_invalid(y_true, y_pred):
        return np.nan

    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], metric_optimise=always_invalid))
    regression.model_hyperoptimise()
    regression.model_select_best()
    assert regression.estimators == [] and "No completed trials" in capsys.readouterr().out


def test_transformer_settings_of_best_trial_are_restored(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], n_trial=4,
                                                               transformer_kwargs={'n_quantiles': 100, 'output_distribution': 'normal'}))
    regression.model_hyperoptimise()
    regression.model_select_best()
    transformer = final_model(regression).transformer
    assert isinstance(transformer, QuantileTransformer)
    assert transformer.n_quantiles == 100 and transformer.output_distribution == 'normal'


@pytest.mark.xfail(reason="BUG: model_select_best passes random_state positionally into the `trial` parameter of "
                          "TransformerChooser, so the transformer always gets the default random_state of 42")
def test_transformer_uses_configured_random_state(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], random_state=7,
                                                               transformer_kwargs={'n_quantiles': 100, 'output_distribution': 'normal'}))
    regression.model_hyperoptimise()
    regression.model_select_best()
    assert final_model(regression).transformer.random_state == 7


def test_transformer_can_be_skipped(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], transformer_kwargs={'skip': True}))
    regression.model_hyperoptimise()
    regression.model_select_best()
    assert final_model(regression).transformer is None


# ----------------------------------------------- feature engineering -----------------------------------------------
def test_pca_is_restored_in_selected_pipeline(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], pca_value=2))
    regression.apply()
    assert isinstance(step(regression, 'pca'), PCA) and step(regression, 'pca').n_components == 2


def test_polynomial_features_are_selected_when_they_help(automl_kwargs):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 2)), columns=list('ab'))
    y = pd.Series(X.a * X.b + rng.normal(scale=.01, size=300), name='target')  # -- impossible for a linear model
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['bayesianridge'], n_trial=12, poly_value=2))
    regression.apply()
    assert isinstance(step(regression, 'poly'), PolynomialFeatures)
    assert regression.summary['stacked'][-1][0] > .7


def test_spline_features_are_restored_in_selected_pipeline(automl_kwargs):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(-3, 3, size=(300, 1)), columns=['a'])
    y = pd.Series(np.sin(X.a * 2) + rng.normal(scale=.01, size=300), name='target')
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['bayesianridge'], n_trial=12,
                                                               spline_value={'n_knots': 12, 'degree': 3}))
    regression.apply()
    assert step(regression, 'spline').n_knots == 12
    assert regression.summary['stacked'][-1][0] > .9


def test_feature_expansion_is_not_tried_without_settings(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars']))
    regression.model_hyperoptimise()
    assert 'feature_combo' not in study_of(regression, 'lassolars').best_params


@pytest.mark.parametrize('settings', [dict(nominal_columns=['nominal']), dict(ordinal_columns=['ordinal']),
                                      dict(nominal_columns=['nominal'], ordinal_columns=['ordinal'])])
@pytest.mark.parametrize('model_name', ['bayesianridge', 'lightgbm'])
def test_categorical_features(regression_data, automl_kwargs, settings, model_name):
    X, y = regression_data
    X = X.assign(nominal=pd.cut(X.a, 3, labels=['x', 'y', 'z']).astype(str), ordinal=pd.cut(X.b, 3, labels=['low', 'mid', 'high']).astype(str))
    X = X.drop(columns=[column for column in ['nominal', 'ordinal'] if column not in sum(settings.values(), [])])
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=[model_name], boosted_early_stopping_rounds=3, **settings))
    regression.apply()
    assert completed_trials(regression, model_name)
    assert np.isfinite(regression.summary['stacked'][-1][0])


# --------------------------------------------------- evaluation ---------------------------------------------------
@pytest.fixture
def evaluated(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars', 'knn'], n_trial=3))
    regression.apply()
    return regression


def test_summary_structure(evaluated):
    assert list(evaluated.summary) == ['lassolars_best', 'knn_best', 'stacked']
    for scores in evaluated.summary.values():
        assert np.shape(scores) == (3, 2)  # -- 2 assessment metrics + the loss, each [mean, std]


def test_summary_is_calculated_on_test_folds(evaluated):
    """Recalculates the summary independently of the code under test."""
    for name, pipeline in evaluated.estimators:
        folds = list(evaluated.cross_validation.split(evaluated.X_test))
        predictions = [pipeline.predict(evaluated.X_test.iloc[test]) for _, test in folds]
        targets = [evaluated.y_test.iloc[test] for _, test in folds]
        median_error = [median_absolute_error(t, p) for t, p in zip(targets, predictions)]
        r2 = [r2_score(t, p) for t, p in zip(targets, predictions)]
        np.testing.assert_allclose(evaluated.summary[name], [[np.mean(median_error), np.std(median_error)],
                                                             [np.mean(r2), np.std(r2)], [np.mean(r2), np.std(r2)]])


def test_base_models_are_fitted_on_all_training_data(evaluated):
    assert dict(evaluated.estimators)['knn_best'].named_steps['model'].regressor_.n_samples_fit_ == 240


def test_custom_assessment_metrics(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], metric_assess=[mean_squared_error]))
    regression.apply()
    assert np.shape(regression.summary['stacked']) == (2, 2)


def test_failing_assessment_metric_is_reported_as_nan(regression_data, automl_kwargs):
    def failing_metric(y_true, y_pred):
        raise ValueError

    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], metric_assess=[failing_metric]))
    regression.apply()
    assert np.isnan(regression.summary['stacked'][0][0])
    assert np.isfinite(regression.summary['stacked'][1][0])


def test_failing_assessment_metric_warns_when_warnings_are_enabled(regression_data, automl_kwargs):
    def failing_metric(y_true, y_pred):
        raise ValueError('cannot assess')

    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], metric_assess=[failing_metric],
                                                               warning_verbosity='always'))
    regression.model_hyperoptimise()
    regression.model_select_best()
    with pytest.warns(UserWarning, match='cannot assess'):
        regression.model_evaluate()


def test_stacked_predictions_cover_test_set(evaluated):
    assert evaluated.y_pred.shape == (60,)
    np.testing.assert_array_equal(evaluated.y_pred, evaluated._model_final.predict(evaluated.X_test))


def test_stacked_model_beats_baseline(evaluated):
    assert r2_score(evaluated.y_test, evaluated.y_pred) > .9


@pytest.mark.parametrize('metric, direction, needs_proba', [(accuracy_score, 'maximize', False), (log_loss, 'minimize', True)])
def test_loss_stacking_for_classification(multiclass_data, automl_kwargs, metric, direction, needs_proba):
    X, y = multiclass_data
    classification = AutomatedClassification(y=y, X=X, **automl_kwargs(
        models_to_optimize=['knn', 'bagging'], metric_optimise=CustomLoss(metric, needs_proba=needs_proba), optimisation_direction=direction))
    classification.apply()
    assert isinstance(classification._model_final, LossStackingClassifier)
    assert classification._model_final.coef_.shape == (2,)
    assert np.isfinite(classification.summary['stacked'][-1][0])


def test_default_stacking_is_loss_based(evaluated):
    assert isinstance(evaluated._model_final, LossStackingRegressor)


def test_ridge_stacking_regression(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars', 'knn'], stacking_meta_learner='ridge'))
    regression.apply()
    assert isinstance(regression._model_final, StackingRegressor)
    assert r2_score(regression.y_test, regression.y_pred) > .8


def test_ridge_stacking_multi_output(multioutput_data, automl_kwargs):
    X, y = multioutput_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars', 'knn'], stacking_meta_learner='ridge'))
    regression.apply()
    assert isinstance(regression._model_final, MultiOutputStackingRegressor)
    assert isinstance(regression.y_pred, pd.DataFrame) and list(regression.y_pred.columns) == ['first', 'second']


def test_ridge_stacking_classification(multiclass_data, automl_kwargs):
    X, y = multiclass_data
    classification = AutomatedClassification(y=y, X=X, **automl_kwargs(models_to_optimize=['knn', 'bagging'], stacking_meta_learner='ridge'))
    classification.apply()
    assert isinstance(classification._model_final, StackingClassifier)
    assert np.isfinite(classification.summary['stacked'][0][0])


def test_ridge_stacking_keeps_groups_together(grouped_data, automl_kwargs):
    X, y = grouped_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], group_by_index_columns=['group'],
                                                               stacking_meta_learner='ridge', test_frac=.5))
    regression.apply()
    groups = regression.X_train.index.get_level_values('group')
    for train, test in regression._model_final.cv:
        assert not set(groups[train]) & set(groups[test])


@pytest.mark.xfail(raises=RuntimeError, reason="BUG: the ridge meta-learner scores labels with metric_optimise, also if "
                                               "needs_proba=True, which fails for losses that require probabilities")
def test_ridge_stacking_with_probability_loss(multiclass_data, automl_kwargs):
    X, y = multiclass_data
    classification = AutomatedClassification(y=y, X=X, **automl_kwargs(
        models_to_optimize=['knn', 'bagging'], stacking_meta_learner='ridge', metric_optimise=CustomLoss(log_loss, needs_proba=True),
        optimisation_direction='minimize'))
    classification.apply()
    assert 'stacked' in classification.summary


# ------------------------------------------------ stored stacked model ------------------------------------------------
def test_stacked_model_is_stored_and_reloads(evaluated):
    with open(f"{evaluated.write_folder}stacked_model.cloudpickle", 'rb') as f:
        reloaded = cloudpickle.load(f)
    np.testing.assert_array_equal(reloaded.predict(evaluated.X_test), evaluated.y_pred)


@pytest.mark.parametrize('model_name', ['lightgbm', 'xgboost'])
def test_stacked_model_with_group_objective_reloads(grouped_data, automl_kwargs, model_name):
    X, y = grouped_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=[model_name], group_by_index_columns=['group'],
                                                               test_frac=.5))
    regression.apply()
    with open(f"{regression.write_folder}stacked_model.cloudpickle", 'rb') as f:
        reloaded = cloudpickle.load(f)
    np.testing.assert_array_equal(reloaded.predict(regression.X_test), regression.y_pred)


def stored_model_time(automl):
    return os.stat(f"{automl.write_folder}stacked_model.cloudpickle").st_mtime_ns


def test_stacked_model_is_not_overwritten_by_default(evaluated, capsys):
    before = stored_model_time(evaluated)
    evaluated.model_evaluate()
    assert stored_model_time(evaluated) == before
    assert "not saved" in capsys.readouterr().out


def test_stacked_model_is_overwritten_on_request(evaluated, capsys):
    before = stored_model_time(evaluated)
    evaluated.overwrite_stacked_model = True
    evaluated.model_evaluate()
    assert stored_model_time(evaluated) > before
    assert "overwritten" in capsys.readouterr().out


# ---------------------------------------------------- reproducibility ----------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize('models', [['lassolars', 'knn'], ['lightgbm'], ['xgboost'], ['catboost']], ids=lambda m: m[0])
def test_runs_are_reproducible(regression_data, automl_kwargs, tmp_path, models):
    X, y = regression_data
    summaries = []
    for folder in ['first', 'second']:
        regression = AutomatedRegression(y=y.copy(), X=X.copy(), **automl_kwargs(models_to_optimize=models, n_trial=3,
                                                                                 write_folder=str(tmp_path / folder)))
        regression.apply()
        summaries.append(regression.summary)
    assert summaries[0] == summaries[1]


# ------------------------------------------------------ all models ------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize('model_name', [pytest.param(name, marks=pytest.mark.xfail(raises=ImportError, reason="BUG: see test_model_selectors"))
                                        if name == 'sgd' else name for name in REGRESSORS])
def test_regressor_end_to_end(regression_data, automl_kwargs, model_name):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=[model_name]))
    regression.apply()
    assert set(regression.summary) == {f'{model_name}_best', 'stacked'}
    assert np.isfinite(regression.summary['stacked'][-1][0])


@pytest.mark.slow
@pytest.mark.parametrize('model_name', CLASSIFIERS)
@pytest.mark.parametrize('data', ['binary_data', 'multiclass_data'])
def test_classifier_end_to_end(request, automl_kwargs, model_name, data):
    X, y = request.getfixturevalue(data)
    classification = AutomatedClassification(y=y, X=X, **automl_kwargs(models_to_optimize=[model_name]))
    classification.apply()
    assert np.isfinite(classification.summary['stacked'][-1][0])
    assert set(np.unique(classification.y_pred)) <= set(y)


MULTI_OUTPUT_BUG = pytest.mark.xfail(raises=RuntimeError, reason="BUG: these models lack native multi-output but are missing from "
                                                                 "models_without_native_multioutput, so they are not wrapped and fail "
                                                                 "(TODO in automl.py). histgradientboost is even a default model")


@pytest.mark.slow
@pytest.mark.parametrize('model_name', [pytest.param(name, marks=MULTI_OUTPUT_BUG)
                                        if name in ['adaboost', 'gradientboost', 'histgradientboost', 'svr'] else name
                                        for name in REGRESSORS if name != 'sgd'])
def test_multi_output_regressor_end_to_end(multioutput_data, automl_kwargs, model_name):
    X, y = multioutput_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=[model_name]))
    regression.apply()
    assert isinstance(regression.y_pred, pd.DataFrame)
    assert regression.y_pred.shape == (60, 2) and list(regression.y_pred.columns) == ['first', 'second']
    assert regression.y_pred.index.equals(regression.X_test.index)


@pytest.mark.slow
@pytest.mark.parametrize('model_name', BOOSTED_MODELS)
def test_multi_output_boosted_models_with_early_stopping(multioutput_data, automl_kwargs, model_name):
    X, y = multioutput_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=[model_name], boosted_early_stopping_rounds=3))
    regression.apply()
    assert len(completed_trials(regression, model_name)) == 2


@pytest.mark.slow
@pytest.mark.parametrize('labels', [[3, 7], ['cat', 'dog']], ids=['integers', 'strings'])
@pytest.mark.parametrize('model_name', ['lightgbm', 'catboost', pytest.param('xgboost', marks=pytest.mark.xfail(
    reason="BUG: XGBClassifier requires labels 0..n-1 and labels are not encoded, so a trial crashes and the "
           "model is silently left out of the results"))])
def test_classifier_with_arbitrary_labels(binary_data, automl_kwargs, model_name, labels):
    X, y = binary_data
    y = y.map(dict(enumerate(labels)))
    classification = AutomatedClassification(y=y, X=X, **automl_kwargs(models_to_optimize=[model_name], boosted_early_stopping_rounds=3))
    classification.apply()
    assert set(np.unique(classification.y_pred)) <= set(labels)
    assert classification.summary['stacked'][-1][0] > .5


@pytest.mark.xfail(reason="BUG: n_classes is derived with set() of a DataFrame, which iterates over column names, so it is 2 "
                          "for any number of classes")
def test_number_of_classes_is_detected(multiclass_data, automl_kwargs):
    X, y = multiclass_data
    classification = AutomatedClassification(y=y, X=X, **automl_kwargs(models_to_optimize=['lightgbm']))
    create_params = classification._models_optimize['lightgbm'][1]
    assert create_params(optuna.create_study().ask())['num_classes'] == 3


# ------------------------------------------------- custom loss through stages -------------------------------------------------
def test_training_objective_reaches_final_models(regression_data, automl_kwargs):
    loss = CustomLoss(std_difference, grad_hess=squared_error_grad_hess)
    X, y = regression_data
    with pytest.warns(LossFallbackWarning, match="do not support custom training objectives"):
        regression = AutomatedRegression(y=y, X=X, **automl_kwargs(
            models_to_optimize=['lightgbm', 'lassolars'], metric_optimise=loss, optimisation_direction='minimize'))
    regression.model_hyperoptimise()
    regression.model_select_best()

    boosted = dict(regression.estimators)['lightgbm_best'].named_steps['model']
    assert isinstance(boosted.transformer, TargetCenterer)  # -- non-linear transformers would invalidate the gradients
    assert callable(boosted.regressor.get_params()['objective'])
    assert not isinstance(dict(regression.estimators)['lassolars_best'].named_steps['model'].transformer, TargetCenterer)


@pytest.mark.slow
def test_custom_objective_changes_what_is_learned(regression_data, automl_kwargs, tmp_path):
    def quantile_grad_hess(y_true, y_pred):  # -- pinball loss at 0.9: predictions shift to upper part of the data
        return np.where(y_pred < y_true, -.9, .1), np.ones_like(y_pred)

    X, y = regression_data
    means = {}
    for name, grad_hess in {'squared': squared_error_grad_hess, 'quantile': quantile_grad_hess}.items():
        loss = CustomLoss(std_difference, grad_hess=grad_hess)
        regression = AutomatedRegression(y=y, X=X, **automl_kwargs(
            models_to_optimize=['xgboost'], metric_optimise=loss, optimisation_direction='minimize', write_folder=str(tmp_path / name)))
        regression.apply()
        means[name] = np.mean(regression.y_pred)
    assert means['quantile'] > means['squared']


@pytest.mark.slow
@pytest.mark.parametrize('model_name', BOOSTED_MODELS)
@pytest.mark.parametrize('agg_func', ['mean', 'sum', np.nanmean])
def test_grouped_regression_with_early_stopping(grouped_data, automl_kwargs, model_name, agg_func):
    X, y = grouped_data
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', LossFallbackWarning)
        regression = AutomatedRegression(y=y, X=X, **automl_kwargs(
            models_to_optimize=[model_name], group_by_index_columns=['group'], agg_func=agg_func, boosted_early_stopping_rounds=3,
            test_frac=.5))
    regression.apply()
    assert completed_trials(regression, model_name)  # -- some parameters legitimately fail on 3 features in LightGBM
    assert np.isfinite(regression.summary['stacked'][-1][0])


def test_grouped_summary_is_calculated_on_group_aggregates(grouped_data, automl_kwargs):
    X, y = grouped_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], group_by_index_columns=['group'],
                                                               metric_assess=[mean_squared_error], test_frac=.5))
    regression.apply()

    pipeline = regression.estimators[0][1]
    scores = []
    for _, test in regression._cv_splits(regression.X_test):
        fold_X, fold_y = regression.X_test.iloc[test], regression.y_test.iloc[test]
        prediction = pd.Series(np.ravel(pipeline.predict(fold_X)), index=fold_X.index)
        scores.append(mean_squared_error(fold_y.groupby(level='group').mean(), prediction.groupby(level='group').mean()))
    assert regression.summary['lassolars_best'][0][0] == pytest.approx(np.mean(scores))


# ---------------------------------------------- cross-validation strategies ----------------------------------------------
@pytest.mark.parametrize('cv', [KFold(3), GroupKFold(3)], ids=lambda cv: type(cv).__name__)
def test_supported_cross_validators(grouped_data, automl_kwargs, cv):
    X, y = grouped_data
    settings = dict(group_by_index_columns=['group'], test_frac=.5) if isinstance(cv, GroupKFold) else {}
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], cross_validation=cv, **settings))
    regression.apply()
    assert 'stacked' in regression.summary


@pytest.mark.xfail(raises=TypeError, reason="BUG: _cv_splits does not pass y to split(), which StratifiedKFold requires, the error "
                                            "in optimisation is swallowed and the failure only surfaces in model_evaluate")
def test_stratified_cross_validation(binary_data, automl_kwargs):
    X, y = binary_data
    classification = AutomatedClassification(y=y, X=X, **automl_kwargs(models_to_optimize=['knn'], cross_validation=StratifiedKFold(3)))
    classification.apply()
    assert 'stacked' in classification.summary


@pytest.mark.xfail(raises=ValueError, reason="BUG: as StratifiedKFold, while listed in group_aware_cross_validators")
def test_stratified_group_cross_validation(grouped_data, automl_kwargs):
    X, y = grouped_data
    classification = AutomatedClassification(y=(y > y.median()).astype(int), X=X, **automl_kwargs(
        models_to_optimize=['knn'], test_frac=.5, cross_validation=StratifiedGroupKFold(3), group_by_index_columns=['group']))
    assert len(classification._cv_splits(classification.X_train)) == 3


@pytest.mark.xfail(raises=RuntimeError, reason="BUG: `n_splits` is read from the attribute, which these splitters lack "
                                               "(get_n_splits() should be used), while they are listed as group-aware")
@pytest.mark.parametrize('cv', [LeaveOneGroupOut(), LeavePGroupsOut(2)], ids=lambda cv: type(cv).__name__)
def test_leave_groups_out_cross_validation(grouped_data, automl_kwargs, cv):
    X, y = grouped_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], cross_validation=cv,
                                                               group_by_index_columns=['group'], test_frac=.5))
    regression.apply()
    assert 'stacked' in regression.summary


@pytest.mark.xfail(raises=RuntimeError, reason="BUG: cross_val_predict in the stacking step requires splitters that partition "
                                               "the data, so optimisation succeeds but stacking fails after all that work")
@pytest.mark.parametrize('cv, settings', [(ShuffleSplit(3, random_state=0), {}), (TimeSeriesSplit(3), {}),
                                          (GroupShuffleSplit(3, random_state=0), dict(group_by_index_columns=['group'], test_frac=.5))],
                         ids=['ShuffleSplit', 'TimeSeriesSplit', 'GroupShuffleSplit'])
def test_non_partitioning_cross_validation(grouped_data, automl_kwargs, cv, settings):
    X, y = grouped_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], cross_validation=cv, **settings))
    regression.apply()
    assert 'stacked' in regression.summary
