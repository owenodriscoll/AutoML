"""Model loaders and their hyperparameter search spaces, independent of AutomatedML."""

import numpy as np
import optuna
import pytest
from sklearn.datasets import make_classification, make_regression

from automl.classifiers import classifier_selector
from automl.regressors import regressor_selector

REGRESSORS = ['dummy', 'lightgbm', 'xgboost', 'catboost', 'bayesianridge', 'lassolars', 'adaboost', 'gradientboost',
              'histgradientboost', 'knn', 'sgd', 'bagging', 'svr', 'elasticnet']
CLASSIFIERS = ['dummy', 'lightgbm', 'xgboost', 'catboost', 'adaboost', 'gradientboost', 'histgradientboost', 'knn', 'sgd',
               'bagging', 'svc']


def sampled_params(create_params, seed):
    return create_params(optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed)).ask())


# --------------------------------------------- selection ---------------------------------------------
@pytest.mark.parametrize('name', REGRESSORS)
def test_regressor_loads(name):
    model, create_params = regressor_selector([name], random_state=0)[name]
    assert callable(model) and callable(create_params)


@pytest.mark.parametrize('name', CLASSIFIERS)
def test_classifier_loads(name):
    model, create_params = classifier_selector([name], n_classes=2, random_state=0)[name]
    assert callable(model) and callable(create_params)


def test_selection_keeps_requested_order():
    names = ['knn', 'lassolars', 'dummy']
    assert list(regressor_selector(names)) == names
    names = ['knn', 'svc', 'dummy']
    assert list(classifier_selector(names, n_classes=2)) == names


def test_no_valid_regressor_reports_and_returns_none(capsys):
    assert regressor_selector(['nonexistent']) is None
    assert 'No valid regressor' in capsys.readouterr().out


def test_no_valid_classifier_reports_and_returns_none(capsys):
    assert classifier_selector(['nonexistent'], n_classes=2) is None
    assert 'no valid classifier' in capsys.readouterr().out


def test_invalid_regressor_name_is_skipped(capsys):
    assert list(regressor_selector(['lassolars', 'nonexistent'])) == ['lassolars']
    assert "Invalid regressor(s) supplied: ['nonexistent']" in capsys.readouterr().out


def test_invalid_classifier_name_is_skipped():
    assert list(classifier_selector(['knn', 'nonexistent'], n_classes=2)) == ['knn']


# --------------------------------------------- search spaces ---------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('name', REGRESSORS)
def test_sampled_regressor_hyperparameters_fit_and_predict(name, seed):
    X, y = make_regression(n_samples=150, n_features=4, random_state=0)
    model, create_params = regressor_selector([name], random_state=0)[name]
    fitted = model().set_params(**sampled_params(create_params, seed)).fit(X, y)
    assert fitted.predict(X).shape == (150,)


@pytest.mark.slow
@pytest.mark.parametrize('seed', range(3))
@pytest.mark.parametrize('n_classes', [2, 3])
@pytest.mark.parametrize('name', CLASSIFIERS)
def test_sampled_classifier_hyperparameters_fit_and_predict(name, n_classes, seed):
    X, y = make_classification(n_samples=150, n_features=6, n_informative=4, n_classes=n_classes, random_state=0)
    model, create_params = classifier_selector([name], n_classes=n_classes, random_state=0)[name]
    fitted = model().set_params(**sampled_params(create_params, seed)).fit(X, y)
    assert set(np.ravel(fitted.predict(X))) <= set(y)


@pytest.mark.parametrize('name, parameter', [('lightgbm', 'random_state'), ('xgboost', 'random_state'),
                                             ('catboost', 'random_seed'), ('lassolars', 'random_state'),
                                             ('adaboost', 'random_state'), ('gradientboost', 'random_state'),
                                             ('histgradientboost', 'random_state'), ('sgd', 'random_state'), ('bagging', 'random_state'),
                                             ('svr', 'random_state'), ('elasticnet', 'random_state')])
def test_regressor_random_state_is_fixed(name, parameter):
    assert sampled_params(regressor_selector([name], random_state=123)[name][1], 0)[parameter] == 123


@pytest.mark.parametrize('name, parameter', [('lightgbm', 'random_state'), ('xgboost', 'random_state'),
                                             ('catboost', 'random_seed'), ('adaboost', 'random_state'),
                                             ('gradientboost', 'random_state'), ('histgradientboost', 'random_state'),
                                             ('sgd', 'random_state'), ('bagging', 'random_state'), ('svc', 'random_state')])
def test_classifier_random_state_is_fixed(name, parameter):
    assert sampled_params(classifier_selector([name], n_classes=2, random_state=123)[name][1], 0)[parameter] == 123


@pytest.mark.parametrize('selector', [regressor_selector, lambda names, **kw: classifier_selector(names, n_classes=2, **kw)])
def test_xgboost_tree_parameters_depend_on_booster(selector):
    create_params = selector(['xgboost'], random_state=0)['xgboost'][1]
    for seed in range(30):
        params = sampled_params(create_params, seed)
        assert ('max_depth' in params) == (params['booster'] != 'gblinear')
        assert ('rate_drop' in params) == (params['booster'] == 'dart')


@pytest.mark.parametrize('depth_limit, max_estimators', [(12, 200), (10, 300), (0, 400)])
def test_xgboost_estimators_are_limited_for_deep_trees(depth_limit, max_estimators):
    create_params = regressor_selector(['xgboost'], random_state=0)['xgboost'][1]
    sampled = [sampled_params(create_params, seed) for seed in range(100)]
    n_estimators = [p['n_estimators'] for p in sampled if p.get('max_depth', -1) >= depth_limit]
    assert n_estimators and max(n_estimators) <= max_estimators


def test_catboost_iterations_are_limited_for_deep_trees():
    create_params = regressor_selector(['catboost'], random_state=0)['catboost'][1]
    sampled = [sampled_params(create_params, seed) for seed in range(50)]
    assert all(p['iterations'] <= 300 for p in sampled if p['depth'] >= 8)
    assert all(p['iterations'] <= 400 for p in sampled if p['depth'] >= 6)


def test_lightgbm_classifier_receives_number_of_classes():
    create_params = classifier_selector(['lightgbm'], n_classes=5, random_state=0)['lightgbm'][1]
    params = sampled_params(create_params, 0)
    assert params['num_classes'] == 5 and params['objective'] == 'multiclass'
