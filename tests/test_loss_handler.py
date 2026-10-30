"""Unit tests of the loss plumbing. Behaviour through the boosted libraries themselves is tested in test_loss.py"""

import copy
import functools

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, \
    root_mean_squared_error
from sklearn.preprocessing import QuantileTransformer
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from automl import CustomLoss, LossFallbackWarning
from automl.loss import CatBoostEvalMetric, EvalMetric, GradHessObjective, LightGBMEvalMetric, TargetCenterer, \
    TrainingObjective, XGBoostEvalMetric, absolute_error_grad_hess, find_known_objective, group_sum, linear_aggregation, \
    squared_error_grad_hess

from helpers import loss_handler, std_difference

MODELS = {'lightgbm': LGBMRegressor, 'xgboost': XGBRegressor, 'catboost': CatBoostRegressor}


@pytest.fixture
def index():
    return pd.MultiIndex.from_arrays([['b', 'b', 'a', 'a', 'a', 'c'], range(6)], names=['group', 'row'])


# ----------------------------------------------- CustomLoss -----------------------------------------------
def test_loss_is_named_after_metric():
    assert CustomLoss(r2_score).name == 'r2_score'
    assert CustomLoss(r2_score, name='mine').name == 'mine'


def test_loss_without_function_name_has_default_name():
    assert CustomLoss(functools.partial(r2_score)).name == 'custom_loss'


def test_lambda_grad_hess_is_rejected():
    with pytest.raises(AssertionError, match="Lambda"):
        CustomLoss(r2_score, grad_hess=lambda y_true, y_pred: (y_pred, y_pred))


def test_non_callable_metric_is_rejected():
    with pytest.raises(AssertionError, match="not callable"):
        CustomLoss(None)


# --------------------------------------------- known objectives ---------------------------------------------
@pytest.mark.parametrize('metric', [mean_squared_error, root_mean_squared_error, r2_score])
def test_squared_error_metrics_are_recognised(metric):
    assert find_known_objective(metric)['grad_hess'] is squared_error_grad_hess


def test_absolute_error_metric_is_recognised():
    assert find_known_objective(mean_absolute_error)['grad_hess'] is absolute_error_grad_hess


def test_unknown_metric_is_not_recognised():
    assert find_known_objective(std_difference) is None


def test_user_function_sharing_name_with_sklearn_metric_is_not_recognised():
    def r2_score(y_true, y_pred):
        return 0.

    assert find_known_objective(r2_score) is None


def test_known_gradients():
    y_true, y_pred = np.array([1., 2.]), np.array([3., 1.])
    np.testing.assert_array_equal(squared_error_grad_hess(y_true, y_pred)[0], [2., -1.])
    np.testing.assert_array_equal(squared_error_grad_hess(y_true, y_pred)[1], [1., 1.])
    np.testing.assert_array_equal(absolute_error_grad_hess(y_true, y_pred)[0], [1., -1.])


@pytest.mark.parametrize('agg_func, expected', [('mean', 'mean'), (np.mean, 'mean'), (np.nanmean, 'mean'), ('sum', 'sum'),
                                                (np.sum, 'sum'), (np.nansum, 'sum'), ('median', None), (np.median, None)])
def test_linear_aggregation(agg_func, expected):
    assert linear_aggregation(agg_func) == expected


# ---------------------------------------------- LossHandler ----------------------------------------------
@pytest.mark.parametrize('greater_is_better, sign', [(True, -1), (False, 1)])
def test_sign_converts_to_value_to_minimise(greater_is_better, sign):
    assert loss_handler(greater_is_better=greater_is_better).sign == sign


def test_group_labels(index):
    assert loss_handler().group_labels(index) is None
    handler = loss_handler(group_by_index_columns=['group'])
    np.testing.assert_array_equal(handler.group_labels(index), [0, 0, 1, 1, 1, 2])
    assert handler.group_labels(None) is None


def test_group_labels_of_multiple_columns():
    index = pd.MultiIndex.from_arrays([[1, 1, 1, 2], ['x', 'y', 'y', 'x'], range(4)], names=['a', 'b', 'c'])
    np.testing.assert_array_equal(loss_handler(group_by_index_columns=['a', 'b']).group_labels(index), [0, 1, 1, 2])


def test_non_list_grouping_is_ignored():
    assert not loss_handler(group_by_index_columns='group').is_grouped


def test_aggregate_without_grouping_passes_through():
    y_true, y_pred = object(), object()
    assert loss_handler().aggregate(y_true, y_pred) == (y_true, y_pred)


@pytest.mark.parametrize('agg_func, expected_true, expected_pred', [('mean', [4, 1, 6], [40, 10, 60]),
                                                                    ('sum', [12, 2, 6], [120, 20, 60])])
def test_aggregate_per_group(index, agg_func, expected_true, expected_pred):
    y_true = pd.DataFrame({'y': [1., 1., 3., 4., 5., 6.]}, index=index)
    y_pred = np.array([10., 10., 30., 40., 50., 60.])
    true, pred = loss_handler(group_by_index_columns=['group'], agg_func=agg_func).aggregate(y_true, y_pred)
    np.testing.assert_array_equal(true.loc[['a', 'b', 'c']].to_numpy().ravel(), expected_true)
    np.testing.assert_array_equal(pred.loc[['a', 'b', 'c']].to_numpy().ravel(), expected_pred)


def test_aggregate_multi_output(index):
    y_true = pd.DataFrame({'u': np.arange(6.), 'v': np.arange(6.) * 2}, index=index)
    true, pred = loss_handler(group_by_index_columns=['group']).aggregate(y_true, y_true.to_numpy() + 1)
    assert true.shape == pred.shape == (3, 2)
    np.testing.assert_array_equal(pred.to_numpy(), true.to_numpy() + 1)


def test_score_uses_group_aggregates(index):
    y_true = pd.DataFrame({'y': [1., 3., 2., 2., 2., 4.]}, index=index)
    y_pred = np.array([3., 1., 3., 1., 2., 4.])  # -- wrong per row, correct per group
    assert loss_handler(group_by_index_columns=['group']).score(y_true, y_pred) == pytest.approx(1.)
    assert loss_handler().score(y_true, y_pred) < 1.


def test_proba_output():
    binary = np.array([[.2, .8], [.6, .4]])
    multiclass = np.array([[.2, .3, .5]])
    np.testing.assert_array_equal(loss_handler().proba_output(binary), [.8, .4])
    np.testing.assert_array_equal(loss_handler().proba_output(multiclass), multiclass)
    np.testing.assert_array_equal(loss_handler().proba_output(np.array([.1, .9])), [.1, .9])


def test_predict_form_follows_loss():
    class Model:
        def predict(self, X):
            return 'labels'

        def predict_proba(self, X):
            return np.array([[.3, .7]])

    assert loss_handler(accuracy_score).predict(Model(), None) == 'labels'
    np.testing.assert_array_equal(loss_handler(CustomLoss(log_loss, needs_proba=True)).predict(Model(), None), [.7])


# ------------------------------------------- objective resolution -------------------------------------------
def test_user_grad_hess_takes_precedence_over_known_objective():
    loss = CustomLoss(r2_score, grad_hess=absolute_error_grad_hess)
    objective = loss_handler(loss).resolve_objectives(['xgboost'])['xgboost']
    assert objective.kind == 'custom' and objective.grad_hess is absolute_error_grad_hess


def test_grouped_catboost_with_grad_hess_falls_back_to_default():
    loss = CustomLoss(std_difference, grad_hess=squared_error_grad_hess)
    with pytest.warns(LossFallbackWarning, match="chunks of rows"):
        objectives = loss_handler(loss, False, group_by_index_columns=['group']).resolve_objectives(['catboost'])
    assert objectives['catboost'].kind == 'default'


def test_grouped_known_metric_with_non_linear_aggregation_uses_row_level_objective():
    with pytest.warns(LossFallbackWarning, match="row-level native objective"):
        objectives = loss_handler(r2_score, group_by_index_columns=['group'], agg_func='median').resolve_objectives(['lightgbm'])
    assert objectives['lightgbm'] == TrainingObjective('native', native='regression')


def test_multioutput_known_metric_uses_native_objective_per_target():
    objectives = loss_handler(r2_score, is_multioutput=True).resolve_objectives(['lightgbm', 'xgboost', 'catboost'])
    assert {objective.kind for objective in objectives.values()} == {'native'}


def test_fallback_warnings_are_combined_per_reason():
    with pytest.warns(LossFallbackWarning) as caught:
        loss_handler(std_difference, False).resolve_objectives(['lightgbm', 'xgboost', 'catboost'])
    assert len(caught) == 1


def test_classification_without_grad_hess_does_not_warn(recwarn):
    objectives = loss_handler(accuracy_score, ml_objective='classification').resolve_objectives(['lightgbm', 'svc'])
    assert {objective.kind for objective in objectives.values()} == {'default'}
    assert not [w for w in recwarn if issubclass(w.category, LossFallbackWarning)]


def test_unresolved_model_uses_default_objective():
    handler = loss_handler()
    assert not handler.uses_custom_objective('lightgbm')
    assert not handler.needs_per_target_wrapper('lightgbm', True)


@pytest.mark.parametrize('name, without_native, multioutput, joint, expected', [
    ('lightgbm', True, True, False, True), ('xgboost', True, True, True, False), ('lassolars', False, True, False, False),
    ('lightgbm', True, False, False, False)])
def test_per_target_wrapper(name, without_native, multioutput, joint, expected):
    handler = loss_handler(is_multioutput=multioutput)
    handler.objectives[name] = TrainingObjective('custom', native_multioutput=joint)
    assert handler.needs_per_target_wrapper(name, without_native) == expected


# ------------------------------------------- estimator configuration -------------------------------------------
def configured(handler, model_name, **kwargs):
    handler.resolve_objectives([model_name])
    return handler.configure_estimator(MODELS[model_name](), model_name, **kwargs)


def test_non_boosted_model_is_left_untouched():
    model = Ridge()
    assert loss_handler().configure_estimator(model, 'bayesianridge', early_stopping=True) == (model, {})


@pytest.mark.parametrize('model_name, parameter, native', [('lightgbm', 'objective', 'regression_l1'),
                                                           ('xgboost', 'objective', 'reg:absoluteerror'),
                                                           ('catboost', 'loss_function', 'MAE')])
def test_native_objective_is_set(model_name, parameter, native):
    model, _ = configured(loss_handler(mean_absolute_error, False), model_name)
    assert model.get_params()[parameter] == native


@pytest.mark.parametrize('model_name, parameter', [('lightgbm', 'objective'), ('xgboost', 'objective'), ('catboost', 'loss_function')])
def test_custom_objective_is_set(model_name, parameter):
    loss = CustomLoss(std_difference, grad_hess=squared_error_grad_hess)
    model, _ = configured(loss_handler(loss, False), model_name)
    assert isinstance(model.get_params()[parameter], GradHessObjective)


def test_catboost_custom_objective_requires_evaluation_metric():
    loss = CustomLoss(std_difference, grad_hess=squared_error_grad_hess)
    model, _ = configured(loss_handler(loss, False), 'catboost')
    assert model.get_params()['eval_metric'] == 'RMSE'


def test_early_stopping_metric_per_library():
    _, fit_params = configured(loss_handler(), 'lightgbm', early_stopping=True)
    assert isinstance(fit_params['eval_metric'], LightGBMEvalMetric)

    model, fit_params = configured(loss_handler(), 'xgboost', early_stopping=True)
    assert isinstance(model.get_params()['eval_metric'], XGBoostEvalMetric)
    assert fit_params == {'verbose': False}

    model, _ = configured(loss_handler(), 'catboost', early_stopping=True)
    assert isinstance(model.get_params()['eval_metric'], CatBoostEvalMetric)


def test_no_early_stopping_metric_without_early_stopping():
    model, fit_params = configured(loss_handler(), 'lightgbm')
    assert fit_params == {} and 'eval_metric' not in model.get_params()


def test_per_target_wrapped_models_keep_default_early_stopping_metric():
    model, fit_params = configured(loss_handler(is_multioutput=True), 'lightgbm', early_stopping=True)
    assert 'eval_metric' not in fit_params
    model, _ = configured(loss_handler(is_multioutput=True), 'xgboost', early_stopping=True)
    assert model.get_params()['eval_metric'] is None


# ------------------------------------------------ gradients ------------------------------------------------
def test_gradient_without_grouping_calls_grad_hess_directly():
    y_true, y_pred = np.array([1., 2.]), np.array([2., 4.])
    grad, hess = GradHessObjective(loss_handler(), squared_error_grad_hess)(y_true, y_pred)
    np.testing.assert_array_equal(grad, [1., 2.])
    np.testing.assert_array_equal(hess, [1., 1.])


def test_gradient_is_evaluated_on_original_scale():
    seen = {}

    def recording_grad_hess(y_true, y_pred):
        seen.update(y_true=y_true, y_pred=y_pred)
        return y_pred - y_true, np.ones_like(y_pred)

    objective = GradHessObjective(loss_handler(), recording_grad_hess)
    objective.target_transformer = TargetCenterer().fit(np.array([[10.], [20.]]))
    grad, _ = objective(np.array([-5., 5.]), np.array([-4., 8.]))
    np.testing.assert_array_equal(seen['y_true'], [10., 20.])
    np.testing.assert_array_equal(seen['y_pred'], [11., 23.])
    np.testing.assert_array_equal(grad, [1., 3.])


def test_mismatching_group_labels_raise(index):
    objective = GradHessObjective(loss_handler(group_by_index_columns=['group']), squared_error_grad_hess).bind(index=index)
    with pytest.raises(ValueError, match="do not match"):
        objective(np.zeros(5), np.zeros(5))


@pytest.mark.parametrize('agg_func', ['mean', 'sum'])
def test_leaf_containing_a_group_takes_newton_step_of_group_loss(index, agg_func):
    """A leaf value of -sum(grad)/sum(hess) must move the group aggregate exactly on target for squared error."""
    y_true, y_pred = np.array([1., 3., 2., 4., 9., 5.]), np.zeros(6)
    handler = loss_handler(group_by_index_columns=['group'], agg_func=agg_func)
    grad, hess = GradHessObjective(handler, squared_error_grad_hess).bind(index=index)(y_true, y_pred)

    for group in ['a', 'b', 'c']:
        rows = index.get_level_values('group') == group
        step = -grad[rows].sum() / hess[rows].sum()
        target = pd.Series(y_true[rows]).agg(agg_func)
        assert pd.Series(y_pred[rows] + step).agg(agg_func) == pytest.approx(target)


def test_group_gradient_multi_output(index):
    handler = loss_handler(group_by_index_columns=['group'], is_multioutput=True)
    objective = GradHessObjective(handler, squared_error_grad_hess).bind(index=index)
    grad, hess = objective(np.zeros((6, 2)), np.arange(12.).reshape(6, 2))
    assert grad.shape == hess.shape == (6, 2)
    np.testing.assert_allclose(grad[0], grad[1])  # -- same group, same gradient


def test_catboost_interface_returns_derivatives_to_maximise():
    objective = GradHessObjective(loss_handler(), squared_error_grad_hess)
    derivatives = objective.calc_ders_range([3., 1.], [1., 2.], None)
    assert derivatives == [(-2., -1.), (1., -1.)]


def test_group_sum():
    codes = np.array([1, 0, 1])
    np.testing.assert_array_equal(group_sum(np.array([1., 2., 3.]), codes), [2., 4.])
    np.testing.assert_array_equal(group_sum(np.array([[1., 2.], [3., 4.], [5., 6.]]), codes), [[3., 4.], [6., 8.]])


# ------------------------------------------------ evaluation metrics ------------------------------------------------
def test_adapters_are_shared_between_copies_to_retain_binding():
    adapter = EvalMetric(loss_handler())
    assert copy.deepcopy(adapter) is adapter
    assert clone(XGBRegressor(eval_metric=adapter)).get_params()['eval_metric'] is adapter


def test_adapter_takes_name_of_loss():
    assert EvalMetric(loss_handler()).__name__ == 'r2_score'


def test_eval_metric_on_data_other_than_the_bound_evaluation_set_is_nan(index):
    metric = EvalMetric(loss_handler()).bind(index=index)
    assert np.isnan(metric.score(np.zeros(3), np.zeros(3)))


def test_eval_metric_reverts_target_transformation():
    y = np.random.default_rng(0).normal(size=(100, 1))
    metric = EvalMetric(loss_handler(mean_squared_error, False))
    metric.target_transformer = QuantileTransformer(n_quantiles=50, output_distribution='normal').fit(y)
    transformed = metric.target_transformer.transform(y).ravel()
    assert metric.score(transformed, transformed + 1e-9) == pytest.approx(0, abs=1e-6)
    assert metric.score(transformed, transformed + 1) > 0


def test_eval_metric_groups(index):
    metric = EvalMetric(loss_handler(group_by_index_columns=['group'])).bind(index=index)
    y_true = np.array([1., 3., 2., 2., 2., 4.])
    assert metric.score(y_true, np.array([3., 1., 3., 1., 2., 4.])) == pytest.approx(1.)


@pytest.mark.parametrize('y_pred', [np.array([.9, .2, .1]), np.array([[.1, .9], [.8, .2], [.9, .1]])])
def test_eval_metric_translates_encoded_labels_to_classes(y_pred):
    seen = {}

    def recording_accuracy(y_true, y_pred):
        seen.update(y_true=np.ravel(y_true), y_pred=y_pred)
        return accuracy_score(y_true, y_pred)

    metric = EvalMetric(loss_handler(recording_accuracy, ml_objective='classification')).bind(classes=np.array([3, 7]))
    assert metric.score(np.array([1., 0., 1.]), y_pred) == pytest.approx(2 / 3)
    np.testing.assert_array_equal(seen['y_true'], [7, 3, 7])
    np.testing.assert_array_equal(seen['y_pred'], [7, 3, 3])


def test_eval_metric_passes_probabilities_when_needed():
    def probability_sum(y_true, y_pred):
        return float(np.sum(y_pred))

    handler = loss_handler(CustomLoss(probability_sum, needs_proba=True), False, ml_objective='classification')
    metric = EvalMetric(handler).bind(classes=np.array([0, 1]))
    assert metric.score(np.array([0., 1.]), np.array([.25, .5])) == pytest.approx(.75)


def test_library_metric_interfaces():
    y_true, y_pred = np.array([1., 2., 3.]), np.array([1., 2., 4.])
    expected = r2_score(y_true, y_pred)

    assert LightGBMEvalMetric(loss_handler())(y_true, y_pred) == ('r2_score', pytest.approx(expected), True)
    assert XGBoostEvalMetric(loss_handler())(y_true, y_pred) == pytest.approx(-expected)
    assert XGBoostEvalMetric(loss_handler(mean_squared_error, False))(y_true, y_pred) == pytest.approx(1 / 3)


def test_catboost_metric_interface():
    metric = CatBoostEvalMetric(loss_handler())
    assert metric.is_max_optimal()
    assert metric.get_final_error(1.5, 2.) == 1.5
    assert metric.evaluate([np.array([1., 2., 4.])], np.array([1., 2., 3.]), None) == (pytest.approx(.5), 1.)


def test_catboost_metric_unbound_data_scores_zero(index):
    metric = CatBoostEvalMetric(loss_handler()).bind(index=index)
    assert metric.evaluate([np.zeros(2)], np.zeros(2), None) == (0., 1.)


def test_catboost_metric_converts_raw_scores_to_probabilities():
    seen = {}

    def recording_loss(y_true, y_pred):
        seen['y_pred'] = y_pred
        return 0.

    handler = loss_handler(CustomLoss(recording_loss, needs_proba=True), False, ml_objective='classification')
    metric = CatBoostEvalMetric(handler).bind(classes=np.array([0, 1, 2]))

    metric.evaluate([np.array([0., np.log(3)])], np.array([0., 1.]), None)
    np.testing.assert_allclose(seen['y_pred'], [.5, .75])

    metric.evaluate(np.array([[0., 1.], [0., 0.], [0., -1.]]), np.array([0., 1.]), None)
    assert seen['y_pred'].shape == (2, 3)
    np.testing.assert_allclose(seen['y_pred'].sum(axis=1), 1.)
    np.testing.assert_allclose(seen['y_pred'][0], 1 / 3)


# ------------------------------------------------ target centerer ------------------------------------------------
@pytest.mark.parametrize('y', [np.array([[1.], [2.], [6.]]), np.array([[1., 10.], [3., 20.]])])
def test_target_centerer_round_trip(y):
    centerer = TargetCenterer().fit(y)
    np.testing.assert_allclose(centerer.transform(y).mean(axis=0), 0.)
    np.testing.assert_allclose(centerer.inverse_transform(centerer.transform(y)), y)


def test_target_centerer_leaves_gradient_unchanged():
    y_true, y_pred = np.array([[1.], [2.]]), np.array([[2.], [4.]])
    centerer = TargetCenterer().fit(y_true)
    np.testing.assert_allclose(squared_error_grad_hess(centerer.transform(y_true), centerer.transform(y_pred))[0],
                               squared_error_grad_hess(y_true, y_pred)[0])
