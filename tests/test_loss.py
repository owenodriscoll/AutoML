import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import QuantileTransformer

from automl import AutomatedRegression, AutomatedClassification, CustomLoss, LossFallbackWarning, OptimisationFailureWarning
from automl.loss import LossHandler, GradHessObjective, squared_error_grad_hess, TargetCenterer
from automl.estimators import LossAwarePipeline, LossAwareTransformedTargetRegressor
from automl.stacking import LossStackingRegressor


# --------------------------------------------- helpers ---------------------------------------------
def std_difference(y_true, y_pred):
    """Example of a metric without known training objective"""
    return float(abs(np.std(np.asarray(y_true)) - np.std(np.asarray(y_pred))))


def ratio_error(y_true, y_pred):
    """Multi-output metric on the ratio of the two targets"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true[:, 0] / y_true[:, 1] - y_pred[:, 0] / y_pred[:, 1]) ** 2))


def ratio_error_grad_hess(y_true, y_pred):
    r_true, r_pred = y_true[:, 0] / y_true[:, 1], y_pred[:, 0] / y_pred[:, 1]
    d0, d1 = 1 / y_pred[:, 1], -y_pred[:, 0] / y_pred[:, 1] ** 2
    grad = 2 * (r_pred - r_true)[:, None] * np.c_[d0, d1]
    hess = 2 * np.c_[d0, d1] ** 2
    return grad, hess


def make_grouped_data(n_groups=60, rows_per_group=12, seed=0):
    """Targets are only valid on group scale: the group mean is a function of the features, rows are noisy."""
    rng = np.random.default_rng(seed)
    index = pd.MultiIndex.from_product([[f"g{i:03d}" for i in range(n_groups)], range(rows_per_group)],
                                       names=['safe', 'index'])
    X = pd.DataFrame(rng.normal(size=(len(index), 4)), index=index, columns=list('abcd'))
    group_signal = X.groupby(level='safe').transform('mean') @ np.array([3., -2., 1., 0.])
    y = pd.Series(group_signal + rng.normal(scale=3, size=len(index)), index=index, name='target')
    return X, y


def automl_kwargs(tmp_path, **kwargs):
    defaults = dict(n_trial=3, timeout_study=60, write_folder=str(tmp_path / 'automl'), reload_study=False,
                    random_state=0, fit_frac=[1])
    return {**defaults, **kwargs}


def loss_handler(metric=r2_score, greater_is_better=True, **kwargs):
    return LossHandler(CustomLoss(metric) if callable(metric) else metric, greater_is_better, **kwargs)


# ------------------------------------- objective resolution -------------------------------------
def test_lambda_metric_rejected():
    with pytest.raises(AssertionError):
        CustomLoss(metric=lambda y_true, y_pred: 0.)


def test_known_metric_uses_native_objective_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error', LossFallbackWarning)
        objectives = loss_handler(r2_score).resolve_objectives(['lightgbm', 'xgboost', 'catboost', 'lassolars'])
    assert objectives['lightgbm'].native == 'regression'
    assert objectives['catboost'].native == 'RMSE'
    assert objectives['lassolars'].kind == 'default'
    assert loss_handler(mean_absolute_error, False).resolve_objectives(['xgboost'])['xgboost'].native == 'reg:absoluteerror'


def test_unknown_metric_without_grad_hess_warns_and_uses_default():
    with pytest.warns(LossFallbackWarning, match="no known equivalent training objective"):
        objectives = loss_handler(std_difference, False).resolve_objectives(['lightgbm', 'bayesianridge'])
    assert objectives['lightgbm'].kind == 'default'


def test_unknown_metric_with_grad_hess_uses_custom_objective():
    loss = CustomLoss(std_difference, grad_hess=squared_error_grad_hess)
    with pytest.warns(LossFallbackWarning, match="do not support custom training objectives"):
        objectives = loss_handler(loss, False).resolve_objectives(['catboost', 'bayesianridge'])
    assert objectives['catboost'].kind == 'custom'


def test_grouped_known_metric_uses_group_level_objective():
    with pytest.warns(LossFallbackWarning, match="CatBoost evaluates custom objectives on chunks"):
        objectives = loss_handler(r2_score, group_by_index_columns=['safe'], agg_func=np.nanmean).resolve_objectives(
            ['lightgbm', 'xgboost', 'catboost'])
    assert objectives['lightgbm'].kind == objectives['xgboost'].kind == 'custom'
    assert objectives['catboost'].kind == 'native'


def test_grouped_non_linear_aggregation_warns():
    with pytest.warns(LossFallbackWarning, match="agg_func"):
        objectives = loss_handler(r2_score, group_by_index_columns=['safe'], agg_func='median').resolve_objectives(['xgboost'])
    assert objectives['xgboost'].kind == 'native'


def test_classification_grad_hess_warns():
    loss = CustomLoss(accuracy_score, grad_hess=squared_error_grad_hess)
    with pytest.warns(LossFallbackWarning, match="only supported for regression"):
        objectives = loss_handler(loss, ml_objective='classification').resolve_objectives(['lightgbm'])
    assert objectives['lightgbm'].kind == 'default'


def test_multioutput_joint_grad_hess_only_for_xgboost():
    loss = CustomLoss(ratio_error, grad_hess=ratio_error_grad_hess)
    with pytest.warns(LossFallbackWarning, match="wrapped per target"):
        objectives = loss_handler(loss, False, is_multioutput=True).resolve_objectives(['xgboost', 'lightgbm'])
    assert objectives['xgboost'].kind == 'custom' and objectives['xgboost'].native_multioutput
    assert objectives['lightgbm'].kind == 'default'


# ------------------------------------------ gradients ------------------------------------------
@pytest.mark.parametrize('agg_func', ['mean', 'sum'])
def test_group_chain_rule_matches_finite_differences(agg_func):
    rng = np.random.default_rng(1)
    index = pd.MultiIndex.from_arrays([rng.integers(0, 5, 40), np.arange(40)], names=['safe', 'index'])
    y_true, y_pred = rng.normal(size=40), rng.normal(size=40)
    handler = loss_handler(group_by_index_columns=['safe'], agg_func=agg_func)
    objective = GradHessObjective(handler, squared_error_grad_hess).bind(index=index)

    def group_loss(pred):  # squared-error loss of the group aggregates, weighted by group size for 'mean'
        frame = pd.DataFrame({'t': y_true, 'p': pred}, index=index).groupby(level='safe').agg(agg_func)
        weight = pd.Series(index.get_level_values('safe')).value_counts() if agg_func == 'mean' else 1
        return 0.5 * np.sum(weight * (frame.p - frame.t) ** 2)

    grad, _ = objective(y_true, y_pred)
    eps = 1e-6
    numeric = [(group_loss(y_pred + eps * np.eye(40)[i]) - group_loss(y_pred - eps * np.eye(40)[i])) / (2 * eps)
               for i in range(40)]
    np.testing.assert_allclose(grad, numeric, atol=1e-6)


@pytest.mark.parametrize('model_name', ['lightgbm', 'xgboost'])
def test_grouped_objective_fits_group_means(model_name):
    """Training on the group-level loss should fit group means at least as well as training per row."""
    X, y = make_grouped_data()
    handler = loss_handler(group_by_index_columns=['safe'], agg_func='mean')
    handler.resolve_objectives([model_name])

    from automl.regressors import regressor_selector
    model_class = regressor_selector([model_name], random_state=0)[model_name][0]
    params = {'lightgbm': dict(n_estimators=200, verbosity=-1), 'xgboost': dict(n_estimators=200)}[model_name]

    model, _ = handler.configure_estimator(model_class(**params), model_name)
    assert handler.uses_custom_objective(model_name)
    pipeline = LossAwarePipeline([('model', LossAwareTransformedTargetRegressor(model, transformer=TargetCenterer()))])
    pipeline.fit(X, y.to_frame())
    prediction = pd.Series(np.ravel(pipeline.predict(X)), index=X.index)

    group_mean = lambda s: s.groupby(level='safe').mean()
    assert r2_score(group_mean(y), group_mean(prediction)) > 0.9


# ------------------------------------------ early stopping ------------------------------------------
@pytest.mark.parametrize('model_name', ['lightgbm', 'xgboost', 'catboost'])
def test_early_stopping_metric_evaluated_on_original_scale(model_name):
    X, y = make_grouped_data()
    seen = []

    def recording_metric(y_true, y_pred):
        seen.append(float(np.mean(np.asarray(y_true))))
        return r2_score(y_true, y_pred)

    handler = loss_handler(recording_metric, group_by_index_columns=['safe'])
    handler.resolve_objectives([model_name])
    from automl.regressors import regressor_selector
    model_class = regressor_selector([model_name], random_state=0)[model_name][0]
    model = model_class(**({'verbosity': -1} if model_name == 'lightgbm' else
                           {'logging_level': 'Silent'} if model_name == 'catboost' else {}))
    model.set_params(early_stopping_rounds=5)
    model, fit_params = handler.configure_estimator(model, model_name, early_stopping=True)

    pipeline = LossAwarePipeline([('model', LossAwareTransformedTargetRegressor(
        model, transformer=QuantileTransformer(n_quantiles=100, output_distribution='normal')))])
    train, test = X.index[:480], X.index[480:]
    pipeline.fit(X.loc[train], y.loc[train].to_frame(), eval_set=(X.loc[test], y.loc[test].to_frame()),
                 **{f'model__{k}': v for k, v in fit_params.items()})

    expected = y.loc[test].groupby(level='safe').mean().mean()
    assert seen and np.allclose(seen, expected)


# --------------------------------------------- stacking ---------------------------------------------
def test_loss_stacking_handles_multioutput_ratio_loss():
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)))
    y = pd.DataFrame({'a': 5 + X[0] + rng.normal(scale=.3, size=300), 'b': 10 + X[1] + rng.normal(scale=.3, size=300)})

    handler = loss_handler(ratio_error, False)
    stacker = LossStackingRegressor([('ridge', Ridge()), ('knn', KNeighborsRegressor())], loss=handler,
                                    cv=KFold(5, shuffle=True, random_state=0)).fit(X, y)
    assert stacker.coef_.shape == (2, 2) and (stacker.coef_ >= 0).all()
    assert stacker.predict(X).shape == (300, 2)

    # -- the weights are optimised on the out-of-fold predictions, where they must beat equal weights
    oof = np.stack(stacker._oof_predictions(X, y, ['predict', 'predict']))
    optimised = LossStackingRegressor._blend(oof, stacker.coef_, stacker.intercept_)
    equal = LossStackingRegressor._blend(oof, np.full((2, 2), .5), np.zeros(2))
    assert ratio_error(y, optimised) <= ratio_error(y, equal)


# ---------------------------------------- AutomatedML integration ----------------------------------------
def test_grouped_regression_end_to_end(tmp_path):
    X, y = make_grouped_data()
    with pytest.warns(LossFallbackWarning):
        regression = AutomatedRegression(y=y, X=X, **automl_kwargs(
            tmp_path, metric_optimise=std_difference, optimisation_direction='minimize',
            models_to_optimize=['lightgbm', 'bayesianridge'], boosted_early_stopping_rounds=5,
            group_by_index_columns=['safe'], agg_func=np.nanmean))

    # -- grouped data defaults to group-aware cross-validation, folds never share groups
    assert isinstance(regression.cross_validation, GroupKFold)
    groups = regression.X_train.index.get_level_values('safe')
    for train, test in regression._cv_splits(regression.X_train):
        assert not set(groups[train]) & set(groups[test])

    regression.apply()
    assert isinstance(regression._model_final, LossStackingRegressor)
    assert set(regression.summary) == {'lightgbm_best', 'bayesianridge_best', 'stacked'}
    assert all(len(metrics) == 3 for metrics in regression.summary.values())  # 2 metric_assess + the loss
    assert np.isfinite(regression.summary['stacked'][-1][0])


def test_multioutput_regression_with_joint_grad_hess(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, 3)))
    y = pd.DataFrame({'a': 5 + X[0] + rng.normal(scale=.3, size=400), 'b': 10 + X[1] + rng.normal(scale=.3, size=400)})
    loss = CustomLoss(ratio_error, grad_hess=ratio_error_grad_hess)
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(
        tmp_path, metric_optimise=loss, optimisation_direction='minimize', models_to_optimize=['xgboost'],
        boosted_early_stopping_rounds=5))
    assert regression._loss.objectives['xgboost'].native_multioutput

    regression.apply()
    assert regression.y_pred.shape == y.loc[regression.test_index].shape
    assert np.isfinite(regression.summary['stacked'][-1][0])


def test_classification_end_to_end(tmp_path):
    from sklearn.datasets import make_classification
    from sklearn.metrics import log_loss
    X, y = make_classification(n_samples=400, n_features=6, n_informative=4, n_classes=3, random_state=0)
    classification = AutomatedClassification(y=y, X=X, **automl_kwargs(
        tmp_path, metric_optimise=CustomLoss(log_loss, needs_proba=True), optimisation_direction='minimize',
        models_to_optimize=['lightgbm', 'svc'], boosted_early_stopping_rounds=5))
    with pytest.warns(OptimisationFailureWarning, match="of the trials of 'svc' failed"):  # -- has no `predict_proba`
        classification.apply()
    assert np.isfinite(classification.summary['stacked'][-1][0])


def test_non_group_aware_cross_validation_warns(tmp_path):
    X, y = make_grouped_data()
    with pytest.warns(UserWarning, match="not group-aware"):
        AutomatedRegression(y=y, X=X, **automl_kwargs(tmp_path, cross_validation=KFold(5), models_to_optimize=['lassolars'],
                                                      group_by_index_columns=['safe']))
