import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from automl.classifiers import classifier_selector
from automl.estimators import EvalSetMultiOutputRegressor, LossAwarePipeline, LossAwareTransformedTargetRegressor, find_adapters
from automl.loss import CustomLoss, EvalMetric, GradHessObjective, LossHandler, TargetCenterer, squared_error_grad_hess
from automl.regressors import regressor_selector

from helpers import BOOSTED_MODELS, loss_handler


class RecordingRegressor(RegressorMixin, BaseEstimator):
    """Stores what reaches the final step of a pipeline."""

    def __init__(self, objective=None, eval_metric=None):
        self.objective = objective
        self.eval_metric = eval_metric

    def fit(self, X, y, eval_set=None, **fit_params):
        self.X_fit_, self.y_fit_, self.eval_set_, self.fit_params_ = X, y, eval_set, fit_params
        return self

    def predict(self, X):
        return np.zeros(len(X))


class RecordingClassifier(ClassifierMixin, RecordingRegressor):
    pass


@pytest.fixture
def frame():
    index = pd.MultiIndex.from_product([[1, 2, 3, 4], range(3)], names=['group', 'row'])
    X = pd.DataFrame(np.arange(24.).reshape(12, 2), index=index, columns=['a', 'b'])
    return X, pd.DataFrame({'y': np.arange(12.)}, index=index)


# ------------------------------------------- find_adapters -------------------------------------------
def test_find_adapters_in_parameters_and_fit_params():
    objective, metric = GradHessObjective(loss_handler(), squared_error_grad_hess), EvalMetric(loss_handler())
    found = find_adapters(RecordingRegressor(objective=objective), {'eval_metric': metric, 'other': 1})
    assert {id(a) for a in found} == {id(objective), id(metric)}


def test_find_adapters_in_nested_parameters_is_unique():
    metric = EvalMetric(loss_handler())
    model = LossAwareTransformedTargetRegressor(EvalSetMultiOutputRegressor(RecordingRegressor(eval_metric=metric)))
    assert find_adapters(model) == [metric]
    assert find_adapters(model, {'eval_metric': metric}) == [metric]


def test_find_adapters_filters_on_kind():
    metric = EvalMetric(loss_handler())
    assert find_adapters(RecordingRegressor(eval_metric=metric), kind=GradHessObjective) == []


def test_find_adapters_of_object_without_parameters():
    assert find_adapters(object()) == []


# ----------------------------------------- LossAwarePipeline -----------------------------------------
def test_pipeline_binds_training_index_to_objective(frame):
    X, y = frame
    objective = GradHessObjective(loss_handler(group_by_index_columns=['group']), squared_error_grad_hess)
    LossAwarePipeline([('model', RecordingRegressor(objective=objective))]).fit(X, y)
    assert objective.index.equals(X.index)
    np.testing.assert_array_equal(objective.codes, np.repeat(range(4), 3))
    assert objective.classes is None


def test_pipeline_binds_evaluation_index_to_metric(frame):
    X, y = frame
    metric = EvalMetric(loss_handler())
    LossAwarePipeline([('model', RecordingRegressor(eval_metric=metric))]).fit(
        X.iloc[:9], y.iloc[:9], eval_set=(X.iloc[9:], y.iloc[9:]))
    assert metric.index.equals(X.index[9:])


def test_pipeline_binds_classes_for_classifiers(frame):
    X, y = frame
    metric = EvalMetric(loss_handler(accuracy_score, ml_objective='classification'))
    labels = pd.DataFrame({'y': np.tile([7, 3], 6)}, index=y.index)
    LossAwarePipeline([('model', RecordingClassifier(eval_metric=metric))]).fit(X, labels, eval_set=(X, labels))
    np.testing.assert_array_equal(metric.classes, [3, 7])


def test_pipeline_transforms_eval_set_with_steps_fitted_on_training_data(frame):
    X, y = frame
    X_train, X_val = X.iloc[:9], X.iloc[9:]
    pipeline = LossAwarePipeline([('scaler', StandardScaler()), ('model', RecordingRegressor())])
    pipeline.fit(X_train, y.iloc[:9], eval_set=(X_val, y.iloc[9:]))

    (X_val_seen, y_val_seen), = pipeline.named_steps['model'].eval_set_
    expected = StandardScaler().fit(X_train).transform(X_val)  # -- not fitted on the validation data
    np.testing.assert_allclose(X_val_seen, expected)
    assert y_val_seen.equals(y.iloc[9:])


def test_pipeline_without_preprocessing_passes_eval_set_unchanged(frame):
    X, y = frame
    pipeline = LossAwarePipeline([('model', RecordingRegressor())]).fit(X, y, eval_set=(X, y))
    assert pipeline.named_steps['model'].eval_set_[0][0] is X


def test_pipeline_ignores_skipped_steps(frame):
    X, y = frame
    pipeline = LossAwarePipeline([('scaler', None), ('pca', None), ('model', RecordingRegressor())])
    pipeline.fit(X, y, eval_set=(X, y))
    np.testing.assert_array_equal(pipeline.named_steps['model'].eval_set_[0][0], X)


def test_pipeline_forwards_fit_parameters(frame):
    X, y = frame
    pipeline = LossAwarePipeline([('model', RecordingRegressor())]).fit(X, y, model__verbose=False)
    assert pipeline.named_steps['model'].fit_params_ == {'verbose': False}


def test_pipeline_can_be_cloned(frame):
    X, y = frame
    objective = GradHessObjective(loss_handler(), squared_error_grad_hess)
    pipeline = clone(LossAwarePipeline([('model', RecordingRegressor(objective=objective))]))
    assert pipeline.fit(X, y).predict(X).shape == (12,)


# ------------------------------------ transformed target regressor ------------------------------------
def test_target_regressor_transforms_eval_set_targets():
    X, y = np.arange(20.).reshape(10, 2), np.arange(10.)
    regressor = RecordingRegressor()
    model = LossAwareTransformedTargetRegressor(regressor, transformer=TargetCenterer())
    model.fit(X, y, eval_set=[(X, y)])
    np.testing.assert_allclose(model.regressor_.eval_set_[0][1], y - y.mean())


def test_target_regressor_without_transformer_leaves_eval_set_unchanged():
    X, y = np.arange(20.).reshape(10, 2), np.arange(10.)
    model = LossAwareTransformedTargetRegressor(RecordingRegressor()).fit(X, y, eval_set=[(X, y)])
    np.testing.assert_array_equal(model.regressor_.eval_set_[0][1], y)


def test_target_regressor_provides_fitted_transformer_to_adapters():
    X, y = np.arange(20.).reshape(10, 2), np.arange(10.)
    metric = EvalMetric(loss_handler())
    LossAwareTransformedTargetRegressor(RecordingRegressor(eval_metric=metric), transformer=TargetCenterer()).fit(X, y)
    assert metric.target_transformer.mean_ == pytest.approx(4.5)

    LossAwareTransformedTargetRegressor(RecordingRegressor(eval_metric=metric)).fit(X, y)
    assert metric.target_transformer is None


def test_target_regressor_predicts_on_original_scale():
    X = np.arange(40.).reshape(20, 2)
    y = 1000 + 2 * X[:, 0]
    model = LossAwareTransformedTargetRegressor(LinearRegression(), transformer=TargetCenterer()).fit(X, y)
    np.testing.assert_allclose(np.ravel(model.predict(X)), y)


# ------------------------------------------- per-target wrapper -------------------------------------------
def test_multi_output_wrapper_gives_each_model_its_own_target():
    X, y = np.arange(20.).reshape(10, 2), np.arange(20.).reshape(10, 2)
    model = EvalSetMultiOutputRegressor(RecordingRegressor()).fit(X, y, eval_set=[(X, y)], verbose=False)
    for i, estimator in enumerate(model.estimators_):
        np.testing.assert_array_equal(estimator.y_fit_, y[:, i])
        np.testing.assert_array_equal(estimator.eval_set_[0][1], y[:, i])
        assert estimator.fit_params_ == {'verbose': False}
    assert model.predict(X).shape == (10, 2)


def test_multi_output_wrapper_without_eval_set_equals_sklearn():
    X = np.random.default_rng(0).normal(size=(30, 2))
    y = np.c_[X.sum(axis=1), X[:, 0]]
    ours = EvalSetMultiOutputRegressor(LinearRegression()).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), MultiOutputRegressor(LinearRegression()).fit(X, y).predict(X))


def test_multi_output_wrapper_exposes_number_of_features():
    X, y = np.ones((10, 3)), np.ones((10, 2))

    class WithFeatures(RecordingRegressor):
        def fit(self, X, y, **fit_params):
            self.n_features_in_ = X.shape[1]
            return self

    assert EvalSetMultiOutputRegressor(WithFeatures()).fit(X, y, eval_set=[(X, y)]).n_features_in_ == 3


# ---------------------------------- early stopping through the libraries ----------------------------------
LIGHTGBM_METRIC_BUG = pytest.mark.xfail(
    reason="BUG: LightGBM also evaluates its built-in metric and stops on that first metric (e.g. binary_logloss), so early "
           "stopping and the retained round do not follow metric_optimise. Set metric='None' alongside a custom eval_metric")
BOOSTED_MARKED = [pytest.param(name, marks=LIGHTGBM_METRIC_BUG) if name == 'lightgbm' else name for name in BOOSTED_MODELS]
BOOSTER_SILENT = {'lightgbm': {'verbosity': -1, 'n_estimators': 25}, 'xgboost': {'n_estimators': 25},
                  'catboost': {'logging_level': 'Silent', 'iterations': 25}}


def fit_with_early_stopping(model_class, model_name, handler, X, y, split, transformer=None):
    handler.resolve_objectives([model_name])
    model = model_class(**BOOSTER_SILENT[model_name]).set_params(early_stopping_rounds=1000)
    model, fit_params = handler.configure_estimator(model, model_name, early_stopping=True)
    final_step = model if transformer is None else LossAwareTransformedTargetRegressor(model, transformer=transformer)
    pipeline = LossAwarePipeline([('model', final_step)])
    pipeline.fit(X.iloc[:split], y.iloc[:split], eval_set=(X.iloc[split:], y.iloc[split:]),
                 **{f'model__{key}': value for key, value in fit_params.items()})
    return pipeline


@pytest.mark.parametrize('n_classes', [2, 3])
@pytest.mark.parametrize('model_name', BOOSTED_MARKED)
def test_classification_eval_metric_agrees_with_predictions(model_name, n_classes):
    """The accuracy reported to the library must be that of the model: the best round is what `predict` uses."""
    X, y = make_classification(n_samples=300, n_features=6, n_informative=4, n_classes=n_classes, random_state=0)
    X, y = pd.DataFrame(X), pd.DataFrame({'y': y})
    seen = []

    def recording_accuracy(y_true, y_pred):
        seen.append(accuracy_score(y_true, y_pred))
        return seen[-1]

    handler = LossHandler(CustomLoss(recording_accuracy), True, ml_objective='classification')
    model_class = classifier_selector([model_name], n_classes=n_classes, random_state=0)[model_name][0]
    pipeline = fit_with_early_stopping(model_class, model_name, handler, X, y, split=200)

    assert seen
    assert max(seen) == pytest.approx(accuracy_score(y.iloc[200:], pipeline.predict(X.iloc[200:])))


@pytest.mark.parametrize('model_name', BOOSTED_MODELS)
def test_regression_eval_metric_agrees_with_predictions(model_name, regression_data):
    X, y = regression_data
    y = y.to_frame()
    seen = []

    def recording_r2(y_true, y_pred):
        seen.append(r2_score(y_true, y_pred))
        return seen[-1]

    handler = loss_handler(recording_r2)
    model_class = regressor_selector([model_name], random_state=0)[model_name][0]
    pipeline = fit_with_early_stopping(model_class, model_name, handler, X, y, split=200,
                                       transformer=QuantileTransformer(n_quantiles=100, output_distribution='normal'))

    assert max(seen) == pytest.approx(r2_score(y.iloc[200:], pipeline.predict(X.iloc[200:])), abs=1e-6)


@pytest.mark.parametrize('model_name', BOOSTED_MARKED)
def test_early_stopping_honours_optimisation_direction(model_name, regression_data):
    """With a loss which only worsens with more rounds, stopping must retain an early round."""
    X, y = regression_data
    y = y.to_frame()

    def spread(y_true, y_pred):  # -- boosting steadily increases the spread of the predictions
        return float(np.std(y_pred))

    handler = loss_handler(spread, greater_is_better=False)
    model_class = regressor_selector([model_name], random_state=0)[model_name][0]
    pipeline = fit_with_early_stopping(model_class, model_name, handler, X, y, split=200)
    prediction_spread = np.std(np.ravel(pipeline.predict(X.iloc[200:])))
    assert prediction_spread < .5 * np.std(y.iloc[:200].to_numpy())
