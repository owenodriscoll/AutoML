"""
Estimator wrappers that carry the custom loss into the fitting of the models.

The loss adapters (see loss.py) live inside the parameters of the boosted models. These wrappers bind the
information the adapters need, but which is lost once data enters the model: the row index (for group-level
losses), the fitted target transformer (to evaluate the loss on the original scale) and the classes.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .loss import LossAdapter, GradHessObjective, EvalMetric


def find_adapters(estimator, fit_params: dict = None, kind: type = LossAdapter) -> list:
    """Finds the loss adapters in the (nested) parameters of an estimator and in its fit parameters."""
    candidates = list((fit_params or {}).values())
    if hasattr(estimator, 'get_params'):
        candidates += list(estimator.get_params(deep=True).values())

    adapters = {id(c): c for c in candidates if isinstance(c, kind)}
    return list(adapters.values())


class LossAwarePipeline(Pipeline):
    """
    Pipeline which, prior to fitting, binds the row index and classes to the loss adapters of its final step.

    Additionally accepts `eval_set=(X_val, y_val)` for early stopping: X_val is transformed by all but the last
    pipeline steps (fitted on X) and passed as `eval_set` to the final step.
    """

    def fit(self, X, y=None, eval_set=None, **params):
        name, model = self.steps[-1]
        index = getattr(X, 'index', None)
        classes = np.unique(np.asarray(y)) if is_classifier(model) else None

        for adapter in find_adapters(model, params, kind=GradHessObjective):
            adapter.bind(index=index, classes=classes)

        if eval_set is not None:
            X_val, y_val = eval_set
            preprocessing = Pipeline(self.steps[:-1]) if len(self.steps) > 1 else None
            X_val_transformed = preprocessing.fit(X, y).transform(X_val) if preprocessing is not None else X_val
            for adapter in find_adapters(model, params, kind=EvalMetric):
                adapter.bind(index=getattr(X_val, 'index', None), classes=classes)
            params[f'{name}__eval_set'] = [(X_val_transformed, y_val)]

        return super().fit(X, y, **params)


class LossAwareTransformedTargetRegressor(TransformedTargetRegressor):
    """
    TransformedTargetRegressor which also transforms the targets of `eval_set` (otherwise early stopping
    compares predictions on the transformed scale with targets on the original scale) and provides the fitted
    transformer to the loss adapters, such that the loss is evaluated on the original scale.
    """

    def fit(self, X, y, **fit_params):
        transformer = None
        if self.transformer is not None:
            y_array = np.asarray(y, dtype=float)
            transformer = clone(self.transformer).fit(y_array.reshape(len(y_array), -1))

        for adapter in find_adapters(self.regressor, fit_params):
            adapter.target_transformer = transformer

        if (fit_params.get('eval_set') is not None) and (transformer is not None):
            fit_params['eval_set'] = [(X_val, _transform_like(transformer, y_val))
                                      for X_val, y_val in fit_params['eval_set']]

        return super().fit(X, y, **fit_params)


def _transform_like(transformer, y) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return transformer.transform(y.reshape(len(y), -1)).reshape(y.shape)


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """
    Fits a classifier on labels encoded to 0..n-1, which XGBClassifier requires, while predicting the original labels.
    The targets of `eval_set` are encoded accordingly.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, eval_set=None, **fit_params):
        self.encoder_ = LabelEncoder().fit(np.ravel(y))
        self.classes_ = self.encoder_.classes_

        if eval_set is not None:
            fit_params['eval_set'] = [(X_val, self.encoder_.transform(np.ravel(y_val))) for X_val, y_val in eval_set]

        self.estimator_ = clone(self.estimator).fit(X, self.encoder_.transform(np.ravel(y)), **fit_params)
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(np.ravel(self.estimator_.predict(X)).astype(int))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class EvalSetMultiOutputRegressor(MultiOutputRegressor):
    """MultiOutputRegressor which passes each per-target estimator only its own target column of `eval_set`."""

    def fit(self, X, y, **fit_params):
        eval_set = fit_params.pop('eval_set', None)
        if eval_set is None:
            return super().fit(X, y, **fit_params)

        y = np.asarray(y)
        self.estimators_ = []
        for i in range(y.shape[1]):
            eval_set_target = [(X_val, np.asarray(y_val)[:, i]) for X_val, y_val in eval_set]
            self.estimators_.append(clone(self.estimator).fit(X, y[:, i], eval_set=eval_set_target, **fit_params))

        if hasattr(self.estimators_[0], 'n_features_in_'):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        return self
