"""
Stacking meta-learners which directly optimise the (custom, possibly grouped or multi-output) loss.

The base estimators' out-of-fold predictions are combined with non-negative weights (stacked regression,
Breiman 1996). Rather than fitting these weights with least squares, they are found with a derivative-free
optimiser minimising the loss itself, so that any metric can be used, including metrics on group aggregates
and metrics combining multiple targets (e.g. the ratio of two targets).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.model_selection import cross_val_predict

from .loss import LossHandler


class _LossStackingBase(BaseEstimator):

    def __init__(self, estimators, loss: LossHandler, cv=None):
        self.estimators = estimators
        self.loss = loss
        self.cv = cv

    def _oof_predictions(self, X, y, methods: list) -> list:
        """Out-of-fold predictions of each base estimator, with grouped folds when grouping applies."""
        groups = self.loss.group_labels(getattr(X, 'index', None))
        return [cross_val_predict(estimator, X, y, cv=self.cv, groups=groups, method=method)
                for (_, estimator), method in zip(self.estimators, methods)]

    def _fit_estimators(self, X, y):
        self.estimators_ = [(name, clone(estimator).fit(X, y)) for name, estimator in self.estimators]

    def _optimise(self, objective, starts: list, bounds: list) -> np.ndarray:
        """Minimises the objective with Powell's method, starting from the best of several initial guesses."""
        x0 = min(starts, key=objective)
        result = minimize(objective, x0, method='Powell', bounds=bounds)
        return result.x if objective(result.x) <= objective(x0) else x0

    def _penalised(self, value) -> float:
        value = self.loss.sign * value
        return value if np.isfinite(value) else 1e300


class LossStackingRegressor(RegressorMixin, _LossStackingBase):
    """
    Stacking regressor which combines base estimators with non-negative weights per target, plus an intercept
    per target, chosen to directly optimise the loss on out-of-fold predictions. Supports multi-output.

    Parameters
    ----------
    estimators: list of (str, estimator)
        Base estimators.
    loss: LossHandler
        Loss to optimise, including grouping settings.
    cv: cross-validation generator or iterable
        Cross-validation used for the out-of-fold predictions. Groups are derived from the index of X.
    fit_intercept: bool, default=True
        Whether to fit an intercept per target.
    """

    def __init__(self, estimators, loss: LossHandler, cv=None, fit_intercept: bool = True):
        super().__init__(estimators=estimators, loss=loss, cv=cv)
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        y = y if isinstance(y, (pd.DataFrame, pd.Series)) else pd.DataFrame(y, index=getattr(X, 'index', None))
        y = y.to_frame() if isinstance(y, pd.Series) else y
        n_targets = y.shape[1]

        # -- predictions stacked into shape (n_estimators, n_samples, n_targets)
        oof_predictions = self._oof_predictions(X, y, ['predict'] * len(self.estimators))
        predictions = np.stack([np.asarray(p).reshape(len(p), n_targets) for p in oof_predictions])
        n_estimators = len(predictions)
        n_weights = n_estimators * n_targets

        def objective(theta):
            prediction = self._blend(predictions, *self._unpack(theta, n_estimators, n_targets))
            return self._penalised(self.loss.score(y, prediction))

        # -- start from equal weights or from each single best estimator
        intercepts = np.zeros(n_targets if self.fit_intercept else 0)
        starts = [np.r_[np.full(n_weights, 1 / n_estimators), intercepts]]
        starts += [np.r_[np.repeat(np.eye(n_estimators)[i], n_targets), intercepts] for i in range(n_estimators)]
        bounds = [(0, None)] * n_weights + [(None, None)] * len(intercepts)

        theta = self._optimise(objective, starts, bounds)
        self.coef_, self.intercept_ = self._unpack(theta, n_estimators, n_targets)
        self.n_targets_ = n_targets

        self._fit_estimators(X, y)
        return self

    def _unpack(self, theta, n_estimators, n_targets):
        coef = theta[:n_estimators * n_targets].reshape(n_estimators, n_targets)
        intercept = theta[n_estimators * n_targets:] if self.fit_intercept else np.zeros(n_targets)
        return coef, intercept

    @staticmethod
    def _blend(predictions, coef, intercept) -> np.ndarray:
        blended = np.einsum('knt,kt->nt', predictions, coef) + intercept
        return blended[:, 0] if blended.shape[1] == 1 else blended

    def predict(self, X):
        predictions = np.stack([np.asarray(estimator.predict(X)).reshape(len(X), self.n_targets_)
                                for _, estimator in self.estimators_])
        return self._blend(predictions, self.coef_, self.intercept_)


class LossStackingClassifier(ClassifierMixin, _LossStackingBase):
    """
    Stacking classifier which averages the class probabilities of the base estimators with non-negative
    weights chosen to directly optimise the loss on out-of-fold predictions. Estimators without
    `predict_proba` contribute one-hot encoded predictions.
    """

    def fit(self, X, y):
        y_frame = y if isinstance(y, pd.DataFrame) else pd.DataFrame(np.asarray(y), index=getattr(X, 'index', None))
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)

        methods = [self._method(estimator) for _, estimator in self.estimators]
        probabilities = np.stack([self._as_proba(prediction, estimator) for prediction, (_, estimator)
                                  in zip(self._oof_predictions(X, y, methods), self.estimators)])
        n_estimators = len(probabilities)

        def objective(weights):
            return self._penalised(self.loss.score(y_frame, self._output(self._blend(probabilities, weights))))

        starts = [np.full(n_estimators, 1 / n_estimators)] + [np.eye(n_estimators)[i] for i in range(n_estimators)]
        self.coef_ = self._optimise(objective, starts, bounds=[(0, None)] * n_estimators)

        self._fit_estimators(X, y)
        return self

    @staticmethod
    def _method(estimator) -> str:
        return 'predict_proba' if hasattr(estimator, 'predict_proba') else 'predict'

    def _as_proba(self, prediction, estimator) -> np.ndarray:
        if self._method(estimator) == 'predict_proba':
            return np.asarray(prediction)
        return (np.asarray(prediction).reshape(-1, 1) == self.classes_).astype(float)

    @staticmethod
    def _blend(probabilities, weights) -> np.ndarray:
        total = weights.sum()
        weights = weights / total if total > 0 else np.full(len(weights), 1 / len(weights))
        return np.einsum('knc,k->nc', probabilities, weights)

    def _output(self, proba):
        """Output in the form required by the loss: probabilities or labels."""
        if self.loss.loss.needs_proba:
            return self.loss.proba_output(proba)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        probabilities = np.stack([
            self._as_proba(estimator.predict_proba(X) if self._method(estimator) == 'predict_proba'
                           else estimator.predict(X), estimator)
            for _, estimator in self.estimators_
        ])
        return self._blend(probabilities, self.coef_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
