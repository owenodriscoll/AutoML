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
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone, is_classifier
from sklearn.model_selection import check_cv

from .loss import LossHandler


def _rows(data, positions):
    return data.iloc[positions] if hasattr(data, 'iloc') else data[positions]


class _LossStackingBase(BaseEstimator):

    def __init__(self, estimators, loss: LossHandler, cv=None):
        self.estimators = estimators
        self.loss = loss
        self.cv = cv

    def _oof_predictions(self, X, y, methods: list) -> list:
        """
        Out-of-fold predictions of each base estimator, with grouped folds when grouping applies. Unlike with
        sklearn's cross_val_predict the splitter need not partition the data (e.g. TimeSeriesSplit, GroupShuffleSplit):
        rows in several test folds get the mean of their predictions and rows in no test fold are NaN.
        """
        groups = self.loss.group_labels(getattr(X, 'index', None))
        splits = list(check_cv(self.cv, y, classifier=is_classifier(self)).split(X, y, groups))
        return [self._oof_prediction(estimator, X, y, splits, method)
                for (_, estimator), method in zip(self.estimators, methods)]

    def _oof_prediction(self, estimator, X, y, splits: list, method: str) -> np.ndarray:
        folds = []
        for train, test in splits:
            fitted = clone(estimator).fit(_rows(X, train), _rows(y, train))
            folds.append((test, self._fold_output(getattr(fitted, method)(_rows(X, test)), fitted)))

        total = np.zeros((len(X), *folds[0][1].shape[1:]))
        count = np.zeros(total.shape)
        for test, prediction in folds:
            total[test] += prediction
            count[test] += 1

        with np.errstate(invalid='ignore'):
            return total / count

    def _fold_output(self, prediction, estimator) -> np.ndarray:
        return np.asarray(prediction, dtype=float)

    @staticmethod
    def _covered(predictions: np.ndarray) -> np.ndarray:
        """Mask of the rows with out-of-fold predictions, from predictions of shape (n_estimators, n_samples, n_outputs)."""
        return ~np.isnan(predictions).any(axis=(0, 2))

    def _fit_estimators(self, X, y):
        self.estimators_ = [(name, clone(estimator).fit(X, y)) for name, estimator in self.estimators]

    def _optimise(self, objective, starts: list, bounds: list) -> np.ndarray:
        """Minimises the objective with Powell's method, starting from the best of several initial guesses."""
        x0 = min(starts, key=objective)
        result = minimize(objective, x0, method='Powell', bounds=bounds)
        return result.x if objective(result.x) < objective(x0) else x0

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
        covered = self._covered(predictions)
        predictions, y_covered = predictions[:, covered], y.iloc[covered]
        n_estimators = len(predictions)
        n_weights = n_estimators * n_targets

        # -- weights act on centered predictions, with the intercept setting the level. For predictions with an
        # offset, weights and intercept are otherwise strongly coupled, on which the optimiser stalls
        offsets = predictions.mean(axis=1) if self.fit_intercept else np.zeros((n_estimators, n_targets))
        centered = predictions - offsets[:, None]

        def objective(theta):
            prediction = self._blend(centered, *self._unpack(theta, n_estimators, n_targets))
            return self._penalised(self.loss.score(y_covered, prediction))

        def start(weights):
            coef = np.repeat(weights[:, None], n_targets, axis=1)
            return np.r_[coef.ravel(), (coef * offsets).sum(axis=0) if self.fit_intercept else []]

        # -- start from equal weights or from each single best estimator
        starts = [start(np.full(n_estimators, 1 / n_estimators))] + [start(weights) for weights in np.eye(n_estimators)]
        bounds = [(0, None)] * n_weights + [(None, None)] * (n_targets if self.fit_intercept else 0)

        theta = self._optimise(objective, starts, bounds)
        coef, level = self._unpack(theta, n_estimators, n_targets)
        self.coef_, self.intercept_ = coef, level - (coef * offsets).sum(axis=0)
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
        probabilities = np.stack(self._oof_predictions(X, y, methods))
        covered = self._covered(probabilities)
        probabilities, y_frame = probabilities[:, covered], y_frame.iloc[covered]
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

    def _fold_output(self, prediction, estimator) -> np.ndarray:
        """Class probabilities on the columns of `classes_`, as a fitted estimator may not have seen all classes."""
        if self._method(estimator) == 'predict':
            return (np.asarray(prediction).reshape(-1, 1) == self.classes_).astype(float)
        probabilities = np.zeros((len(prediction), len(self.classes_)))
        probabilities[:, np.searchsorted(self.classes_, estimator.classes_)] = prediction
        return probabilities

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
            self._fold_output(estimator.predict_proba(X) if self._method(estimator) == 'predict_proba'
                              else estimator.predict(X), estimator)
            for _, estimator in self.estimators_
        ])
        return self._blend(probabilities, self.coef_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
