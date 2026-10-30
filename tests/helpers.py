"""Shared metrics and stub estimators. Metrics are module-level as AutomatedML rejects lambda functions."""

import numpy as np
import optuna
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import r2_score

from automl.loss import CustomLoss, LossHandler


BOOSTED_MODELS = ['lightgbm', 'xgboost', 'catboost']


def std_difference(y_true, y_pred):
    """Example of a metric without known training objective"""
    return float(abs(np.std(np.asarray(y_true)) - np.std(np.asarray(y_pred))))


def loss_handler(metric=r2_score, greater_is_better=True, **kwargs):
    return LossHandler(CustomLoss(metric) if callable(metric) else metric, greater_is_better, **kwargs)


def study_of(automl, model_name):
    return optuna.load_study(study_name=f"{automl._ml_objective}_{model_name}",
                             storage=f"sqlite:///{automl.write_folder}{model_name}.db")


def n_trials(automl, model_name):
    return len(study_of(automl, model_name).trials)


def completed_trials(automl, model_name):
    return [t for t in study_of(automl, model_name).trials if t.state == optuna.trial.TrialState.COMPLETE]


class ColumnRegressor(RegressorMixin, BaseEstimator):
    """Predicts column(s) of X plus a shift, such that out-of-fold predictions are known exactly."""

    def __init__(self, columns=(0,), shift=0.):
        self.columns = columns
        self.shift = shift

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        prediction = np.asarray(X, dtype=float)[:, list(self.columns)] + self.shift
        return prediction[:, 0] if prediction.shape[1] == 1 else prediction


class ColumnClassifier(ClassifierMixin, BaseEstimator):
    """Predicts the class stored in a column of X, with probability `confidence` above uniform."""

    def __init__(self, column=0, confidence=1.):
        self.column = column
        self.confidence = confidence

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.asarray(X)[:, self.column].astype(self.classes_.dtype)

    def predict_proba(self, X):
        one_hot = (self.predict(X).reshape(-1, 1) == self.classes_).astype(float)
        return self.confidence * one_hot + (1 - self.confidence) / len(self.classes_)


class ColumnClassifierNoProba(ClassifierMixin, BaseEstimator):
    """As ColumnClassifier, without `predict_proba` (e.g. LinearSVC)."""

    def __init__(self, column=0):
        self.column = column

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.asarray(X)[:, self.column].astype(self.classes_.dtype)
