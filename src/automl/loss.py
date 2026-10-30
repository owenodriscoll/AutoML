"""
Loss handling for AutomatedML.

A single (custom) loss is carried through all stages of the learning:
    - hyperparameter optimisation (scoring of the cross-validation folds)
    - early stopping of boosted models (as evaluation metric)
    - training of boosted regression models (as training objective, via gradient and hessian)
    - stacking (the meta-learner directly optimises the loss)
    - evaluation (the loss is always reported in the summary)

When `group_by_index_columns` is set, the loss is evaluated on per-group aggregates of the targets and
predictions in every one of these stages.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, List

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, explained_variance_score, \
    mean_absolute_error, median_absolute_error

from .utils.misc import assert_not_lambda

boosted_models = ['lightgbm', 'xgboost', 'catboost']

# -- parameter names under which the boosted libraries accept a training objective
objective_param_names = {'lightgbm': 'objective', 'xgboost': 'objective', 'catboost': 'loss_function'}


class LossFallbackWarning(UserWarning):
    """Issued when a custom loss cannot be used as training objective and a default objective is used instead."""


@dataclass
class CustomLoss:
    """
    Custom loss used throughout hyperparameter optimisation, early stopping, training, stacking and evaluation.

    Parameters
    ----------
    metric: callable
        Function (y_true, y_pred) -> float. Its optimisation direction is set by `optimisation_direction`
        of AutomatedML. When grouping is applied, the metric receives per-group aggregates.
        NOTE this metric must be pickle-able (lambda functions are non-pickleable)
    grad_hess: callable, optional (default=None)
        Function (y_true, y_pred) -> (grad, hess) returning the gradient and the diagonal of the hessian of
        the loss to MINIMISE with respect to y_pred, as numpy arrays of the same shape as y_pred.
        Follows the convention of custom objectives in LightGBM and XGBoost, i.e. the derivatives of a SUM of
        losses (for a mean-type loss, multiply the derivatives by the number of samples).
        When grouping is applied, y_true and y_pred are per-group aggregates and the chain rule to individual
        rows is applied internally. For multi-output, y_true and y_pred have shape (n_samples, n_targets).
        Only used to train boosted regression models ('lightgbm', 'xgboost', 'catboost'). CatBoost evaluates
        custom objectives on chunks of rows, so for CatBoost grad_hess must be element-wise (each row's derivatives
        only depend on that row) and group-level objectives are not available. If None, a native
        objective equivalent to `metric` is used when known (e.g. squared error for r2_score), else the model
        default objective is used and a LossFallbackWarning is issued.
    needs_proba: bool, optional (default=False)
        Classification only. Whether `metric` requires class probabilities (e.g. log_loss, roc_auc_score)
        instead of class labels. For binary classification the probability of the positive class is provided.
        Models without `predict_proba` (e.g. 'svc') cannot be scored with such a metric and their trials fail.
    name: str, optional
        Name of the loss as reported in the summary, defaults to the name of `metric`.
    """

    metric: Callable
    grad_hess: Callable | None = None
    needs_proba: bool = False
    name: str | None = None

    def __post_init__(self):
        assert_not_lambda(self.metric)
        if self.grad_hess is not None:
            assert_not_lambda(self.grad_hess)
        if self.name is None:
            self.name = getattr(self.metric, '__name__', 'custom_loss')


# -- known metrics for which an equivalent training objective exists, used when no grad_hess is provided
def squared_error_grad_hess(y_true, y_pred):
    return y_pred - y_true, np.ones_like(y_pred)


def absolute_error_grad_hess(y_true, y_pred):
    return np.sign(y_pred - y_true), np.ones_like(y_pred)


known_objectives = {
    'squared_error': {
        'metrics': [mean_squared_error, root_mean_squared_error, r2_score, explained_variance_score],
        'grad_hess': squared_error_grad_hess,
        'native': {'lightgbm': 'regression', 'xgboost': 'reg:squarederror', 'catboost': 'RMSE'},
    },
    'absolute_error': {
        'metrics': [mean_absolute_error, median_absolute_error],
        'grad_hess': absolute_error_grad_hess,
        'native': {'lightgbm': 'regression_l1', 'xgboost': 'reg:absoluteerror', 'catboost': 'MAE'},
    },
}


def find_known_objective(metric: Callable) -> dict | None:
    for known in known_objectives.values():
        for known_metric in known['metrics']:
            same_function = metric is known_metric
            same_sklearn_name = (getattr(metric, '__module__', '') or '').startswith('sklearn.metrics') and \
                getattr(metric, '__name__', None) == known_metric.__name__
            if same_function or same_sklearn_name:
                return known
    return None


def linear_aggregation(agg_func) -> str | None:
    """Returns 'mean' or 'sum' if the aggregation is linear (allowing the chain rule to be applied), else None."""
    if agg_func in ('mean', np.mean, np.nanmean):
        return 'mean'
    if agg_func in ('sum', np.sum, np.nansum):
        return 'sum'
    return None


@dataclass
class TrainingObjective:
    """Resolved training objective of a single model."""
    kind: str = 'default'               # 'default', 'native' or 'custom'
    native: str | None = None           # name of native objective when kind == 'native'
    grad_hess: Callable | None = None   # gradient and hessian when kind == 'custom'
    native_multioutput: bool = False    # whether the model is fitted on all targets jointly (no per-target wrapper)


class LossHandler:
    """
    Internal object combining the CustomLoss with the optimisation direction and grouping settings.
    It scores predictions and configures estimators to train and early-stop on the loss.
    """

    def __init__(self, loss: CustomLoss, greater_is_better: bool, group_by_index_columns: List[str] | None = None,
                 agg_func: Callable | str = 'mean', ml_objective: str = 'regression', is_multioutput: bool = False):
        self.loss = loss
        self.greater_is_better = greater_is_better
        self.group_by_index_columns = group_by_index_columns if isinstance(group_by_index_columns, list) else None
        self.agg_func = agg_func
        self.ml_objective = ml_objective
        self.is_multioutput = is_multioutput
        self.objectives = {}

    @property
    def sign(self) -> int:
        """Multiplier converting the metric to a value to minimise."""
        return -1 if self.greater_is_better else 1

    @property
    def is_grouped(self) -> bool:
        return self.group_by_index_columns is not None

    def group_labels(self, index) -> np.ndarray | None:
        """Integer label per row identifying its group, None if no grouping is applied."""
        if not self.is_grouped or index is None:
            return None
        frame = index.to_frame(index=False)[self.group_by_index_columns]
        return frame.groupby(self.group_by_index_columns, sort=False).ngroup().to_numpy()

    def aggregate(self, y_true, y_pred):
        """Aggregates targets and predictions per group (if grouping applies) prior to calculating metrics."""
        if not self.is_grouped:
            return y_true, y_pred
        y_true_eval = y_true.groupby(level=self.group_by_index_columns).aggregate(self.agg_func)
        y_pred_eval = pd.DataFrame(np.asarray(y_pred), index=y_true.index).groupby(
            level=self.group_by_index_columns).aggregate(self.agg_func)
        return y_true_eval, y_pred_eval

    def score(self, y_true, y_pred) -> float:
        return self.loss.metric(*self.aggregate(y_true, y_pred))

    @staticmethod
    def proba_output(proba: np.ndarray) -> np.ndarray:
        """For binary classification only the probability of the positive class is provided to the metric."""
        proba = np.asarray(proba)
        return proba[:, 1] if (proba.ndim == 2 and proba.shape[1] == 2) else proba

    def predict(self, estimator, X):
        """Prediction in the form required by the loss (class probabilities or values/labels)."""
        if self.loss.needs_proba:
            return self.proba_output(estimator.predict_proba(X))
        return estimator.predict(X)

    # ----------------------------------------- training objectives -----------------------------------------
    def resolve_objectives(self, model_names: List[str]) -> dict:
        """
        Determines per model which training objective is used and warns (once) where the custom loss
        cannot be used for training and a default objective is used instead.
        """
        self.objectives = {name: TrainingObjective() for name in model_names}
        boosted = [name for name in model_names if name in boosted_models]
        name = self.loss.name

        if self.ml_objective != 'regression':
            if self.loss.grad_hess is not None:
                self._warn(f"Custom training objectives (`grad_hess`) are only supported for regression. "
                           f"Classifiers are trained on their default objective; loss '{name}' is still used for "
                           "hyperparameter optimisation, early stopping, stacking and evaluation.")
            return self.objectives

        not_boosted = [model for model in model_names if model not in boosted_models]
        if (self.loss.grad_hess is not None) and not_boosted:
            self._warn(f"Models {not_boosted} do not support custom training objectives and are trained on their "
                       f"built-in loss. Loss '{name}' is still used for their hyperparameter optimisation.")

        known = find_known_objective(self.loss.metric)
        linear_agg = linear_aggregation(self.agg_func)
        fallbacks = {}

        for model in boosted:
            per_target_wrapper = self.is_multioutput and model != 'xgboost'

            # -- CatBoost evaluates custom objectives on arbitrary chunks of rows, which may split groups
            group_level = self.is_grouped and linear_agg and model != 'catboost'
            group_level_impossible = self.is_grouped and not group_level

            # -- 1. user supplied gradient and hessian
            if self.loss.grad_hess is not None and not per_target_wrapper and not group_level_impossible:
                self.objectives[model] = TrainingObjective('custom', grad_hess=self.loss.grad_hess,
                                                           native_multioutput=self.is_multioutput)
                continue

            # -- 2. known equivalent objective, group-level via the chain rule when grouping, else native
            if known is not None and group_level:
                self.objectives[model] = TrainingObjective('custom', grad_hess=known['grad_hess'])
            elif known is not None:
                self.objectives[model] = TrainingObjective('native', native=known['native'][model])

            # -- 3. fall back on the default objective
            if self.loss.grad_hess is not None and per_target_wrapper:
                reason = "multi-output models wrapped per target cannot train on a joint `grad_hess`"
            elif group_level_impossible and not linear_agg and (self.loss.grad_hess is not None or known is not None):
                reason = f"group-level gradients require `agg_func` 'mean' or 'sum', not {self.agg_func!r}"
            elif group_level_impossible and (self.loss.grad_hess is not None or known is not None):
                reason = "CatBoost evaluates custom objectives on chunks of rows, so group-level gradients are unavailable"
            elif self.loss.grad_hess is None and known is None:
                reason = f"metric '{name}' has no known equivalent training objective and no `grad_hess` was supplied"
            else:
                continue
            fallbacks.setdefault(reason, []).append(model)

        for reason, models in fallbacks.items():
            used = ["row-level native objective" if self.objectives[m].kind == 'native' else "default objective"
                    for m in models]
            self._warn(f"Loss '{name}' cannot be used as training objective for {models} ({reason}). "
                       f"Using instead: {dict(zip(models, used))}. The loss is still used for hyperparameter "
                       "optimisation, early stopping, stacking and evaluation.")

        return self.objectives

    @staticmethod
    def _warn(message):
        warnings.warn(message, LossFallbackWarning, stacklevel=3)

    def uses_custom_objective(self, model_name: str) -> bool:
        return self.objectives.get(model_name, TrainingObjective()).kind == 'custom'

    def needs_per_target_wrapper(self, model_name: str, model_without_native_multioutput: bool) -> bool:
        objective = self.objectives.get(model_name, TrainingObjective())
        return self.is_multioutput and model_without_native_multioutput and not objective.native_multioutput

    def configure_estimator(self, estimator, model_name: str, early_stopping: bool = False):
        """
        Sets the training objective and (for early stopping) the evaluation metric of boosted models.

        Returns
        -------
        estimator, dict of fit parameters to pass to the estimator's fit method
        """
        fit_params = {}
        if model_name not in boosted_models:
            return estimator, fit_params

        # -- XGBoost otherwise prints the evaluation of every boosting round
        if early_stopping and model_name == 'xgboost':
            fit_params['verbose'] = False

        params = {}
        objective = self.objectives.get(model_name, TrainingObjective())
        if objective.kind == 'native':
            params[objective_param_names[model_name]] = objective.native
        elif objective.kind == 'custom':
            params[objective_param_names[model_name]] = GradHessObjective(self, objective.grad_hess)

        # -- a joint loss cannot evaluate a single target, so per-target wrapped models early-stop on their default
        per_target_wrapper = self.is_multioutput and not objective.native_multioutput
        if early_stopping and not per_target_wrapper:
            if model_name == 'lightgbm':
                # -- LightGBM otherwise also evaluates its built-in metric, and stops on whichever stalls first
                params['metric'] = 'None'
                fit_params['eval_metric'] = LightGBMEvalMetric(self)
            elif model_name == 'xgboost':
                params['eval_metric'] = XGBoostEvalMetric(self)
            elif model_name == 'catboost':
                params['eval_metric'] = CatBoostEvalMetric(self)

        # -- CatBoost requires an evaluation metric with a custom objective, only reported without early stopping
        elif model_name == 'catboost' and objective.kind == 'custom':
            params['eval_metric'] = 'RMSE'

        return estimator.set_params(**params), fit_params


# ------------------------------------------------ adapters ------------------------------------------------
# Adapters translate the loss to the interfaces of the boosted libraries. Before fitting, they are bound to
# the row index of the data (for group-level losses), the fitted target transformer (to evaluate the loss on
# the original scale) and the classes (to translate encoded labels back), see estimators.LossAwarePipeline.

class LossAdapter:
    def __init__(self, handler: LossHandler):
        self.handler = handler
        self.index = None
        self.codes = None
        self.target_transformer = None
        self.classes = None
        self.__name__ = handler.loss.name  # xgboost names callables by their __name__

    def __deepcopy__(self, memo):
        # -- adapters are shared between clones of an estimator (sklearn's clone deep-copies parameters and
        # CatBoost deep-copies them in get_params). This keeps the binding visible to the estimator actually
        # being fitted; binding happens right before each fit, so sequential fits of clones are unaffected
        return self

    def bind(self, index=None, classes=None):
        self.index = index
        self.codes = self.handler.group_labels(index)
        self.classes = classes
        return self

    def _original_scale(self, y: np.ndarray) -> np.ndarray:
        if self.target_transformer is None:
            return y
        return self.target_transformer.inverse_transform(y.reshape(len(y), -1)).reshape(y.shape)


class GradHessObjective(LossAdapter):
    """
    Training objective computing gradient and hessian of the (group-level) loss for each row.

    With G_g and H_g the derivatives of the loss with respect to group aggregate a_g, the per-row derivatives
    follow from the chain rule:
        'mean': grad_i = G_g,  hess_i = H_g
        'sum':  grad_i = G_g,  hess_i = n_g * H_g
    such that a leaf containing an entire group takes exactly the Newton step of the group-level loss.
    For 'mean', the group losses are thereby weighted by the group size n_g, keeping the derivatives on the
    scale of row-level objectives, so regularisation hyperparameters (e.g. min_child_weight) behave as usual.
    NOTE only shift-type target transformers (TargetCenterer) are allowed, which leave derivatives unchanged.
    """

    def __init__(self, handler: LossHandler, grad_hess: Callable):
        super().__init__(handler)
        self.grad_hess = grad_hess

    def gradient(self, y_true, y_pred):
        y_true = self._original_scale(np.asarray(y_true, dtype=float))
        y_pred = self._original_scale(np.asarray(y_pred, dtype=float))

        if self.codes is None:
            grad, hess = self.grad_hess(y_true, y_pred)
            return np.asarray(grad, dtype=float), np.asarray(hess, dtype=float)

        if len(self.codes) != len(y_true):
            raise ValueError(f"Group labels ({len(self.codes)}) do not match number of samples ({len(y_true)})")

        is_mean = linear_aggregation(self.handler.agg_func) == 'mean'
        n_group = np.bincount(self.codes).astype(float).reshape(-1, *([1] * (y_true.ndim - 1)))
        divisor = n_group if is_mean else 1.

        grad_group, hess_group = self.grad_hess(group_sum(y_true, self.codes) / divisor,
                                                group_sum(y_pred, self.codes) / divisor)
        hess_group = np.asarray(hess_group, dtype=float) * (1. if is_mean else n_group)
        return np.asarray(grad_group, dtype=float)[self.codes], hess_group[self.codes]

    # -- LightGBM and XGBoost interface
    def __call__(self, y_true, y_pred):
        return self.gradient(y_true, y_pred)

    # -- CatBoost interface, CatBoost expects derivatives of the quantity to MAXIMISE
    def calc_ders_range(self, approxes, targets, weights):
        grad, hess = self.gradient(np.asarray(targets), np.asarray(approxes))
        return list(zip(-grad, -hess))


def group_sum(a: np.ndarray, codes: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return np.bincount(codes, weights=a)
    return np.stack([np.bincount(codes, weights=a[:, i]) for i in range(a.shape[1])], axis=1)


class EvalMetric(LossAdapter):
    """Evaluates the loss on the evaluation set during training, used for early stopping."""

    def score(self, y_true, y_pred) -> float:
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

        # -- e.g. CatBoost also evaluates custom metrics on the training data, which is not bound
        if self.index is not None and len(self.index) != len(y_true):
            return np.nan

        if self.handler.ml_objective == 'regression':
            y_true = self._original_scale(y_true.astype(float))
            y_pred = self._original_scale(y_pred.astype(float))
        else:
            # -- the libraries provide label-encoded targets, translate back to the original classes
            classes = self.classes if self.classes is not None else np.arange(max([2, *np.shape(y_pred)[1:]]))
            y_true = classes[y_true.astype(int)]
            if not self.handler.loss.needs_proba:
                labels = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else (y_pred > 0.5).astype(int)
                y_pred = classes[labels]

        return self.handler.score(pd.DataFrame(y_true, index=self.index), y_pred)


class LightGBMEvalMetric(EvalMetric):
    def __call__(self, y_true, y_pred):
        return self.handler.loss.name, self.score(y_true, y_pred), self.handler.greater_is_better


class XGBoostEvalMetric(EvalMetric):
    # -- XGBoost minimises custom metrics in its sklearn interface
    def __call__(self, y_true, y_pred, sample_weight=None):
        return self.handler.sign * self.score(y_true, y_pred)


class CatBoostEvalMetric(EvalMetric):
    def is_max_optimal(self):
        return self.handler.greater_is_better

    def evaluate(self, approxes, target, weight):
        approx = np.asarray(approxes, dtype=float)

        # -- CatBoost provides raw scores for classification
        if self.handler.ml_objective == 'classification':
            if approx.shape[0] == 1:
                approx = 1 / (1 + np.exp(-approx[0]))
            else:
                exp = np.exp(approx - approx.max(axis=0))
                approx = (exp / exp.sum(axis=0)).T
        else:
            approx = approx[0] if approx.shape[0] == 1 else approx.T

        value = self.score(np.asarray(target), approx)
        return (0.0 if np.isnan(value) else value), 1.0

    def get_final_error(self, error, weight):
        return error


class TargetCenterer(TransformerMixin, BaseEstimator):
    """
    Subtracts the per-target mean. Used instead of the quantile transformer when boosted models train on a
    custom objective: custom objectives start boosting from zero (rather than from the target mean) and,
    unlike non-linear transformations, a shift leaves the gradient and hessian of the loss unchanged.
    """

    def fit(self, X, y=None):
        self.mean_ = np.mean(np.asarray(X, dtype=float), axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) - self.mean_

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float) + self.mean_
