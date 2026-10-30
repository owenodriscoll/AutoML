[![PyPI version](https://img.shields.io/pypi/v/py-automl-lib.svg?color=4c1)](https://pypi.org/project/py-automl-lib/)

# automl: Automated Machine Learning
## Intro
automl is a python project focussed on automating much of the machine learning efforts encountered in zero-dimensional regression and classification (and thus not multidimensional data such as for a CNN). It relies on existing Python packages Sci-Kit Learn, Optuna and model specific packages LightGBM, CatBoost and XGBoost.

automl works by assessing the performance of various machine-learning models for a set number of trials over a pre-defined range of hyperparameters. During succesive trials the hyperparameters are optimized following a user-defined methodology (the default optimisation uses Bayesian search). Unpromising trials are stopped (pruned) early by assessing performance on an incrementally increasing fraction of training data, saving computational resources. Hyperparameter optimization trials are stored locally on disk, allowing the training to be picked up after interuption. The best trials of the defined models are reloaded and combined, or stacked, to form a final model. This final model is assessed and, due to the nature of stacking, tends to outperform any of its constituting models.

automl contains several additional functionalities beyond the hyperoptimization and stacking of models: 
* scaling of the input `X`-matrix (tested for on default)
* normal transformation of the `y`-matrix (tested for on default)
* PCA compression
* spline transformation
* polynomial expansion
* categorical feature support (nominal and ordinal)
* bagging of weak models in addition to optimized models
* custom loss functions, carried through optimization, early stopping, training, stacking and evaluation
* group-level losses for multi-index data (e.g. targets only valid on a coarser scale than the observations)
* multithreading
* feature-importance analyses with `shap`


## Installation
Create a new environment to prevent pip install from breaking anything. Include Python version 3.11
```
conda create -n ENVNAME -c conda-forge python=3.11
```

Activate new environment
```
conda activate ENVNAME
```

Pip install 
```
python3 -m pip install py-automl-lib
```

Optionally include the `shap` package for feature-importance analyses (see `example_notebook.ipynb` chapter 7.)
```
python3 -m pip install py-automl-lib[shap]
```


## Use

For a more detailed example checkout `examples/example_notebook.ipynb`

Minimal use case regression:
```python
from sklearn.metrics import r2_score
from automl import AutomatedRegression

X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, random_state=42)

regression = AutomatedRegression(
    y=y,
    X=X,
    n_trial=10,
    timeout_study=100
    metric_optimise=r2_score,
    optimisation_direction='maximize',
    models_to_optimize=['bayesianridge', 'lightgbm'],
    )
    
regression.apply()
regression.summary
```

Expanded options use case regression:
```python
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from automl import AutomatedRegression

X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, random_state=42)

# -- adding categorical features
df_X = pd.DataFrame(X)
df_X['nine'] = pd.cut(df_X[9], bins=[-float('Inf'), -3, -1, 1, 3, float('Inf')], labels=['a', 'b', 'c', 'd', 'e'])
df_X['ten'] = pd.cut(df_X[9], bins=[-float('Inf'), -1, 1, float('Inf')], labels=['A', 'B', 'C'])
df_y = pd.Series(y)

regression = AutomatedRegression(
    y=df_y,
    X=df_X,
    test_frac=0.2,
    fit_frac=[0.2, 0.4, 0.6, 1],
    n_trial=50,
    timeout_study=600,
    timeout_trial=120,
    metric_optimise=r2_score,
    optimisation_direction='maximize',
    cross_validation=KFold(n_splits=5, shuffle=True, random_state=42),
    sampler=TPESampler(seed=random_state),
    pruner=HyperbandPruner(min_resource=1, max_resource='auto', reduction_factor=3),
    reload_study=False,
    reload_trial_cap=False,
    write_folder='/auto_regression_test',
    models_to_optimize=['bayesianridge', 'lightgbm'],
    nominal_columns=['nine'],
    ordinal_columns=['ten'],
    pca_value=0.95,
    spline_value={'n_knots': 5, 'degree':3},
    poly_value={'degree': 2, 'interaction_only': True},
    boosted_early_stopping_rounds=100,
    n_weak_models=5,
    random_state=42,
    )

regression.apply()
regression.summary
    
```

Failures during optimization do not stop the run: a failing trial or model is skipped and reported with an `OptimisationFailureWarning`, at any `warning_verbosity`. If no model remains, `summary` stays empty.

## Testing

Install the test dependencies and run the suite from the root of the repository:
```
python3 -m pip install -e ".[test]"
pytest                  # everything, a few minutes
pytest -m "not slow"    # quick subset, excludes exhaustive tests of all models and settings
pytest --cov            # with coverage, fails below the threshold in pyproject.toml
pytest -rx              # lists the known bugs
```

The same suite runs on GitHub Actions for pushes to `main` and `dev`, for pull requests, and weekly, as dependencies are not pinned and new releases can break the package without any change here.

Known bugs are recorded as `@pytest.mark.xfail(reason="BUG: ...")` tests describing the correct behaviour. These are strict: after fixing a bug its test fails as `XPASS(strict)` until the marker is removed.

## Custom losses

The loss passed as `metric_optimise` is used in every stage of the learning:

| Stage | How the loss is used |
|---|---|
| Hyperparameter optimization | Scores the cross-validation folds |
| Early stopping (`lightgbm`, `xgboost`, `catboost`) | Evaluation metric on the validation fold |
| Training of boosted regression models | Training objective, from its gradient and hessian |
| Stacking (`stacking_meta_learner='loss'`) | Non-negative weights of the models are chosen to optimise the loss |
| Evaluation | Always reported, as last metric in `summary` |

A plain metric `(y_true, y_pred) -> float` suffices for all stages but training. For training, the boosted models need the gradient and hessian of the loss. These are known for common metrics (e.g. `r2_score` and `mean_squared_error` train on squared error, `mean_absolute_error` on absolute error). For other metrics, supply them with `CustomLoss`; otherwise the models train on their default objective and a `LossFallbackWarning` is issued.

```python
import numpy as np
from automl import AutomatedRegression, CustomLoss

def ratio_error(y_true, y_pred):
    """Squared error of the ratio of two targets"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean((y_true[:, 0] / y_true[:, 1] - y_pred[:, 0] / y_pred[:, 1]) ** 2)

def ratio_error_grad_hess(y_true, y_pred):
    """Gradient and diagonal hessian w.r.t. y_pred, of the summed loss (LightGBM/XGBoost convention)"""
    r_true, r_pred = y_true[:, 0] / y_true[:, 1], y_pred[:, 0] / y_pred[:, 1]
    d_ratio = np.c_[1 / y_pred[:, 1], -y_pred[:, 0] / y_pred[:, 1] ** 2]
    return 2 * (r_pred - r_true)[:, None] * d_ratio, 2 * d_ratio ** 2

regression = AutomatedRegression(
    y=df_y,  # two target columns
    X=df_X,
    metric_optimise=CustomLoss(ratio_error, grad_hess=ratio_error_grad_hess),
    optimisation_direction='minimize',
    models_to_optimize=['xgboost', 'bayesianridge'],
    )
```

For classification, set `CustomLoss(metric, needs_proba=True)` for metrics requiring class probabilities (e.g. `log_loss`). Classifiers always train on their default objective.

Limitations:
* only `lightgbm`, `xgboost` and `catboost` train on custom objectives; other models still use the loss for optimization and stacking
* for multi-output, only `xgboost` (fitted on all targets jointly) trains on a loss combining targets; models wrapped per target fall back
* `catboost` evaluates custom objectives on chunks of rows, so its `grad_hess` must be element-wise and group-level objectives fall back

### Group-level losses

With `group_by_index_columns`, targets and predictions are aggregated per group (with `agg_func`) before the loss is calculated, in all stages above. The models are thereby constrained to fit the target on the scale of the groups, while deviations within groups are only constrained by regularisation. Training on group-level gradients requires `agg_func` to be `'mean'`/`np.nanmean` or `'sum'`. The train-test split and the cross-validation (`GroupKFold` by default) keep groups together, preventing leakage between training and validation data.

```python
regression = AutomatedRegression(
    y=df_y,  # multi-index ['safe', 'group', 'index']
    X=df_X,
    metric_optimise=r2_score,
    optimisation_direction='maximize',
    group_by_index_columns=['safe'],
    agg_func='mean',
    models_to_optimize=['lightgbm', 'xgboost', 'bayesianridge'],
    )
```
