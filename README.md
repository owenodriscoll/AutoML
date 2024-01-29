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
* multithreading
* feature-importance analyses with `shap`


## Setup
### Method 1: pip install
Create a new environment to prevent pip install from breaking anything. Include a Python version 3.11
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

### Method 2: cloning
Clone the repository
```
git clone https://github.com/owenodriscoll/AutoML
```

Navigate to the cloned local repository and create the conda environment with all requirement packages
```
conda env create --name ENVNAME --file environment.yml
```

Activate new environment
```
conda activate ENVNAME
```

Having created an environment with all dependencies, install AutoML:
```
pip install git+https://github.com/owenodriscoll/AutoML.git
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
    timeout=100
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
    timeout=600,
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
