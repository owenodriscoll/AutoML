# auto_ML_custom
## Contents
- `automated_ML.py`: file containing automised machine learning algorithms 
- `example_automated_ML.ipynb`: example notebook for regression on California house price dataset
- `optuna_visualisation.py`: python script to visualise intermediate optimisation REQUIRES UPDATE
- `auto_ML_env.yml.`: environment file containing all the necessary packages (along with a LOT of clutter, best to manually download missing packages)



## automated_ML.py
Package to call for automated regression. The `automated_regression` function performs several operations in one:
* Splits data into training and testing fractions
* Optimisation of specified regression models following specified metrics
* Optimisation of various X-scalers and y-transformers
* Optionally optimises the inclusion of PCA compression, spline transformer or a polynomial expansion of X
* Stacks optimized regressors using a final ridge regression
* Returns the performance metrics per regressor (including final stacked regressor) on test dataset

Note: 
- Recommended to select a unique `write_folder` to store intermediate optimisation progress, otherwise each run will generate (or overwrite) previous optimisation.
- Function is designed for continuous data and regression only, not time series or classifications.

### Example
For a more detailed example checkout `example_automated_ML.ipynb`

```python
import pandas as pd
import sklearn
from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()
X_full, y_full = dataset.data, dataset.target
y = pd.DataFrame(y_full)
X = pd.DataFrame(X_full)

metric_performance_summary_dict, idexes_test_kfold, test_index, train_index, y_pred, y_test = automated_regression(
            y = y, X = X, test_frac = 0.2, timeout = 600, n_trial = 100, 
            metric_optimise = sklearn.metrics.mean_pinball_loss,  metric_assess = [sklearn.metrics.mean_pinball_loss, sklearn.metrics.mean_squared_error, sklearn.metrics.r2_score],
            optimisation_direction = 'minimize',  overwrite = True, 
            list_regressors_hyper = ['lightgbm', 'bayesianridge'], list_regressors_training = None, 
            random_state = 42)

```

## optuna_visualisation.py
insert

## to do
* add classification
* add ability to restart training
* add add classes?
