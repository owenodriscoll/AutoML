# auto_ML_custom
Package to call for automated regression. The `automated_regression` function performs several operations in one:
* Optimisation of specified regression models
* Optimisation of various X-scalers and y-transformers
* Optionally optimises the inclusion of PCA compression, spline transformer or a polynomial expansion of X

```python
import sklearn
import pandas as pd
import auto_ML_custom as autoML
from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()
X_full, y_full = dataset.data, dataset.target
y = pd.DataFrame(y_full)
X = pd.DataFrame(X_full)

metric_performance_summary_dict, idexes_test_kfold, y_pred, y_test = autoML.automated_regression(
    y = y, X = X, test_frac = 0.2, timeout = 600, n_trial = 100, 
    metric_optimise = sklearn.metrics.mean_pinball_loss,  metric_assess = [sklearn.metrics.mean_pinball_loss, sklearn.metrics.mean_squared_error, sklearn.metrics.r2_score],
    optimisation_direction = 'minimize',  overwrite = True, 
    list_regressors_hyper = ['lightgbm', 'bayesianridge'], list_regressors_training = None, 
    random_state = 42)

```
