# %%

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from AutoML.AutoML import AutomatedRegression

X, y = make_regression(n_samples=2000, n_features=10, n_informative=5, random_state=42)

input_dict = dict(y=pd.DataFrame(y),
                  X=pd.DataFrame(X),
                  pca_value=None,
                  spline_value=None,
                  poly_value=None,
                  n_trial=30,
                  overwrite=True,
                  metric_optimise=r2_score,
                  optimisation_direction='maximize',
                  list_regressors_optimise=['lightgbm', 'lassolars'])

test = AutomatedRegression(**input_dict)
test.apply()
