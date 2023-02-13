import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from AutoML.AutoML import AutomatedRegression

X, y = make_regression(n_samples=1000, n_features=20, n_informative=5, random_state=42)

input_dict = dict(y=pd.DataFrame(y),
                  X=pd.DataFrame(X),
                  pca_value=0.95,
                  spline_value=None,
                  poly_value= 2,
                  n_trial=4,
                  write_folder = '/export/home/owen/Documents/scripts/', 
                  overwrite=True,
                  metric_optimise=r2_score,
                  optimisation_direction='maximize',
                  list_regressors_optimise=['lightgbm'])

test = AutomatedRegression(**input_dict)

test.apply()
test.regression_select_best()
test.regression_evaluate()
test.summary