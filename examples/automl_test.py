import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from AutoML.AutoML import AutomatedRegression

X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, random_state=42)

df_X = pd.DataFrame(X)
df_X[0] = 1
df_X['nine'] = pd.cut(df_X[9], bins=[-float('Inf'), -3, -1, 1, 3, float('Inf')], labels=['a', 'b', 'c', 'd', 'e'])
df_X['ten'] = pd.cut(df_X[9], bins=[-float('Inf'), -1, 1, float('Inf')], labels=['A', 'B', 'C'])
df_y = pd.Series(y)


# prepare Autoregression object
test = AutomatedRegression(
    y=df_y,
    X=df_X,
    # pca_value=0.95,
    # spline_value= 2,
    # poly_value={'degree': 2, 'interaction_only': True},
    n_trial=20,
    nominal_columns= ['nine'],
    ordinal_columns= ['ten'],
    write_folder='/export/home/owen/Documents/scripts/AutoML/tests/auto_regression0',
    reload_study=False, 
    reload_trial_cap=False,
    metric_optimise=r2_score,
    optimisation_direction='maximize',
    boosted_early_stopping_rounds=20,
    list_regressors_optimise=['lightgbm', 'lassolars', 'bayesianridge'],
    list_regressors_assess=['lightgbm', 'lassolars', 'bayesianridge'],
    )

test.apply()
# test.split_train_test()
# test.regression_hyperoptimise()
# test.summary

