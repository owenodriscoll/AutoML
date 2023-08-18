import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import r2_score, accuracy_score, precision_score
# If AutoML is not installed, uncomment below to lines to allow for correct import
# import os
# os.chdir("..")
from AutoML import AutomatedRegression, AutomatedClassification

# %reset -f

#%% Regression

X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=42)


df_X = pd.DataFrame(X)
df_X['nine'] = pd.cut(df_X[9], bins=[-float('Inf'), -3, -1, 1, 3, float('Inf')], labels=['a', 'b', 'c', 'd', 'e'])
df_X['ten'] = pd.cut(df_X[9], bins=[-float('Inf'), -1, 1, float('Inf')], labels=['A', 'B', 'C'])
df_y = pd.Series(y)


# prepare Autoregression object
regression = AutomatedRegression(
    y=df_y,
    X=df_X,
    # pca_value=0.95,
    # spline_value= 2,
    # poly_value={'degree': 2, 'interaction_only': True},
    n_trial=10,
    nominal_columns=['nine'],
    ordinal_columns=['ten'],
    reload_study=True,
    reload_trial_cap=False,
    write_folder=f'{os.path.dirname(__file__)}/testdir',
    metric_optimise=r2_score,
    optimisation_direction='maximize',
    models_to_optimize=['bayesianridge', 'lightgbm', 'lassolars', 'xgboost', 'catboost'],
    models_to_assess=[ 'lassolars', 'lightgbm','bayesianridge', 'xgboost', 'catboost'],
    boosted_early_stopping_rounds = 20,
    n_weak_models=5
    )

regression.apply()
regression.summary

# regression.model_select_best()
# regression.model_evaluate()

#%% Classification


X, y = make_classification(
    n_samples=1000, n_features=15, n_redundant=0, n_informative=10, random_state=42, n_classes = 3, n_clusters_per_class=1
)
classification = AutomatedClassification(
    y=y,
    X=X,
    # pca_value=0.95,
    # spline_value= 2,
    # poly_value={'degree': 2, 'interaction_only': True},
    n_trial=30,
    # nominal_columns=['nine'],
    # ordinal_columns=['ten'],
    reload_study=True,
    reload_trial_cap=True,
    write_folder='/export/home/owen/Documents/scripts/AutoML/tests/auto_classification0',
    metric_assess=[lambda y_pred, y_true: precision_score(y_pred, y_true, average = 'macro')],
    optimisation_direction='maximize',
    models_to_optimize=['sgd', 'lightgbm', 'svc'],
    models_to_assess=[ 'svc', 'lightgbm','sgd'],
    boosted_early_stopping_rounds = 20,
    n_weak_models=5
    )

classification.apply()
classification.summary


# adapt metrics to classification to apply per class, weighted, binary etc, maybe using scorer