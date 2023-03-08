import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import r2_score, accuracy_score, precision_score
from AutoML.AutoML import AutomatedRegression, AutomatedClassification

#%% Regression

X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, random_state=42)

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
    nominal_columns= ['nine'],
    ordinal_columns= ['ten'],
    overwrite=True,
    write_folder = '/export/home/owen/Documents/test/regression/',
    metric_optimise=r2_score,
    optimisation_direction='maximize',
    list_regressors_optimise=['lightgbm'], #, 'xgboost', 'catboost']
    boosted_early_stopping_rounds = 20,
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
    y=pd.DataFrame(y),
    X=pd.DataFrame(X),
    # pca_value=0.95,
    # spline_value=None,
    # poly_value= 2,
    n_trial=10,
    overwrite=True,
    write_folder = '/export/home/owen/Documents/test/classification/',
    metric_optimise = accuracy_score,
    metric_assess = [lambda y_pred, y_true: precision_score(y_pred, y_true, average = 'macro')],
    optimisation_direction='maximize',
    list_classifiers_optimise = ['histgradientboost'], #'lightgbm', 'xgboost', 'catboost'
    list_classifiers_assess = ['histgradientboost'],
    boosted_early_stopping_rounds = 20,
    )

classification.apply()
classification.summary


# n_classes should be computed from training data only
# adapt metrics to classification to apply per class, weighted, binary etc, maybe using scorer