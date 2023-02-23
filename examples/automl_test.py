import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import r2_score, accuracy_score, make_scorer, precision_score
from AutoML.AutoML import AutomatedRegression, AutomatedClassification

# X, y = make_regression(n_samples=1000, n_features=20, n_informative=5, random_state=42)

# regression = AutomatedRegression(
#     y=pd.DataFrame(y),
#     X=pd.DataFrame(X),
#     pca_value=0.95,
#     spline_value=None,
#     poly_value= 2,
#     n_trial=4,
#     write_folder = '/export/home/owen/Documents/scripts/', 
#     overwrite=True,
#     metric_optimise=r2_score,
#     optimisation_direction='maximize',
#     list_regressors_optimise=['lightgbm']
#     )

# # regression.apply()
# regression.split_train_test()
# regression.model_hyperoptimise()
# regression.model_select_best()
# regression.model_evaluate()
# regression.summary


X, y = make_classification(
    n_samples=1000, n_features=15, n_redundant=0, n_informative=10, random_state=42, n_classes = 3, n_clusters_per_class=1
)

classification = AutomatedClassification(
    y=pd.DataFrame(y),
    X=pd.DataFrame(X),
    pca_value=0.95,
    spline_value=None,
    # poly_value= 2,
    n_trial=40,
    write_folder = '/export/home/owen/Documents/scripts/', 
    overwrite=True,
    metric_optimise= accuracy_score,
    metric_assess = [lambda y_pred, y_true: precision_score(y_pred, y_true, average = 'macro')],
    optimisation_direction='maximize',
    list_classifiers_optimise=['lightgbm']
    )

# classification.apply()
classification.split_train_test()
classification.model_hyperoptimise()
classification.model_select_best()
classification.model_evaluate()
classification.summary



# n_classes should be computed from training data only
# adapt metrics to classification to apply per class, weighted, binary etc, maybe using scorer
