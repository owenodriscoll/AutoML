import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from AutoML.AutoML import AutomatedRegression

# %reset -f

#%%
X, y = make_regression(n_samples=1000, n_features=10, n_informative=2, random_state=42)

df_X = pd.DataFrame(X)
df_X['nine'] = pd.cut(df_X[9], bins=[-float('Inf'), -3, -1, 1, 3, float('Inf')], labels=['a', 'b', 'c', 'd', 'e'])
df_X['ten'] = pd.cut(df_X[9], bins=[-float('Inf'), -1, 1, float('Inf')], labels=['A', 'B', 'C'])
df_y = pd.Series(y)


# prepare Autoregression object
test = AutomatedRegression(
    y=df_y,
    X=df_X,
    # pca_value=0.95,
    # spline_value=2,
    # poly_value={'degree': 2, 'interaction_only': True},
    n_trial=20,
    nominal_columns= ['nine'],
    ordinal_columns= ['ten'],
    write_folder='/export/home/owen/Documents/scripts/AutoML/tests/auto_regression3',
    reload_study=True,
    reload_trial_cap=True,
    metric_optimise=r2_score,
    optimisation_direction='maximize',
    boosted_early_stopping_rounds=5,
    list_regressors_optimise=['bayesianridge', 'lightgbm', 'lassolars',],
    list_regressors_assess=[ 'lassolars', 'lightgbm','bayesianridge'],
    n_weak_models=5
    )

test.apply()
# test.split_train_test()
# test.regression_hyperoptimise()
# test.summary

# import random
# n_weak_models = 2
# random.seed(self.random_state)
# df_t = study.trials_dataframe() #attrs=("number", "value", "params", "state"))
# df_t_non_pruned = df_t[df_t.state == 'COMPLETE']

# # -- select best
# if self.optimisation_direction == 'maximize':
#     idx_best = df_t_non_pruned.value.argmax()
# elif self.optimisation_direction == 'minimize':
#     idx_best = df_t_non_pruned.value.argmin()
    
# idx_remaining = df_t_non_pruned.number.values.tolist()
# idx_remaining.remove(idx_best)
# idx_sampled = [idx_best] + random.sample(idx_remaining, n_weak_models) 
# weak_model_insert = [regressor_name+'_best']  + [regressor_name+'_'+str(i) for i in idx_sampled[1:]]

# model_params = study.trials[idx_sampled[0]].params
# parameter_dict = {k: model_params[k] for k in study.best_params.keys() & set(list_params_regressor)}

# categorical = CategoricalChooser(self.ordinal_columns, self.nominal_columns).fit()
# spline = SplineChooser(spline_value=model_params.get('spline_value')).fit()
# poly = PolyChooser(poly_value=model_params.get('poly_value')).fit()
# pca = PcaChooser(pca_value=model_params.get('pca_value')).fit()
# scaler = ScalerChooser(arg=model_params.get('scaler')).string_to_func()
# transformer = TransformerChooser(model_params.get('n_quantiles'), self.random_state).fit()
# transformed_regressor = TransformedTargetRegressor(
#     # index 0 is the regressor, index 1 is hyper-optimization function
#     regressor=self.regressors_2_assess[regressor_name][0](**parameter_dict),
#     transformer=transformer
# )

# idx_insert = self.list_regressors_assess.index(regressor_name)
# self.list_regressors_assess = ['lassolars', 'lightgbm', 'bayesianridge']
# tt = self.list_regressors_assess
# tt[idx_insert:idx_insert+1] = weak_model_insert





