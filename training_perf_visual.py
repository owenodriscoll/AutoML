#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:00:28 2022

@author: owen
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:56:41 2022

@author: owen
"""

import numpy as np
from matplotlib import pyplot as plt
import joblib
import optuna

# buoy_filtOnPca080Neighbours100
# buoy_filtOnPca080Neighbours100_randSampMedPrune
# buoy_filtOnPca080Neighbours100_TPESampHyperPrune5Models
# buoy_filtOnPca080Neighbours100_RandSampMedPrune5Models
# buoy_filtOnPca080Neighbours100_TPESampHyperPrune5Models_TrainSplitHyperFit
# california_TPE_hyper


folder_name = 'buoy_filtOnPca080Neighbours100_TPESampHyperPrune5Models'
file_name = 'catboost'
study = joblib.load('/home/owen/Documents/models/optuna/cells/' + folder_name + '/' + file_name + '.pkl')

#%%

data = study

df = data.trials_dataframe()
df.dropna(inplace=True)
df.reset_index(inplace=True)

df['time'] = df.datetime_complete - df.datetime_start
df['time'] = df.time.astype('int') / (1_000_000_000)
df = df[df.time>0]

names = []

for col in df.columns.values:
    if col[1] == '':
        names.append(col[0])
    else:
        names.append(col[1])

df.columns = names

#%%

# figs = optuna.visualization.matplotlib.plot_parallel_coordinate(study, target=lambda t: t.values[0], target_name="MAE")

# connected hyper parameters
fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study, target=lambda t: t.values[0], target_name="MAE")
# fig.set_ylim([-0.25,-0.12])
# fig.get_legend().remove()

# individual hyper parameters
figs = optuna.visualization.matplotlib.plot_slice(study)
_ = [fig.set_ylim([-0.25,-0.12]) for fig in figs]

# figs = optuna.visualization.matplotlib.plot_slice(study, target=lambda t: t.values[0], target_name="MAE") # for multi scorers
# [fig.set_ylim([-0.3,-0.10]) for fig in figs]

# Intermediate results with pruning
fig = optuna.visualization.matplotlib.plot_intermediate_values(study)
fig.set_ylim([-0.25,-0.12])
fig.get_legend().remove()

# fig.set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(study.trials)))))



number_pruned = sum(study.trials_dataframe()['state'] == 'PRUNED')
print(r"# of pruned trials: %2.0i" %number_pruned)

# study performance through trials
fig = optuna.visualization.matplotlib.plot_optimization_history(study)
fig.set_ylim([-0.25,-0.12])


# figs = optuna.visualization.matplotlib.plot_parallel_coordinate(study, target=lambda t: t.values[1], target_name="r2")
# figs = optuna.visualization.matplotlib.plot_slice(study, target=lambda t: t.values[1], target_name="r2")
# [fig.set_ylim([0.2,0.7]) for fig in figs]

# optuna.visualization.matplotlib.plot_contour(study, target=lambda t: t.values[0], target_name="r2")


#%%

# from optuna.visualization import plot_contour
# from optuna.visualization import plot_edf
# from optuna.visualization import plot_intermediate_values
# from optuna.visualization import plot_optimization_history
# from optuna.visualization import plot_parallel_coordinate
# from optuna.visualization import plot_param_importances
# from optuna.visualization import plot_slice




