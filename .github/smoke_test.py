"""
Checks that a package installed without test dependencies works, from loading all boosted libraries to a final model.
Run from outside the repository, as the local `src` would otherwise not be noticed if it differs from what is installed.
"""
import tempfile
import warnings

import numpy as np
import pandas as pd

from automl import AutomatedClassification, AutomatedRegression, OptimisationFailureWarning

warnings.simplefilter('error', OptimisationFailureWarning)  # -- a failing model only warns, so would otherwise pass here

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list('abc'))
targets = {AutomatedRegression: pd.Series(X.a + rng.normal(scale=.1, size=200), name='target'),
           AutomatedClassification: pd.Series((X.a > 0).astype(int), name='target')}

for automl_class, y in targets.items():
    automl = automl_class(y=y, X=X, models_to_optimize=['lightgbm', 'xgboost', 'catboost'], n_trial=2, timeout_study=120, fit_frac=[1],
                          write_folder=tempfile.mkdtemp())
    automl.apply()
    assert 'stacked' in automl.summary, f"{automl_class.__name__} produced no stacked model"
    print(f"{automl_class.__name__} works: {sorted(automl.summary)}")
