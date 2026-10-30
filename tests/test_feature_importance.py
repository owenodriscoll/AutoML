import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from automl import AutomatedRegression

pytest.importorskip('shap')

SHAP_ARGUMENTS = dict(n_train_points=5, n_test_points=4, kwargs_shap_plot={'show': False})


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


@pytest.fixture
def regression(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars']))
    regression.apply()
    return regression


def test_returns_shap_values_per_feature(regression):
    values, data = regression.model_feature_importance(cluster=False, **SHAP_ARGUMENTS)
    assert isinstance(values, pd.DataFrame)
    assert values.shape == (4, 4) and list(values.columns) == list('abcd')
    assert data.index.isin(regression.X_test.index).all()


def test_shap_values_identify_informative_features(regression):
    values, _ = regression.model_feature_importance(cluster=False, **SHAP_ARGUMENTS)
    importance = values.abs().mean()
    assert importance['a'] > importance['d']
    assert importance['b'] > importance['d']


def test_clustered_background_data(regression):
    values, _ = regression.model_feature_importance(cluster=True, **SHAP_ARGUMENTS)
    assert values.shape == (4, 4)


def test_multi_output_returns_values_per_target(multioutput_data, automl_kwargs):
    X, y = multioutput_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars']))
    regression.apply()
    values, _ = regression.model_feature_importance(cluster=False, **SHAP_ARGUMENTS)
    assert list(values) == ['first', 'second']
    assert values['first'].shape == (4, 3)


def test_stored_model_is_reloaded(regression, regression_data, automl_kwargs):
    X, y = regression_data
    reloaded = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], reload_study=True))
    assert reloaded._model_final is None
    reloaded.model_feature_importance(cluster=False, **SHAP_ARGUMENTS)
    assert reloaded._model_final is not None


def test_without_model_raises(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars']))
    with pytest.raises(Exception, match="No trained model"):
        regression.model_feature_importance()


@pytest.mark.xfail(raises=ValueError, reason="BUG: the defaults (200 clusters, 200 test rows) are not limited by the size of the "
                                             "data, so they crash for any dataset under 1000 rows at test_frac=0.2")
def test_default_arguments_work_on_small_datasets(regression):
    values, _ = regression.model_feature_importance(kwargs_shap_plot={'show': False})
    assert len(values) == len(regression.X_test)
