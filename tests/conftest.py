import matplotlib

matplotlib.use('Agg')  # -- plots must not open windows, or fail without a display in CI

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.datasets import make_classification

optuna.logging.set_verbosity(optuna.logging.WARNING)


def pytest_collection_modifyitems(items):
    """A failing trial only warns, so that a test would otherwise pass on a broken model.
    Set per test, as a filter in pyproject.toml imports the package before coverage starts measuring."""
    for item in items:
        item.add_marker(pytest.mark.filterwarnings('error::automl.utils.function_helper.OptimisationFailureWarning'))


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path, monkeypatch):
    """Models write to the working directory (e.g. catboost_info/), keep this out of the repository."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def automl_kwargs(tmp_path):
    """Keyword arguments of a quick AutomatedML run."""

    def build(**kwargs):
        defaults = dict(n_trial=2, timeout_study=60, write_folder=str(tmp_path / 'automl'), random_state=0, fit_frac=[1])
        return {**defaults, **kwargs}

    return build


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list('abcd'))
    y = pd.Series(3 * X.a - 2 * X.b + X.c ** 2 + rng.normal(scale=.2, size=300), name='target')
    return X, y


@pytest.fixture
def multioutput_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=list('abc'))
    y = pd.DataFrame({'first': 5 + X.a + rng.normal(scale=.2, size=300),
                      'second': 10 + X.b + rng.normal(scale=.2, size=300)})
    return X, y


@pytest.fixture
def grouped_data():
    """Multi-index data with 40 groups of 8 rows, where only the group mean of the target carries signal."""
    rng = np.random.default_rng(0)
    index = pd.MultiIndex.from_product([[f"g{i:02d}" for i in range(40)], range(8)], names=['group', 'row'])
    X = pd.DataFrame(rng.normal(size=(len(index), 3)), index=index, columns=list('abc'))
    signal = X.groupby(level='group').transform('mean') @ np.array([3., -2., 0.])
    y = pd.Series(signal + rng.normal(scale=.5, size=len(index)), index=index, name='target')
    return X, y


@pytest.fixture
def binary_data():
    X, y = make_classification(n_samples=300, n_features=6, n_informative=4, n_classes=2, random_state=0)
    return pd.DataFrame(X, columns=list('abcdef')), pd.Series(y, name='target')


@pytest.fixture
def multiclass_data():
    X, y = make_classification(n_samples=300, n_features=6, n_informative=4, n_classes=3, random_state=0)
    return pd.DataFrame(X, columns=list('abcdef')), pd.Series(y, name='target')
