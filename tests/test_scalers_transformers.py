import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, QuantileTransformer, RobustScaler, SplineTransformer, \
    StandardScaler

from automl.scalers_transformers import CategoricalChooser, FourrierExpansion, PcaChooser, PolyChooser, ScalerChooser, \
    SplineChooser, TransformerChooser, decorator_report


@pytest.fixture
def trial():
    return optuna.create_study().ask()


# ------------------------------------------- PCA, poly, spline -------------------------------------------
@pytest.mark.parametrize('chooser, value, expected, attribute, attribute_value', [
    (PcaChooser, 2, PCA, 'n_components', 2),
    (PcaChooser, .95, PCA, 'n_components', .95),
    (PcaChooser, {'n_components': 3, 'whiten': True}, PCA, 'whiten', True),
    (PolyChooser, 3, PolynomialFeatures, 'degree', 3),
    (PolyChooser, {'degree': 2, 'interaction_only': True}, PolynomialFeatures, 'interaction_only', True),
    (SplineChooser, 6, SplineTransformer, 'n_knots', 6),
    (SplineChooser, {'n_knots': 5, 'degree': 2}, SplineTransformer, 'degree', 2),
])
def test_chooser_creates_configured_transformer(chooser, value, expected, attribute, attribute_value):
    transformer = chooser(value).fit()
    assert isinstance(transformer, expected)
    assert getattr(transformer, attribute) == attribute_value


@pytest.mark.parametrize('chooser', [PcaChooser, PolyChooser, SplineChooser])
def test_chooser_without_value_is_skipped(chooser, trial):
    assert chooser(None, trial=trial).fit_report_trial() is None
    assert trial.params == {}


def test_chooser_reports_parameters_to_trial_for_reconstruction(trial):
    transformer = PcaChooser(2, trial=trial).fit_report_trial()
    assert trial.params['pca_value'] == transformer.get_params()

    rebuilt = PcaChooser(trial.params['pca_value']).fit()
    assert rebuilt.get_params() == transformer.get_params()


def test_decorator_report_returns_self_when_requested():
    class Thing:
        value = None

        @decorator_report('value', to_return_self=True)
        def report(self):
            raise AssertionError('should be skipped')

    thing = Thing()
    assert thing.report() is thing


# ------------------------------------------------ scaler ------------------------------------------------
@pytest.mark.parametrize('arg, expected', [('minmax', MinMaxScaler), ('standard', StandardScaler), ('robust', RobustScaler)])
def test_scaler_from_string(arg, expected):
    assert isinstance(ScalerChooser(arg=arg).string_to_func(), expected)


@pytest.mark.parametrize('arg', [None, '', 'unknown'])
def test_scaler_without_valid_string_is_skipped(arg):
    assert ScalerChooser(arg=arg).string_to_func() is None


def test_scaler_is_suggested_from_trial(trial):
    scaler = ScalerChooser(trial=trial).suggest_fit()
    assert trial.params['scaler'] in [None, 'minmax', 'standard', 'robust']
    rebuilt = ScalerChooser(arg=trial.params['scaler']).string_to_func()
    assert type(scaler) is type(rebuilt)


# ------------------------------------------- target transformer -------------------------------------------
def test_transformer_is_suggested_when_nothing_provided():
    seen = set()
    for _ in range(12):
        trial = optuna.create_study().ask()
        transformer = TransformerChooser(trial=trial, random_state=3).suggest_and_fit()
        seen.add(trial.params['transformers'])
        if trial.params['transformers'] is None:
            assert transformer is None
        else:
            assert isinstance(transformer, QuantileTransformer)
            assert transformer.random_state == 3
            assert transformer.n_quantiles == trial.params['n_quantiles']
            assert transformer.output_distribution == trial.params['output_distribution']
    assert seen == {None, 'quantile_trans'}


def test_transformer_skip(trial):
    assert TransformerChooser(trial=trial, skip=True).suggest_and_fit() is None
    assert trial.params == {'transformers': None}


def test_transformer_uses_provided_settings(trial):
    transformer = TransformerChooser(n_quantiles=200, output_distribution='normal', trial=trial).suggest_and_fit()
    assert transformer.n_quantiles == 200 and transformer.output_distribution == 'normal'


def test_transformer_is_rebuilt_from_parameters():
    transformer = TransformerChooser(100, 'uniform', random_state=5).fit()
    assert transformer.get_params()['n_quantiles'] == 100 and transformer.random_state == 5


def test_transformer_without_parameters_is_skipped():
    assert TransformerChooser().fit() is None


# ------------------------------------------------ categorical ------------------------------------------------
@pytest.fixture
def categorical_frame():
    return pd.DataFrame({'number': [1., 2., 3., 4.], 'nominal': ['a', 'b', 'a', 'c'], 'ordinal': ['x', 'y', 'y', 'z']})


def test_no_categorical_columns_is_skipped():
    assert CategoricalChooser(None, None).fit() is None


@pytest.mark.parametrize('ordinal, nominal, n_columns', [(['ordinal'], ['nominal'], 5), (['ordinal'], None, 2),
                                                         (None, ['nominal'], 4)])
def test_categorical_encoding(categorical_frame, ordinal, nominal, n_columns):
    frame = categorical_frame.drop(columns=[c for c in ['ordinal', 'nominal'] if c not in (ordinal or []) + (nominal or [])])
    encoder = CategoricalChooser(ordinal, nominal).fit()
    assert isinstance(encoder, ColumnTransformer)
    encoded = encoder.fit_transform(frame)
    assert encoded.shape == (4, n_columns)
    assert np.issubdtype(encoded.dtype, np.number)


def test_unknown_nominal_category_is_ignored(categorical_frame):
    frame = categorical_frame[['nominal']]
    encoder = CategoricalChooser(None, ['nominal']).fit().fit(frame)
    np.testing.assert_array_equal(encoder.transform(pd.DataFrame({'nominal': ['new']})), [[0, 0, 0]])


# ------------------------------------------------ fourrier (deprecated) ------------------------------------------------
def test_fourrier_expansion_appends_cosine_and_sine_per_order():
    X = np.arange(10.).reshape(5, 2)
    expanded = FourrierExpansion(3).fit(X).transform(X)
    assert expanded.shape == (5, 2 + 2 * 2 * 3)
    np.testing.assert_array_equal(expanded[:, :2], X)


def test_fourrier_expansion_without_value_returns_none():
    expansion = FourrierExpansion(None)
    assert expansion.fit() is None and expansion.transform(np.zeros((2, 2))) is None
