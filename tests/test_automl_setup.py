"""Everything AutomatedML does on construction: validation, defaults, data preparation and storage."""

import os

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, median_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

import automl
from automl import AutomatedClassification, AutomatedRegression, CustomLoss
from automl.automl import AutomatedML


@pytest.fixture
def regression(regression_data, automl_kwargs):
    X, y = regression_data
    return AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars']))


def test_public_api():
    for name in ['AutomatedRegression', 'AutomatedClassification', 'CustomLoss', 'LossFallbackWarning', 'OptimisationFailureWarning',
                 'watermark_from_pyproject']:
        assert hasattr(automl, name)


# ------------------------------------------------ validation ------------------------------------------------
def test_invalid_meta_learner_is_rejected(regression_data, automl_kwargs):
    X, y = regression_data
    with pytest.raises(AssertionError, match="stacking_meta_learner"):
        AutomatedRegression(y=y, X=X, **automl_kwargs(stacking_meta_learner='forest'))


def test_lambda_metric_is_rejected(regression_data, automl_kwargs):
    X, y = regression_data
    with pytest.raises(AssertionError, match="Lambda"):
        AutomatedRegression(y=y, X=X, **automl_kwargs(metric_optimise=lambda y_true, y_pred: 0.))


def test_grouping_column_must_be_in_index(grouped_data, automl_kwargs):
    X, y = grouped_data
    with pytest.raises((AssertionError, KeyError)):
        AutomatedRegression(y=y, X=X, **automl_kwargs(group_by_index_columns=['nonexistent']))


def test_unknown_model_name_does_not_drop_valid_models(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars', 'nonexistent']))
    assert list(regression._models_optimize) == ['lassolars']


# ------------------------------------------------- defaults -------------------------------------------------
def test_default_regression_settings(regression):
    assert regression.metric_optimise.metric is r2_score
    assert regression.metric_assess == [median_absolute_error, r2_score]
    assert regression.models_to_assess == regression.models_to_optimize
    assert regression.fit_frac == [1]


def test_default_models(regression_data, automl_kwargs, binary_data):
    X, y = regression_data
    assert AutomatedRegression(y=y, X=X, **automl_kwargs()).models_to_optimize == ['lassolars', 'bayesianridge', 'histgradientboost']
    X, y = binary_data
    assert AutomatedClassification(y=y, X=X, **automl_kwargs()).models_to_optimize == ['svc', 'sgd', 'histgradientboost']


def test_default_classification_settings(binary_data, automl_kwargs):
    X, y = binary_data
    classification = AutomatedClassification(y=y, X=X, **automl_kwargs(models_to_optimize=['knn']))
    assert classification.metric_optimise.metric is accuracy_score
    assert len(classification.metric_assess) == 2


def test_plain_metric_is_wrapped_and_custom_loss_kept(regression_data, automl_kwargs):
    X, y = regression_data
    loss = CustomLoss(r2_score, name='mine')
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(metric_optimise=loss, models_to_optimize=['lassolars']))
    assert regression.metric_optimise is loss


def test_default_cross_validation(regression):
    assert isinstance(regression.cross_validation, KFold)
    assert regression.cross_validation.n_splits == 5 and regression.cross_validation.shuffle
    assert regression.cross_validation.random_state == regression.random_state


def test_default_cross_validation_for_groups(grouped_data, automl_kwargs):
    X, y = grouped_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], group_by_index_columns=['group']))
    assert isinstance(regression.cross_validation, GroupKFold)


def test_custom_settings_are_kept(regression_data, automl_kwargs):
    X, y = regression_data
    cv, sampler, pruner = KFold(3), optuna.samplers.RandomSampler(seed=0), optuna.pruners.MedianPruner()
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], cross_validation=cv,
                                                               sampler=sampler, pruner=pruner))
    assert regression.cross_validation is cv and regression.sampler is sampler and regression.pruner is pruner


def test_default_sampler_and_pruner(regression):
    assert isinstance(regression.sampler, optuna.samplers.TPESampler)
    assert isinstance(regression.pruner, optuna.pruners.HyperbandPruner)


@pytest.mark.xfail(reason="BUG: samplers are recognised by module name instead of isinstance, so a subclass defined "
                          "outside optuna (e.g. from optunahub or optuna-integration) is silently replaced by TPESampler")
def test_custom_sampler_subclass_is_kept(regression_data, automl_kwargs):
    class MySampler(optuna.samplers.RandomSampler):
        pass

    X, y = regression_data
    sampler = MySampler()
    assert AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], sampler=sampler)).sampler is sampler


@pytest.mark.xfail(reason="BUG: pruners are recognised by module name instead of isinstance, so a custom pruner is "
                          "silently replaced by HyperbandPruner")
def test_custom_pruner_subclass_is_kept(regression_data, automl_kwargs):
    class MyPruner(optuna.pruners.MedianPruner):
        pass

    X, y = regression_data
    pruner = MyPruner()
    assert AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], pruner=pruner)).pruner is pruner


def test_transformer_kwargs_default_to_empty(regression):
    assert regression.transformer_kwargs == {}


def test_parent_class_requires_a_metric(regression_data, automl_kwargs):
    X, y = regression_data
    with pytest.raises(AssertionError):
        AutomatedML(y=y, X=X, **automl_kwargs())


# -------------------------------------------------- input data --------------------------------------------------
@pytest.mark.parametrize('convert_y', [lambda y: y, lambda y: y.to_frame(), lambda y: y.to_numpy(), lambda y: y.to_numpy().reshape(-1, 1)],
                         ids=['series', 'frame', 'array', 'column'])
@pytest.mark.parametrize('convert_X', [lambda X: X, lambda X: X.to_numpy()], ids=['frame', 'array'])
def test_input_formats(regression_data, automl_kwargs, convert_X, convert_y):
    X, y = regression_data
    regression = AutomatedRegression(y=convert_y(y), X=convert_X(X), **automl_kwargs(models_to_optimize=['lassolars']))
    assert isinstance(regression.X_train, pd.DataFrame) and isinstance(regression.y_train, pd.DataFrame)
    assert regression.y_train.shape == (240, 1) and regression.X_train.shape == (240, 4)
    assert not regression._is_multioutput


def test_series_as_features(regression_data, automl_kwargs):
    X, y = regression_data
    assert AutomatedRegression(y=y, X=X['a'], **automl_kwargs(models_to_optimize=['lassolars'])).X_train.shape == (240, 1)


def test_columns_are_converted_to_strings(automl_kwargs):
    X = np.random.default_rng(0).normal(size=(100, 2))
    regression = AutomatedRegression(y=X[:, 0], X=X, **automl_kwargs(models_to_optimize=['lassolars']))
    assert list(regression.X_train.columns) == ['0', '1'] and list(regression.y_train.columns) == ['0']


@pytest.mark.xfail(reason="BUG: column names are converted to strings in place, altering the DataFrames of the user")
def test_user_data_is_not_modified(automl_kwargs):
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 2)))
    y = pd.DataFrame({0: X[0]})
    AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars']))
    assert list(X.columns) == [0, 1] and list(y.columns) == [0]


def test_multi_output_is_detected(multioutput_data, automl_kwargs):
    X, y = multioutput_data
    assert AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars']))._is_multioutput


def test_unspecified_non_numeric_columns_are_reported(regression_data, automl_kwargs, capsys):
    X, y = regression_data
    AutomatedRegression(y=y, X=X.assign(text='a'), **automl_kwargs(models_to_optimize=['lassolars']))
    output = capsys.readouterr().out
    assert "Possible ordinal or nominal columns" in output and 'text' in output


@pytest.mark.parametrize('settings, columns', [(dict(nominal_columns=['text']), ['text']), (dict(ordinal_columns=['text']), ['text']),
                                               (dict(nominal_columns=['text'], ordinal_columns=['other']), ['text', 'other'])],
                         ids=['nominal', 'ordinal', 'both'])
def test_specified_non_numeric_columns_are_not_reported(regression_data, automl_kwargs, capsys, settings, columns):
    X, y = regression_data
    AutomatedRegression(y=y, X=X.assign(**{column: 'a' for column in columns}),
                        **automl_kwargs(models_to_optimize=['lassolars'], **settings))
    assert "Possible ordinal or nominal columns" not in capsys.readouterr().out


# ------------------------------------------------ train-test split ------------------------------------------------
def test_split_is_disjoint_and_complete(regression):
    assert not set(regression.train_index) & set(regression.test_index)
    assert len(regression.train_index) + len(regression.test_index) == 300


def test_split_respects_test_fraction(regression_data, automl_kwargs):
    X, y = regression_data
    assert len(AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], test_frac=.5)).X_test) == 150


def test_split_keeps_features_and_targets_aligned(regression):
    assert regression.X_train.index.equals(regression.y_train.index)
    assert regression.X_test.index.equals(regression.y_test.index)


def test_split_depends_on_random_state(regression_data, automl_kwargs, tmp_path):
    X, y = regression_data
    splits = [AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], random_state=seed,
                                                            write_folder=str(tmp_path / str(i)))).test_index
              for i, seed in enumerate([0, 0, 1])]
    np.testing.assert_array_equal(splits[0], splits[1])
    assert set(splits[0]) != set(splits[2])


def test_classification_split_is_stratified(automl_kwargs):
    rng = np.random.default_rng(0)
    y = pd.Series(np.r_[np.zeros(450, dtype=int), np.ones(50, dtype=int)], name='target')
    X = pd.DataFrame(rng.normal(size=(500, 2)), columns=list('ab'))
    classification = AutomatedClassification(y=y, X=X, **automl_kwargs(models_to_optimize=['knn']))
    assert classification.y_test.mean().item() == pytest.approx(.1)
    assert classification.y_train.mean().item() == pytest.approx(.1)


def test_grouped_split_keeps_groups_together(grouped_data, automl_kwargs):
    X, y = grouped_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], group_by_index_columns=['group']))
    train = set(regression.X_train.index.get_level_values('group'))
    test = set(regression.X_test.index.get_level_values('group'))
    assert not train & test and len(train | test) == 40


def test_rows_of_a_group_stay_together_in_cross_validation(grouped_data, automl_kwargs):
    X, y = grouped_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], group_by_index_columns=['group']))
    groups = regression.X_train.index.get_level_values('group')
    for train, test in regression._cv_splits(regression.X_train):
        assert not set(groups[train]) & set(groups[test])


# -------------------------------------------------- partial fits --------------------------------------------------
def test_sample_fraction_selects_rows(regression):
    index = regression._sample_fraction(regression.X_train, .25)
    assert len(index) == 60 and index.isin(regression.X_train.index).all()


def test_sample_fraction_is_reproducible(regression):
    assert regression._sample_fraction(regression.X_train, .5).equals(regression._sample_fraction(regression.X_train, .5))


def test_sample_fraction_selects_entire_groups(grouped_data, automl_kwargs):
    X, y = grouped_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], group_by_index_columns=['group']))
    selected = regression._sample_fraction(regression.X_train, .5)
    assert len(selected) == 8 * 16
    assert (regression.X_train.loc[selected].groupby(level='group').size() == 8).all()


def test_sample_fraction_selects_at_least_one_group(grouped_data, automl_kwargs):
    X, y = grouped_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], group_by_index_columns=['group']))
    assert len(regression._sample_fraction(regression.X_train, 1e-6)) == 8


@pytest.mark.xfail(reason="BUG: partial fits select rows by label with .loc, so duplicated index values multiply rows: "
                          "fractions are larger than requested and rows are repeated")
def test_partial_fit_with_duplicated_index_selects_requested_fraction(regression_data, automl_kwargs):
    X, y = regression_data
    regression = AutomatedRegression(y=pd.concat([y, y]), X=pd.concat([X, X]), **automl_kwargs(models_to_optimize=['lassolars']))
    selected = regression._sample_fraction(regression.X_train, .5)
    assert len(regression.X_train.loc[selected]) == len(regression.X_train) // 2


# ----------------------------------------------------- storage -----------------------------------------------------
def test_folders_are_created(regression):
    assert os.path.isdir(regression.write_folder) and os.path.isdir(regression.write_folder_sampler)
    assert os.path.isdir(regression.write_folder + 'input_data')


def test_trailing_slash_is_added(regression_data, tmp_path):
    X, y = regression_data
    regression = AutomatedRegression(y=y, X=X, write_folder=str(tmp_path / 'folder'), models_to_optimize=['lassolars'])
    assert regression.write_folder.endswith('folder/')


def test_existing_folder_is_reused(regression_data, automl_kwargs):
    X, y = regression_data
    for _ in range(2):
        AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars']))


@pytest.mark.parametrize('name', ['X_train', 'X_test', 'y_train', 'y_test'])
def test_split_data_is_stored(regression, name):
    stored = pd.read_parquet(f"{regression.write_folder}input_data/{name}.parquet")
    pd.testing.assert_frame_equal(stored, getattr(regression, name))


def test_grouped_data_is_stored_with_index(grouped_data, automl_kwargs):
    X, y = grouped_data
    regression = AutomatedRegression(y=y, X=X, **automl_kwargs(models_to_optimize=['lassolars'], group_by_index_columns=['group']))
    stored = pd.read_parquet(f"{regression.write_folder}input_data/X_test.parquet")
    pd.testing.assert_frame_equal(stored, regression.X_test)

