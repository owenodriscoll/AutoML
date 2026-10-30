import numpy as np
import pandas as pd
import pytest

from automl.utils.grouped_train_test_splitter import groupwise_train_test_split


@pytest.fixture
def data():
    index = pd.MultiIndex.from_product([['a', 'b', 'c', 'd', 'e'], [1, 2], range(3)], names=['site', 'day', 'row'])
    X = pd.DataFrame({'x': np.arange(len(index), dtype=float)}, index=index)
    y = pd.DataFrame({'y': 2 * X.x})
    return X, y


def groups(frame, names):
    return set(frame.index.droplevel([n for n in frame.index.names if n not in names]).unique())


@pytest.mark.parametrize('names', [['site'], ['site', 'day']])
def test_groups_are_not_split_between_train_and_test(data, names):
    X_train, X_test, y_train, y_test = groupwise_train_test_split(*data, group_names=names, test_size=.4, random_state=0)
    assert not groups(X_train, names) & groups(X_test, names)
    assert groups(X_train, names) | groups(X_test, names) == groups(data[0], names)
    assert groups(y_train, names) == groups(X_train, names) and groups(y_test, names) == groups(X_test, names)


def test_all_rows_are_kept(data):
    X_train, X_test, _, _ = groupwise_train_test_split(*data, group_names=['site'], random_state=0)
    assert len(X_train) + len(X_test) == len(data[0])
    assert X_train.index.append(X_test.index).sort_values().equals(data[0].index.sort_values())


def test_test_size_applies_to_groups(data):
    _, X_test, _, _ = groupwise_train_test_split(*data, group_names=['site'], test_size=.4, random_state=0)
    assert len(groups(X_test, ['site'])) == 2


def test_features_and_targets_stay_aligned(data):
    X_train, X_test, y_train, y_test = groupwise_train_test_split(*data, group_names=['site'], random_state=0)
    for X_split, y_split in [(X_train, y_train), (X_test, y_test)]:
        assert X_split.index.equals(y_split.index)
        np.testing.assert_array_equal(y_split.y, 2 * X_split.x)


def test_index_names_are_restored(data):
    X_train, _, y_train, _ = groupwise_train_test_split(*data, group_names=['site'], random_state=0)
    assert list(X_train.index.names) == list(y_train.index.names) == ['site', 'day', 'row']


def test_random_state_makes_split_reproducible(data):
    first = groupwise_train_test_split(*data, group_names=['site'], random_state=3)[1]
    second = groupwise_train_test_split(*data, group_names=['site'], random_state=3)[1]
    assert first.equals(second)


def test_shuffle_false_takes_last_groups_as_test(data):
    _, X_test, _, _ = groupwise_train_test_split(*data, group_names=['site'], test_size=.2, shuffle=False)
    assert groups(X_test, ['site']) == {'e'}


def test_mismatching_index_names_are_rejected(data):
    X, y = data
    with pytest.raises(AssertionError, match="Index names"):
        groupwise_train_test_split(X, y.rename_axis(index=['other', 'day', 'row']), group_names=['site'])


def test_series_target_is_supported(data):
    X, y = data
    y_train = groupwise_train_test_split(X, y.y, group_names=['site'], random_state=0)[2]
    assert y_train.columns.tolist() == ['y']


def test_unequal_group_sizes(data):
    X, y = data
    X, y = X.drop(index=('a', 1)), y.drop(index=('a', 1))
    X_train, X_test, _, _ = groupwise_train_test_split(X, y, group_names=['site'], random_state=0)
    assert len(X_train) + len(X_test) == len(X)
