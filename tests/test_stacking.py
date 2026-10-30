import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, ShuffleSplit, TimeSeriesSplit

from automl.loss import CustomLoss
from automl.multi_output import MultiOutputStackingRegressor
from automl.stacking import LossStackingClassifier, LossStackingRegressor, _LossStackingBase

from helpers import ColumnClassifier, ColumnClassifierNoProba, ColumnRegressor, loss_handler

CV = KFold(4, shuffle=True, random_state=0)


@pytest.fixture
def signal_and_noise():
    """Column 'good' is the target, column 'bad' unrelated to the target."""
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=200), name='y')
    return pd.DataFrame({'good': y + rng.normal(scale=.01, size=200), 'bad': rng.normal(size=200)}), y


# ------------------------------------------------ regressor ------------------------------------------------
@pytest.mark.parametrize('metric, greater_is_better', [(r2_score, True), (mean_squared_error, False), (mean_absolute_error, False)])
def test_weights_select_informative_model(signal_and_noise, metric, greater_is_better):
    X, y = signal_and_noise
    stacker = LossStackingRegressor([('good', ColumnRegressor((0,))), ('bad', ColumnRegressor((1,)))],
                                    loss=loss_handler(metric, greater_is_better), cv=CV).fit(X, y)
    assert stacker.coef_[0, 0] == pytest.approx(1., abs=.1)
    assert stacker.coef_[1, 0] == pytest.approx(0., abs=.1)


def test_weights_are_non_negative(signal_and_noise):
    X, y = signal_and_noise
    estimators = [('inverted', ColumnRegressor((0,))), ('bad', ColumnRegressor((1,)))]
    stacker = LossStackingRegressor(estimators, loss=loss_handler(), cv=CV).fit(X.assign(good=-X.good), y)
    assert (stacker.coef_ >= 0).all()


def test_intercept_corrects_bias(signal_and_noise):
    X, y = signal_and_noise
    stacker = LossStackingRegressor([('biased', ColumnRegressor((0,), shift=3.))], loss=loss_handler(), cv=CV).fit(X, y)
    assert stacker.intercept_[0] < -2
    assert r2_score(y, stacker.predict(X)) > .95


@pytest.mark.parametrize('shift', [30., -1000.])
def test_intercept_corrects_large_bias(signal_and_noise, shift):
    X, y = signal_and_noise
    stacker = LossStackingRegressor([('biased', ColumnRegressor((0,), shift=shift))], loss=loss_handler(), cv=CV).fit(X, y)
    assert stacker.intercept_[0] == pytest.approx(-shift, rel=.02)
    assert r2_score(y, stacker.predict(X)) > .99


def test_weights_are_found_for_targets_with_large_offset():
    rng = np.random.default_rng(0)
    y = pd.Series(300 + rng.normal(size=300))
    X = pd.DataFrame({'one': y + rng.normal(size=300), 'two': y + rng.normal(size=300), 'bad': 300 + rng.normal(size=300)})
    stacker = LossStackingRegressor([('one', ColumnRegressor((0,))), ('two', ColumnRegressor((1,))), ('bad', ColumnRegressor((2,)))],
                                    loss=loss_handler(mean_squared_error, False), cv=CV).fit(X, y)
    assert stacker.coef_[:2, 0].min() > stacker.coef_[2, 0]
    assert mean_squared_error(y, stacker.predict(X)) < mean_squared_error(y, X.one)


def test_intercept_can_be_disabled(signal_and_noise):
    X, y = signal_and_noise
    stacker = LossStackingRegressor([('biased', ColumnRegressor((0,), shift=3.))], loss=loss_handler(), cv=CV,
                                    fit_intercept=False).fit(X, y)
    np.testing.assert_array_equal(stacker.intercept_, [0.])


def test_stacking_is_not_worse_than_best_estimator_out_of_fold(signal_and_noise):
    X, y = signal_and_noise
    estimators = [('good', ColumnRegressor((0,))), ('bad', ColumnRegressor((1,)))]
    stacker = LossStackingRegressor(estimators, loss=loss_handler(), cv=CV).fit(X, y)
    assert r2_score(y, stacker.predict(X)) >= r2_score(y, X.good) - 1e-9


def test_weights_blend_complementary_models():
    rng = np.random.default_rng(0)
    errors = rng.normal(size=(300, 2))
    y = pd.Series(rng.normal(size=300))
    X = pd.DataFrame({'one': y + errors[:, 0], 'two': y + errors[:, 1]})  # -- independent errors: average is best
    stacker = LossStackingRegressor([('one', ColumnRegressor((0,))), ('two', ColumnRegressor((1,)))],
                                    loss=loss_handler(mean_squared_error, False), cv=CV).fit(X, y)
    assert (stacker.coef_ > .3).all()


def test_group_loss_is_optimised_on_group_aggregates():
    """'row' fits the rows but is off per group, 'group' fits the group means: a grouped loss must select 'group'."""
    rng = np.random.default_rng(0)
    index = pd.MultiIndex.from_product([range(40), range(5)], names=['group', 'row'])
    group_mean = pd.Series(np.repeat(rng.normal(size=40), 5), index=index)
    deviation = pd.Series(rng.normal(scale=2, size=200), index=index)
    y = group_mean + deviation - deviation.groupby(level='group').transform('mean')  # -- group means are exact
    X = pd.DataFrame({'group': group_mean, 'row': y + rng.normal(scale=.3, size=200) + np.repeat(rng.normal(size=40), 5)})
    estimators = [('group', ColumnRegressor((0,))), ('row', ColumnRegressor((1,)))]

    grouped = LossStackingRegressor(estimators, loss=loss_handler(group_by_index_columns=['group']), cv=GroupKFold(4)).fit(X, y)
    per_row = LossStackingRegressor(estimators, loss=loss_handler(), cv=CV).fit(X, y)
    assert grouped.coef_[0, 0] > grouped.coef_[1, 0]
    assert per_row.coef_[0, 0] < per_row.coef_[1, 0]


def test_groups_are_passed_to_cross_validation():
    index = pd.MultiIndex.from_product([range(10), range(3)], names=['group', 'row'])
    X, y = pd.DataFrame({'a': np.arange(30.)}, index=index), pd.Series(np.arange(30.), index=index)
    seen = []

    class RecordingCV(GroupKFold):
        def split(self, X, y=None, groups=None):
            seen.append(groups)
            return super().split(X, y, groups)

    LossStackingRegressor([('a', ColumnRegressor())], loss=loss_handler(group_by_index_columns=['group']), cv=RecordingCV(3)).fit(X, y)
    np.testing.assert_array_equal(seen[0], np.repeat(range(10), 3))


@pytest.mark.parametrize('as_target', [lambda y: y, lambda y: y.to_frame(), lambda y: y.to_numpy()], ids=['series', 'frame', 'array'])
def test_regressor_accepts_target_formats(signal_and_noise, as_target):
    X, y = signal_and_noise
    stacker = LossStackingRegressor([('good', ColumnRegressor((0,)))], loss=loss_handler(), cv=CV).fit(X, as_target(y))
    assert stacker.predict(X).shape == (200,)


def test_regressor_multi_output_has_weights_per_target(multioutput_data):
    X, y = multioutput_data
    X = pd.DataFrame({'first': y['first'], 'second': y['second'], 'bad_1': X.a, 'bad_2': X.b})
    stacker = LossStackingRegressor([('good', ColumnRegressor((0, 1))), ('bad', ColumnRegressor((2, 3)))],
                                    loss=loss_handler(), cv=CV).fit(X, y)
    assert stacker.coef_.shape == (2, 2) and stacker.intercept_.shape == (2,)
    assert stacker.predict(X).shape == (300, 2)
    assert r2_score(y, stacker.predict(X)) > .99


def test_regressor_fits_base_estimators_on_all_data(signal_and_noise):
    X, y = signal_and_noise
    estimators = [('good', ColumnRegressor((0,)))]
    stacker = LossStackingRegressor(estimators, loss=loss_handler(), cv=CV).fit(X, y)
    assert stacker.estimators_[0][1] is not estimators[0][1]


def test_regressor_can_be_cloned(signal_and_noise):
    X, y = signal_and_noise
    stacker = LossStackingRegressor([('good', ColumnRegressor((0,)))], loss=loss_handler(), cv=CV)
    assert clone(stacker).fit(X, y).predict(X).shape == (200,)


# ---------------------------------------- splitters without a partition ----------------------------------------
def test_rows_without_out_of_fold_prediction_are_left_out_of_the_weights(signal_and_noise):
    """Rows only in training folds, as the first rows of TimeSeriesSplit. Their loss must not be evaluated."""
    X, y = signal_and_noise
    n_scored = []

    def recording_mse(y_true, y_pred):
        n_scored.append(len(y_true))
        return mean_squared_error(y_true, y_pred)

    stacker = LossStackingRegressor([('good', ColumnRegressor((0,)))], loss=loss_handler(recording_mse, False),
                                    cv=TimeSeriesSplit(3)).fit(X, y)
    assert set(n_scored) == {150}
    assert stacker.coef_[0, 0] == pytest.approx(1., abs=.1)


def test_rows_in_several_test_folds_average_their_predictions():
    X = pd.DataFrame({'a': np.arange(4.)})
    y = pd.Series(np.arange(4.))
    splits = [(np.array([2, 3]), np.array([0, 1])), (np.array([3]), np.array([1, 2]))]

    class OffsetByTrainingSize(ColumnRegressor):
        def fit(self, X, y=None):
            self.shift = len(X)
            return self

    stacker = LossStackingRegressor([('offset', OffsetByTrainingSize())], loss=loss_handler(), cv=splits)
    oof, = stacker._oof_predictions(X, y, ['predict'])
    np.testing.assert_array_equal(oof, [2., (3. + 2.) / 2, 3., np.nan])


def test_classifier_with_splitter_without_partition(labelled):
    X, y = labelled
    estimators = [('bad', ColumnClassifier(1)), ('good', ColumnClassifier(0)), ('no_proba', ColumnClassifierNoProba(1))]
    stacker = LossStackingClassifier(estimators, loss=loss_handler(accuracy_score, ml_objective='classification'),
                                     cv=ShuffleSplit(4, test_size=.3, random_state=0)).fit(X, y)
    assert stacker.coef_[1] > stacker.coef_[0]


def test_classifier_aligns_folds_which_have_not_seen_all_classes():
    X = pd.DataFrame({'a': [0, 1, 2] * 4})
    y = pd.Series([0, 1, 2] * 4)
    splits = [(np.flatnonzero(y != 2), np.arange(12))]  # -- the fold is trained without class 2
    stacker = LossStackingClassifier([('a', ColumnClassifier(0, confidence=.5))], cv=splits,
                                     loss=loss_handler(accuracy_score, ml_objective='classification'))
    stacker.classes_ = np.array([0, 1, 2])
    proba, = stacker._oof_predictions(X, y.to_numpy(), ['predict_proba'])
    assert proba.shape == (12, 3)
    np.testing.assert_array_equal(proba[:, 2], 0.)


# ------------------------------------------------ optimiser ------------------------------------------------
def test_penalty_replaces_non_finite_loss():
    base = _LossStackingBase([], loss=loss_handler(greater_is_better=False))
    assert base._penalised(np.nan) == base._penalised(np.inf) == 1e300
    assert base._penalised(2.) == 2.


def test_penalty_converts_to_value_to_minimise():
    assert _LossStackingBase([], loss=loss_handler(greater_is_better=True))._penalised(2.) == -2.


def test_optimiser_starts_from_best_guess():
    base = _LossStackingBase([], loss=loss_handler())

    def quadratic(x):
        return float(np.sum((x - 2) ** 2))

    result = base._optimise(quadratic, [np.array([0.]), np.array([5.])], bounds=[(0, None)])
    assert result == pytest.approx([2.], abs=1e-4)


def test_optimiser_respects_bounds():
    def decreasing(x):
        return float(x[0])

    base = _LossStackingBase([], loss=loss_handler())
    assert base._optimise(decreasing, [np.array([1.])], bounds=[(0, None)]) == pytest.approx([0.], abs=1e-4)


def test_default_cross_validation_of_stacking(signal_and_noise):
    X, y = signal_and_noise
    assert LossStackingRegressor([('good', ColumnRegressor((0,)))], loss=loss_handler()).fit(X, y).coef_.shape == (1, 1)


def test_stacking_of_arrays():
    X, y = np.random.default_rng(0).normal(size=(40, 2)), np.arange(40.)
    assert LossStackingRegressor([('a', ColumnRegressor((0,)))], loss=loss_handler(), cv=CV).fit(X, y).predict(X).shape == (40,)


def test_constant_penalised_objective_keeps_start():
    def always_invalid(y_true, y_pred):
        return np.nan

    X, y = pd.DataFrame({'a': np.arange(20.)}), pd.Series(np.arange(20.))
    stacker = LossStackingRegressor([('a', ColumnRegressor((0,)))], loss=loss_handler(always_invalid), cv=CV).fit(X, y)
    assert stacker.coef_[0, 0] == pytest.approx(1.)


# ------------------------------------------------ classifier ------------------------------------------------
@pytest.fixture
def labelled():
    """Column 'good' equals the target, column 'bad' is correct for about a third of the rows."""
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 3, 300), name='y')
    return pd.DataFrame({'good': y, 'bad': rng.integers(0, 3, 300)}), y


def test_classifier_selects_informative_model(labelled):
    X, y = labelled
    estimators = [('bad', ColumnClassifier(1)), ('good', ColumnClassifier(0))]
    stacker = LossStackingClassifier(estimators, loss=loss_handler(accuracy_score, ml_objective='classification'), cv=CV).fit(X, y)
    assert stacker.coef_[1] > stacker.coef_[0]
    assert accuracy_score(y, stacker.predict(X)) == 1.


def test_classifier_probabilities_are_valid(labelled):
    X, y = labelled
    estimators = [('bad', ColumnClassifier(1, confidence=.8)), ('good', ColumnClassifier(0, confidence=.8))]
    stacker = LossStackingClassifier(estimators, cv=CV, loss=loss_handler(CustomLoss(log_loss, needs_proba=True), False,
                                                                          ml_objective='classification')).fit(X, y)
    proba = stacker.predict_proba(X)
    assert proba.shape == (300, 3) and (proba >= 0).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.)
    assert log_loss(y, proba) < log_loss(y, ColumnClassifier(1, confidence=.8).fit(X, y).predict_proba(X))


def test_classifier_without_predict_proba_contributes_one_hot(labelled):
    X, y = labelled
    estimators = [('no_proba', ColumnClassifierNoProba(0)), ('bad', ColumnClassifier(1))]
    stacker = LossStackingClassifier(estimators, loss=loss_handler(accuracy_score, ml_objective='classification'), cv=CV).fit(X, y)
    assert stacker.predict_proba(X).shape == (300, 3)
    assert accuracy_score(y, stacker.predict(X)) == 1.


def test_classifier_blend_with_zero_weights_is_uniform():
    probabilities = np.array([[[1., 0.]], [[0., 1.]]])
    np.testing.assert_array_equal(LossStackingClassifier._blend(probabilities, np.zeros(2)), [[.5, .5]])


def test_classifier_blend_normalises_weights():
    probabilities = np.array([[[1., 0.]], [[0., 1.]]])
    np.testing.assert_array_equal(LossStackingClassifier._blend(probabilities, np.array([6., 2.])), [[.75, .25]])


def test_classifier_output_form_follows_loss(labelled):
    X, y = labelled
    stacker = LossStackingClassifier([('good', ColumnClassifier(0))], cv=CV,
                                     loss=loss_handler(accuracy_score, ml_objective='classification')).fit(X, y)
    np.testing.assert_array_equal(stacker._output(np.array([[.1, .2, .7]])), [2])

    stacker.loss = loss_handler(CustomLoss(log_loss, needs_proba=True), False, ml_objective='classification')
    np.testing.assert_array_equal(stacker._output(np.array([[.4, .6]])), [.6])


def test_classifier_with_string_labels():
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(0)
    y = pd.Series(rng.choice(['cat', 'dog'], 100))
    X = pd.DataFrame({'good': (y == 'dog').astype(float)})
    stacker = LossStackingClassifier([('knn', KNeighborsClassifier(n_neighbors=1))], cv=CV,
                                     loss=loss_handler(accuracy_score, ml_objective='classification')).fit(X, y)
    assert list(stacker.classes_) == ['cat', 'dog']
    assert accuracy_score(y, stacker.predict(X)) == 1.


# ------------------------------------------- ridge multi-output stacking -------------------------------------------
def test_multi_output_stacking_shapes_and_quality(multioutput_data):
    X, y = multioutput_data
    X = pd.DataFrame({'first': y['first'], 'second': y['second'], 'bad_1': X.a, 'bad_2': X.b})
    stacker = MultiOutputStackingRegressor([('good', ColumnRegressor((0, 1))), ('bad', ColumnRegressor((2, 3)))],
                                           final_estimator=RidgeCV(), cv=list(CV.split(X))).fit(X, y)
    assert stacker.predict(X).shape == (300, 2)
    assert r2_score(y, stacker.predict(X)) > .99
    assert len(stacker.fitted_estimators_) == 2


def test_multi_output_stacking_meta_learner_sees_out_of_fold_predictions(multioutput_data):
    """A model memorising the training data has to be distrusted by the meta-learner."""
    from sklearn.neighbors import KNeighborsRegressor
    X, y = multioutput_data
    X_noise = X.assign(a=np.random.default_rng(1).normal(size=300), b=np.random.default_rng(2).normal(size=300))
    stacker = MultiOutputStackingRegressor([('memorise', KNeighborsRegressor(n_neighbors=1))],
                                           final_estimator=RidgeCV(), cv=list(CV.split(X))).fit(X_noise, y)
    in_sample = stacker.fitted_estimators_[0][1].predict(X_noise)
    assert r2_score(y, in_sample) == 1.  # -- without out-of-fold predictions the meta-learner would pass these on
    assert r2_score(y, stacker.predict(X_noise)) < .5
