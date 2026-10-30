from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from automl import watermark_from_pyproject
from automl.utils.misc import assert_not_lambda, envelope, outlier_detector, plot_envelope_single

PYPROJECT = Path(__file__).parents[1] / 'pyproject.toml'


# ---------------------------------------------- assert_not_lambda ----------------------------------------------
def test_named_function_is_accepted():
    assert_not_lambda(print)


def test_lambda_is_rejected():
    with pytest.raises(AssertionError, match="Lambda"):
        assert_not_lambda(lambda: None)


def test_non_callable_is_rejected():
    with pytest.raises(AssertionError, match="not callable"):
        assert_not_lambda(1)


# ---------------------------------------------- outlier_detector ----------------------------------------------
@pytest.fixture
def with_outliers():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(200, 3)), columns=list('abc'))
    frame['label'] = 'keep'
    frame.loc[:4, list('abc')] = 50.
    return frame


@pytest.mark.parametrize('pca_comp', [.8, 2, None])
def test_outliers_are_removed(with_outliers, pca_comp):
    cleaned = outlier_detector(with_outliers, 'a', 'c', pca_comp=pca_comp, neighbours=20)
    assert not cleaned.index.isin(range(5)).any()
    assert len(cleaned) > 150


def test_outlier_detection_keeps_all_columns(with_outliers):
    assert list(outlier_detector(with_outliers, 'a', 'c', neighbours=20).columns) == list('abc') + ['label']


def test_outlier_detection_only_uses_selected_columns(with_outliers):
    with_outliers['d'] = np.nan  # -- outside the selected range
    assert len(outlier_detector(with_outliers, 'a', 'c', neighbours=20)) > 150


def test_outlier_plot(with_outliers, monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    outlier_detector(with_outliers, 'a', 'c', pca_comp=2, neighbours=20, plot_PCA=True)
    plt.close('all')


# -------------------------------------------------- envelope --------------------------------------------------
@pytest.fixture
def point_cloud():
    x = np.random.default_rng(0).uniform(1, 9, 2000)
    return pd.DataFrame({'x': x, 'y': 2 * x})


def test_envelope_of_linear_relation(point_cloud):
    center, median, count_x, *_ = envelope(point_cloud, 'x', 'y', 1, 9, steps=5, log=False)
    np.testing.assert_allclose(median.to_numpy() / center, 2, atol=.05)
    assert count_x.sum() == 2000


def test_envelope_quantiles_are_ordered(point_cloud):
    *_, low, mid_low, mid_high, high = envelope(point_cloud.assign(y=np.random.default_rng(0).normal(5, 1, 2000)),
                                                'x', 'y', 0, 20, steps=5, log=False)
    assert (low <= mid_low).all() and (mid_low <= mid_high).all() and (mid_high <= high).all()


def test_envelope_logarithmic_bins(point_cloud):
    center = envelope(point_cloud, 'x', 'y', 0, 2, steps=5, log=True)[0]
    np.testing.assert_allclose(center[0], (1 + 10 ** .5) / 2)


def test_envelope_does_not_modify_input(point_cloud):
    columns = list(point_cloud.columns)
    envelope(point_cloud, 'x', 'y', 0, 20, steps=5, log=False)
    assert list(point_cloud.columns) == columns


@pytest.mark.xfail(raises=(ValueError, AssertionError), reason="BUG: pandas>=3 defaults to observed=True in groupby, so empty bins are dropped "
                                                               "and the statistics no longer align with the bin centers")
def test_envelope_statistics_align_with_bins_when_bins_are_empty():
    frame = pd.DataFrame({'x': [1., 1.1, 9., 9.1], 'y': [1., 1.1, 9., 9.1]})
    center, median, *_ = envelope(frame, 'x', 'y', 0, 10, steps=11, log=False)
    assert len(median) == len(center)


@pytest.mark.xfail(raises=ValueError, reason="BUG: as above, the bins from -1 are empty for the absolute values which are binned, so "
                                             "the plot always fails on x and y lengths. `log=axis_scale == True` also never selects log bins")
def test_plot_envelope(point_cloud):
    *_, figure = plot_envelope_single(point_cloud, 'x', 'y', 26, 'title', 'x', 'y', axis_scale='linear', ax_min=.5, ax_max=20)
    assert len(figure.axes) == 2
    plt.close(figure)


# -------------------------------------------------- watermark --------------------------------------------------
def test_watermark_outside_ipython_reports(capsys):
    watermark_from_pyproject(PYPROJECT)
    assert "must be run in an IPython" in capsys.readouterr().out


def test_watermark_in_ipython_lists_project_dependencies(monkeypatch):
    magics = []

    class IPython:
        def run_line_magic(self, name, line):
            magics.append((name, line))

    monkeypatch.setattr('automl.utils.watermark.get_ipython', IPython)
    watermark_from_pyproject(PYPROJECT)
    assert magics[0] == ('load_ext', 'watermark')
    assert 'optuna' in magics[1][1] and 'pandas>=' not in magics[1][1]
