# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import itertools

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from fairlearn.metrics import RocAuc

from .data_for_test import g_1, g_2, y_score, y_t

PYTEST_MPL_NOT_INSTALLED_MSG = "skipping plotting tests because matplotlib is not installed"


def is_mpl_installed():
    try:
        import matplotlib.pyplot as plt  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


@pytest.fixture()
def two_sensitive_features():
    return np.hstack((g_1.reshape(-1, 1), g_2.reshape(-1, 1)))


# --- Numeric correctness (no matplotlib required) ---


def test_overall_auc_matches_sklearn():
    rc = RocAuc(y_true=y_t, y_score=y_score, sensitive_features=g_1)
    assert rc.overall_auc == pytest.approx(roc_auc_score(y_t, y_score))


def test_auc_by_group_single_feature():
    rc = RocAuc(y_true=y_t, y_score=y_score, sensitive_features=g_1)
    scores = rc.auc_by_group()
    assert set(scores.keys()) == set(np.unique(g_1))
    for group in np.unique(g_1):
        mask = g_1 == group
        assert scores[group] == pytest.approx(roc_auc_score(y_t[mask], y_score[mask]))


def test_auc_by_group_two_features(two_sensitive_features):
    rc = RocAuc(y_true=y_t, y_score=y_score, sensitive_features=two_sensitive_features)
    scores = rc.auc_by_group()
    for a, b in itertools.product(np.unique(g_1), np.unique(g_2)):
        mask = (g_1 == a) & (g_2 == b)
        assert scores[(a, b)] == pytest.approx(roc_auc_score(y_t[mask], y_score[mask]))


def test_auc_by_group_subset(two_sensitive_features):
    rc = RocAuc(y_true=y_t, y_score=y_score, sensitive_features=two_sensitive_features)
    subset = [name for name in rc.by_group().index if name[0] == "aa"]
    scores = rc.auc_by_group(sensitive_index=subset)
    assert set(scores.keys()) == set(subset)


def test_by_group_returns_grouped_series(two_sensitive_features):
    rc = RocAuc(y_true=y_t, y_score=y_score, sensitive_features=two_sensitive_features)
    by_group = rc.by_group()
    # One entry per (g_1, g_2) combination.
    assert len(by_group) == len(np.unique(g_1)) * len(np.unique(g_2))
    # Each entry holds the (y_true, y_score) pair for that subgroup.
    grp_y_true, grp_y_score = by_group.iloc[0]
    assert len(grp_y_true) == len(grp_y_score)


# --- Plotting smoke tests (require matplotlib) ---


@pytest.fixture()
def close_figs():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.close("all")
    yield
    plt.close("all")


@pytest.mark.skipif(not is_mpl_installed(), reason=PYTEST_MPL_NOT_INSTALLED_MSG)
@pytest.mark.usefixtures("close_figs")
class TestPlots:
    def _roc(self, sensitive_features):
        return RocAuc(y_true=y_t, y_score=y_score, sensitive_features=sensitive_features)

    def test_plot_by_group_returns_axes(self, two_sensitive_features):
        import matplotlib

        ax = self._roc(two_sensitive_features).plot_by_group()
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_by_group_draws_expected_curves(self, two_sensitive_features):
        rc = self._roc(two_sensitive_features)
        ax = rc.plot_by_group()
        n_groups = len(rc.by_group().index)
        # One ROC curve per subgroup, plus the baseline and overall curves.
        assert len(ax.get_lines()) == n_groups + 2
        labels = [str(line.get_label()) for line in ax.get_lines()]
        assert all("AUC" in label for label in labels)

    def test_plot_by_group_without_overall_or_baseline(self, two_sensitive_features):
        rc = self._roc(two_sensitive_features)
        ax = rc.plot_by_group(include_overall=False, include_baseline=False)
        assert len(ax.get_lines()) == len(rc.by_group().index)

    def test_plot_overall_returns_axes(self):
        import matplotlib

        ax = self._roc(g_1).plot_overall()
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_baseline_returns_axes(self):
        import matplotlib

        ax = self._roc(g_1).plot_baseline()
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_plot_by_group_accepts_existing_axes(self, two_sensitive_features):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        result = self._roc(two_sensitive_features).plot_by_group(ax=ax)
        assert result is ax
