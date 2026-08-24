# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from fairlearn.metrics import plot_roc_curve_by_group

from .data_for_test import g_1, g_2, y_score, y_t

PYTEST_MPL_NOT_INSTALLED_MSG = "skipping plotting tests because matplotlib is not installed"


def is_mpl_installed():
    try:
        import matplotlib.pyplot as plt  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


def _expected_label(name, y_true, y_score):
    # Mirrors the legend label produced by sklearn's RocCurveDisplay.
    return f"{name} (AUC = {roc_auc_score(y_true, y_score):0.2f})"


@pytest.fixture()
def two_sensitive_features():
    return np.hstack((g_1.reshape(-1, 1), g_2.reshape(-1, 1)))


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
class TestPlotRocCurveByGroup:
    def test_returns_axes(self, two_sensitive_features):
        import matplotlib

        ax = plot_roc_curve_by_group(y_t, y_score, sensitive_features=two_sensitive_features)
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_creates_axes_when_none(self):
        ax = plot_roc_curve_by_group(y_t, y_score, sensitive_features=g_1)
        assert ax is not None

    def test_uses_provided_axes(self):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        result = plot_roc_curve_by_group(y_t, y_score, sensitive_features=g_1, ax=ax)
        assert result is ax

    def test_draws_curve_per_group_plus_overall_and_chance(self, two_sensitive_features):
        ax = plot_roc_curve_by_group(y_t, y_score, sensitive_features=two_sensitive_features)
        n_groups = len(set(zip(g_1, g_2, strict=False)))
        # One ROC curve per subgroup, plus the overall curve and the chance line.
        assert len(ax.get_lines()) == n_groups + 2

    def test_omit_overall_and_chance(self, two_sensitive_features):
        ax = plot_roc_curve_by_group(
            y_t,
            y_score,
            sensitive_features=two_sensitive_features,
            plot_overall=False,
            plot_chance_level=False,
        )
        assert len(ax.get_lines()) == len(set(zip(g_1, g_2, strict=False)))

    def test_chance_level_label(self):
        ax = plot_roc_curve_by_group(y_t, y_score, sensitive_features=g_1)
        labels = [line.get_label() for line in ax.get_lines()]
        assert "Chance level (AUC = 0.50)" in labels

    def test_overall_label_matches_sklearn(self):
        ax = plot_roc_curve_by_group(y_t, y_score, sensitive_features=g_1)
        labels = [line.get_label() for line in ax.get_lines()]
        assert _expected_label("Overall", y_t, y_score) in labels

    def test_group_labels_match_sklearn(self):
        ax = plot_roc_curve_by_group(
            y_t,
            y_score,
            sensitive_features=g_1,
            plot_overall=False,
            plot_chance_level=False,
        )
        labels = [line.get_label() for line in ax.get_lines()]
        for group in np.unique(g_1):
            mask = g_1 == group
            assert _expected_label(str(group), y_t[mask], y_score[mask]) in labels

    def test_merged_labels_for_multiple_features(self, two_sensitive_features):
        ax = plot_roc_curve_by_group(
            y_t,
            y_score,
            sensitive_features=two_sensitive_features,
            plot_overall=False,
            plot_chance_level=False,
        )
        labels = [line.get_label() for line in ax.get_lines()]
        # Combinations of the two features become comma-joined group names.
        assert any(label.startswith("aa,f ") for label in labels)

    def test_accepts_dataframe_sensitive_features(self):
        import pandas as pd

        sensitive_features = pd.DataFrame({"first": g_1, "second": g_2})
        ax = plot_roc_curve_by_group(
            y_t,
            y_score,
            sensitive_features=sensitive_features,
            plot_overall=False,
            plot_chance_level=False,
        )
        assert len(ax.get_lines()) == len(set(zip(g_1, g_2, strict=False)))

    def test_sets_title(self):
        ax = plot_roc_curve_by_group(y_t, y_score, sensitive_features=g_1, title="My ROC")
        assert ax.get_title() == "My ROC"

    def test_inconsistent_lengths_raise(self):
        with pytest.raises(ValueError):
            plot_roc_curve_by_group(y_t, y_score[:-1], sensitive_features=g_1)

    def test_accepts_list_inputs(self):
        ax = plot_roc_curve_by_group(
            list(y_t),
            list(y_score),
            sensitive_features=list(g_1),
            plot_overall=False,
            plot_chance_level=False,
        )
        assert len(ax.get_lines()) == len(np.unique(g_1))

    def test_accepts_pandas_series_inputs(self):
        import pandas as pd

        ax = plot_roc_curve_by_group(
            pd.Series(y_t),
            pd.Series(y_score),
            sensitive_features=pd.Series(g_1),
            plot_overall=False,
            plot_chance_level=False,
        )
        assert len(ax.get_lines()) == len(np.unique(g_1))

    def test_accepts_dict_sensitive_features(self):
        ax = plot_roc_curve_by_group(
            y_t,
            y_score,
            sensitive_features={"first": g_1, "second": g_2},
            plot_overall=False,
            plot_chance_level=False,
        )
        assert len(ax.get_lines()) == len(set(zip(g_1, g_2, strict=False)))

    def test_single_subgroup(self):
        sensitive_features = np.zeros_like(y_t)
        ax = plot_roc_curve_by_group(
            y_t,
            y_score,
            sensitive_features=sensitive_features,
            plot_overall=False,
            plot_chance_level=False,
        )
        assert len(ax.get_lines()) == 1

    def test_overall_only(self):
        ax = plot_roc_curve_by_group(
            y_t,
            y_score,
            sensitive_features=g_1,
            plot_chance_level=False,
        )
        labels = [line.get_label() for line in ax.get_lines()]
        assert len(ax.get_lines()) == len(np.unique(g_1)) + 1
        assert "Chance level (AUC = 0.50)" not in labels

    def test_chance_only(self):
        ax = plot_roc_curve_by_group(
            y_t,
            y_score,
            sensitive_features=g_1,
            plot_overall=False,
        )
        labels = [line.get_label() for line in ax.get_lines()]
        assert len(ax.get_lines()) == len(np.unique(g_1)) + 1
        assert _expected_label("Overall", y_t, y_score) not in labels

    def test_string_labels_with_pos_label(self):
        y_str = np.where(y_t == 1, "yes", "no")
        ax = plot_roc_curve_by_group(
            y_str,
            y_score,
            sensitive_features=g_1,
            pos_label="yes",
            plot_overall=False,
            plot_chance_level=False,
        )
        labels = [line.get_label() for line in ax.get_lines()]
        for group in np.unique(g_1):
            mask = g_1 == group
            expected_auc = roc_auc_score((y_str[mask] == "yes").astype(int), y_score[mask])
            assert f"{group} (AUC = {expected_auc:0.2f})" in labels
