# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from fairlearn.metrics import MetricFrame, plot_roc_curve_by_group

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


def _with_first_missing(values, missing_val):
    values = list(values)
    values[0] = missing_val
    return values


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

    @pytest.mark.parametrize("missing_val", [np.nan, None])
    @pytest.mark.parametrize(
        "make_sf,expected_name",
        [
            (lambda m: _with_first_missing(g_1, m), "sensitive_feature_0"),
            (lambda m: np.array(_with_first_missing(g_1, m), dtype=object), "sensitive_feature_0"),
            (lambda m: pd.Series(_with_first_missing(g_1, m)), "sensitive_feature_0"),
            (lambda m: pd.Series(_with_first_missing(g_1, m), name="sf"), "sf"),
            (lambda m: pd.DataFrame({"sf": _with_first_missing(g_1, m)}), "sf"),
            (
                lambda m: np.array([g_1, _with_first_missing(g_2, m)], dtype=object).T,
                "sensitive_feature_1",
            ),
            (lambda m: pd.DataFrame({"a": g_1, "b": _with_first_missing(g_2, m)}), "b"),
            (lambda m: {"a": list(g_1), "b": _with_first_missing(g_2, m)}, "b"),
        ],
    )
    def test_missing_sensitive_feature_raises_like_metricframe(
        self, missing_val, make_sf, expected_name
    ):
        message = f"Feature '{expected_name}' contains missing values"
        with pytest.raises(ValueError, match=message):
            plot_roc_curve_by_group(y_t, y_score, sensitive_features=make_sf(missing_val))
        with pytest.raises(ValueError, match=message):
            MetricFrame(
                metrics=roc_auc_score,
                y_true=y_t,
                y_pred=y_score,
                sensitive_features=make_sf(missing_val),
            )

    @pytest.mark.parametrize("missing_val", [np.nan, None])
    def test_all_missing_sensitive_feature_values_raise(self, missing_val):
        with pytest.raises(ValueError, match="Feature 'sensitive_feature_0' contains missing"):
            plot_roc_curve_by_group(y_t, y_score, sensitive_features=[missing_val] * len(y_t))
