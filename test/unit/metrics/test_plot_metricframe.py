# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.utils import check_consistent_length

from fairlearn.experimental.enable_metric_frame_plotting import plot_metric_frame
from fairlearn.metrics import MetricFrame
from fairlearn.metrics._plotter import (
    _CONF_INTERVALS_FLIPPED_BOUNDS_ERROR,
    _CONF_INTERVALS_MUST_BE_ARRAY,
    _METRIC_FRAME_INVALID_ERROR,
    _get_conf_intervals_from_metric_frame,
)

from .data_for_test import g_1, y_p, y_t

# We aim to create a 95% confidence interval, so we use a :code:`z_score` of 1.959964
z_score = 1.959964
digits_of_precision = 4


def general_wilson(p, n, digits=digits_of_precision, z=z_score):
    """Return lower and upper bound using Wilson Interval.

    Parameters
    ----------
    p : float
        Proportion of successes.
    n : int
        Total number of trials.
    digits : int
        Digits of precisions to which the returned bound will be rounded
    z : float
        Z-score, which indicates the number of standard deviations of confidence

    Returns
    -------
        np.ndarray
        Array of length 2 of form: [lower_bound, upper_bound]
    """
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z * z / (2 * n)
    adjusted_standard_deviation = np.sqrt((p * (1 - p) + z * z / (4 * n))) / np.sqrt(n)
    lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return np.array([round(lower_bound, digits), round(upper_bound, digits)])


def recall_wilson(y_true, y_pred):
    """Return Wilson Interval bounds for recall metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels
    y_pred : array-like of shape (n_samples,)
        Predicted labels

    Returns
    -------
        np.ndarray
        Array of length 2 of form: [lower_bound, upper_bound]
    """
    check_consistent_length(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    bounds = general_wilson(tp / (tp + fn), tp + fn, digits_of_precision, z_score)
    return bounds


def accuracy_wilson(y_true, y_pred):
    """Return Wilson Interval bounds for accuracy metric.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels
    y_pred : array-like of shape (n_samples,)
        Predicted labels

    Returns
    -------
        np.ndarray
        Array of length 2 of form: [lower_bound, upper_bound]
    """
    check_consistent_length(y_true, y_pred)
    score = accuracy_score(y_true, y_pred)
    bounds = general_wilson(score, len(y_true), digits_of_precision, z_score)
    return bounds


def incorrect_flipped_recall_wilson(y_true, y_pred):
    bounds = recall_wilson(y_true, y_pred)
    return np.array([bounds[1], bounds[0]])


def incorrect_single_bound_recall_wilson(y_true, y_pred):
    bounds = recall_wilson(y_true, y_pred)
    return bounds[0]


metrics_dict = {
    "Recall": recall_score,
    "Recall Bounds": recall_wilson,
    "Accuracy": accuracy_score,
    "Accuracy Bounds": accuracy_wilson,
    "Recall Bounds Flipped": incorrect_flipped_recall_wilson,
    "Recall Bounds Single": incorrect_single_bound_recall_wilson,
}


@pytest.fixture(scope="session")
def sample_metric_frame():
    return MetricFrame(metrics=metrics_dict, y_true=y_t, y_pred=y_p, sensitive_features=g_1)


def test_plotting_output(sample_metric_frame):
    """Tests for the correct output shape and output type."""
    axs = plot_metric_frame(sample_metric_frame).flatten()
    assert len(axs) == 3  # 3 is number of metrics that aren't arrays
    assert isinstance(axs[0], matplotlib.axes.Axes)


def test_invalid_metric_frame():
    """Tests handling of invalid metric frame."""
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(
            "not_metric_frame",
            metrics=["Recall"],
            conf_intervals=["Recall Bounds Flipped"],
        )
    assert str(exc.value) == _METRIC_FRAME_INVALID_ERROR


def test_flipped_bounds(sample_metric_frame):
    """Tests handling of flipped bounds for confidence intervals.

    Flipped bounds are when upper_bound is lower than lower_bound.
    """
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(
            sample_metric_frame,
            metrics=["Recall"],
            conf_intervals=["Recall Bounds Flipped"],
        )
    assert str(exc.value) == _CONF_INTERVALS_FLIPPED_BOUNDS_ERROR


def test_single_bound(sample_metric_frame):
    """Tests handling of invalid confidence interval."""
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(
            sample_metric_frame,
            metrics=["Recall"],
            conf_intervals=["Recall Bounds Single"],
        )
    assert str(exc.value) == _CONF_INTERVALS_MUST_BE_ARRAY


def test_single_ax_input(sample_metric_frame):
    """Tests plotting function works with single axis input."""
    ax = plt.subplot()
    ax = plot_metric_frame(
        sample_metric_frame,
        metrics=["Recall"],
        conf_intervals=["Recall Bounds"],
        ax=ax,
        kind="bar",
        colormap="Pastel1",
    )


def test_multi_ax_input(sample_metric_frame):
    """Tests plotting function works with multiple axis input."""
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].set_title("Recall Plot")
    ax[0].set_ylabel("Recall")
    ax[0].set_xlabel("Race")
    ax[0].set_ylim((0, 1))
    ax = plot_metric_frame(
        sample_metric_frame,
        metrics=["Recall", "Accuracy"],
        conf_intervals=["Recall Bounds", "Accuracy Bounds"],
        ax=ax,
        kind="bar",
        colormap="Pastel1",
    )


def test_auto_ci_single_metric():
    """plot_metric_frame auto-detects by_group_ci for single metric."""
    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_boot=10,
        ci_quantiles=[0.025, 0.975],
        random_state=0,
    )
    ci_map = _get_conf_intervals_from_metric_frame(mf)
    assert len(ci_map) == 1
    for col, (ci_col, ci_values) in ci_map.items():
        assert col == "accuracy_score"
        assert ci_col == "__metricframe_ci_accuracy_score"
        assert len(ci_values) == len(np.unique(g_1))
        for lower, upper in ci_values:
            assert isinstance(lower, float)
            assert isinstance(upper, float)
            assert lower <= upper


def test_auto_ci_multi_metric():
    """plot_metric_frame auto-detects by_group_ci for multiple metrics."""
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "recall": recall_score},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_boot=10,
        ci_quantiles=[0.025, 0.975],
        random_state=0,
    )
    ci_map = _get_conf_intervals_from_metric_frame(mf)
    assert len(ci_map) == 2
    assert "accuracy" in ci_map
    assert "recall" in ci_map
    # Verify the mapping has real CI values
    for col, (ci_col, ci_values) in ci_map.items():
        assert ci_col == f"__metricframe_ci_{col}"
        assert len(ci_values) == len(np.unique(g_1))


def test_no_ci_without_bootstrap():
    """Helper returns empty dict when MetricFrame has no CI."""
    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
    )
    ci_map = _get_conf_intervals_from_metric_frame(mf)
    assert ci_map == {}


def test_auto_ci_reversed_quantiles():
    """Reversed ci_quantiles still produce valid lower <= upper CIs."""
    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_boot=10,
        ci_quantiles=[0.975, 0.025],
        random_state=0,
    )
    ci_map = _get_conf_intervals_from_metric_frame(mf)
    assert len(ci_map) == 1
    for _col, (_ci_col, ci_values) in ci_map.items():
        for lower, upper in ci_values:
            assert lower <= upper, f"Expected lower <= upper, got {lower}, {upper}"


def test_auto_ci_more_than_two_quantiles():
    """More than 2 quantiles should warn and use the outermost pair."""
    import warnings

    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_boot=10,
        ci_quantiles=[0.025, 0.5, 0.975],
        random_state=0,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ci_map = _get_conf_intervals_from_metric_frame(mf)
        assert len(ci_map) == 1
        assert len(w) == 1
        assert "outermost" in str(w[0].message)


def test_plot_metric_frame_auto_ci_reordered_metrics():
    """Reordered metrics get their own error bars, not swapped."""
    mf = MetricFrame(
        metrics={"acc": accuracy_score, "rec": recall_score},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_boot=10,
        ci_quantiles=[0.025, 0.975],
        random_state=0,
    )
    captured = {}

    def fake_plot_df(df, metrics, kind, subplots, legend_label, df_all_errors=None, **kwargs):
        captured["df_all_errors"] = df_all_errors
        captured["metrics"] = metrics

    import fairlearn.metrics._plotter as mod

    monkeypatch = __import__("pytest").MonkeyPatch()
    monkeypatch.setattr(mod, "_plot_df", fake_plot_df)
    # Reverse the metric order: recall first, accuracy second
    plot_metric_frame(mf, metrics=["rec", "acc"])
    assert captured["df_all_errors"] is not None
    # recall error bar should reflect recall CI, not accuracy CI
    rec_err = captured["df_all_errors"]["rec"]
    # recall should have non-zero errors (from bootstrapping)
    assert not all(
        e[0] == 0 and e[1] == 0 for e in rec_err
    ), "recall CI should not be all-zero if bootstrapping produced variance"
    monkeypatch.undo()


def test_plot_metric_frame_auto_ci_subset_metrics():
    """Subset of metrics should not crash."""
    mf = MetricFrame(
        metrics={"acc": accuracy_score, "rec": recall_score},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_boot=10,
        ci_quantiles=[0.025, 0.975],
        random_state=0,
    )
    captured = {}

    def fake_plot_df(df, metrics, kind, subplots, legend_label, df_all_errors=None, **kwargs):
        captured["df_all_errors"] = df_all_errors
        captured["metrics"] = metrics

    import fairlearn.metrics._plotter as mod

    monkeypatch = __import__("pytest").MonkeyPatch()
    monkeypatch.setattr(mod, "_plot_df", fake_plot_df)
    # Only request one of the two metrics
    plot_metric_frame(mf, metrics=["acc"])
    assert captured["df_all_errors"] is not None
    assert len(captured["df_all_errors"].columns) == 1
    monkeypatch.undo()


def test_plot_metric_frame_auto_ci_single(monkeypatch):
    """End-to-end: plot_metric_frame uses auto CI for single metric."""
    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_boot=10,
        ci_quantiles=[0.025, 0.975],
        random_state=0,
    )
    captured = {}

    def fake_plot_df(df, metrics, kind, subplots, legend_label, df_all_errors=None, **kwargs):
        captured["df_all_errors"] = df_all_errors
        captured["metrics"] = metrics
        captured["df"] = df

    monkeypatch.setattr("fairlearn.metrics._plotter._plot_df", fake_plot_df)
    plot_metric_frame(mf)
    assert captured["df_all_errors"] is not None, "auto CI should produce error bars"
    assert len(captured["metrics"]) == 1


def test_plot_metric_frame_auto_ci_multi(monkeypatch):
    """End-to-end: plot_metric_frame uses auto CI for multiple metrics."""
    mf = MetricFrame(
        metrics={"acc": accuracy_score, "rec": recall_score},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_boot=10,
        ci_quantiles=[0.025, 0.975],
        random_state=0,
    )
    captured = {}

    def fake_plot_df(df, metrics, kind, subplots, legend_label, df_all_errors=None, **kwargs):
        captured["df_all_errors"] = df_all_errors
        captured["metrics"] = metrics

    monkeypatch.setattr("fairlearn.metrics._plotter._plot_df", fake_plot_df)
    plot_metric_frame(mf)
    assert captured["df_all_errors"] is not None, "auto CI for multi metric"
    assert len(captured["metrics"]) == 2


def test_plot_metric_frame_no_ci_without_bootstrap(monkeypatch):
    """End-to-end: no error bars when MetricFrame has no bootstrap CI."""
    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
    )
    captured = {}

    def fake_plot_df(df, metrics, kind, subplots, legend_label, df_all_errors=None, **kwargs):
        captured["df_all_errors"] = df_all_errors

    monkeypatch.setattr("fairlearn.metrics._plotter._plot_df", fake_plot_df)
    plot_metric_frame(mf, metrics="accuracy_score")
    assert captured["df_all_errors"] is None, "no CI without bootstrap"


def test_explicit_conf_intervals_not_overridden_by_auto(monkeypatch):
    """When user passes conf_intervals, auto-detect is skipped."""
    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_boot=10,
        ci_quantiles=[0.025, 0.975],
        random_state=0,
    )
    calls = []

    def spy(metric_frame):
        calls.append(1)
        return {}

    monkeypatch.setattr("fairlearn.metrics._plotter._get_conf_intervals_from_metric_frame", spy)
    try:
        plot_metric_frame(mf, metrics=["accuracy_score"], conf_intervals=["custom_ci"])
    except (KeyError, AttributeError):
        pass
    assert len(calls) == 0, "auto-detect must not fire when conf_intervals is explicitly passed"
