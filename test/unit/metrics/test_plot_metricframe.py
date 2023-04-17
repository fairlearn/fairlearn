# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import matplotlib
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
    lower_bound = (
        centre_adjusted_probability - z * adjusted_standard_deviation
    ) / denominator
    upper_bound = (
        centre_adjusted_probability + z * adjusted_standard_deviation
    ) / denominator
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


@pytest.fixture()
def sample_metric_frame():
    return MetricFrame(
        metrics=metrics_dict, y_true=y_t, y_pred=y_p, sensitive_features=g_1
    )


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
    ax = matplotlib.pyplot.subplot()
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
    fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=2)
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
