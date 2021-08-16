# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from fairlearn.experimental.enable_metric_frame_plotting import plot_metric_frame
import matplotlib

from fairlearn.metrics import MetricFrame
from fairlearn.metrics._plotter import (
    _GIVEN_BOTH_ERRORS_AND_CONF_INT_ERROR,
    _METRIC_AND_ERRORS_NOT_SAME_LENGTH_ERROR,
    _METRIC_AND_CONF_INTERVALS_NOT_SAME_LENGTH_ERROR,
    _ERRORS_NEGATIVE_VALUE_ERROR,
    _CONF_INTERVALS_FLIPPED_BOUNDS_ERROR,
    _CONF_INTERVALS_MUST_BE_ARRAY
)

from .data_for_test import y_t, y_p, g_1

z_score = 1.959964
digits_of_precision = 4


def general_wilson(p, n, digits=digits_of_precision, z=z_score):
    """Return lower and upper bound using Wilson Interval."""
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = np.sqrt((p*(1 - p) + z*z / (4*n)))/np.sqrt(n)
    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    return [round(lower_bound, digits), round(upper_bound, digits)]


def general_normal_err_binomial(p, n, digits=digits_of_precision, z=z_score):
    """Return standard error (for binary classification).

    Assumes infinitely large population.
    Should be used when the sampling fraction is small.
    For sampling fraction > 5%, may want to use finite population correction [1]

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Margin_of_error
    """
    return round(z*np.sqrt(p*(1.0-p))/np.sqrt(n), digits)


def recall_wilson(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    bounds = general_wilson(tp/(tp+fn), tp + fn, digits_of_precision, z_score)
    return bounds


def incorrect_flipped_recall_wilson(y_true, y_pred):
    bounds = recall_wilson(y_true, y_pred)
    return bounds[1], bounds[0]


def incorrect_single_bound_recall_wilson(y_true, y_pred):
    bounds = recall_wilson(y_true, y_pred)
    return bounds[0]


def recall_normal_err(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    error = general_normal_err_binomial(tp/(tp+fn), tp + fn, digits_of_precision, z_score)
    return error


def incorrect_negative_recall_normal_err(y_true, y_pred):
    error = recall_normal_err(y_true, y_pred)
    return -1 * error


def accuracy_wilson(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    score = accuracy_score(y_true, y_pred)
    bounds = general_wilson(score, len(y_true), digits_of_precision, z_score)
    return bounds


def accuracy_normal_err(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    score = accuracy_score(y_true, y_pred)
    error = general_normal_err_binomial(score, len(y_true), digits_of_precision, z_score)
    return error


metrics_dict = {
    'Recall': recall_score,
    'Recall Bounds': recall_wilson,
    'Recall Error': recall_normal_err,
    'Accuracy': accuracy_score,
    'Accuracy Bounds': accuracy_wilson,
    'Accuracy Error': accuracy_normal_err,
    'Recall Bounds Flipped': incorrect_flipped_recall_wilson,
    'Recall Bounds Single': incorrect_single_bound_recall_wilson,
    'Recall Error Negative': incorrect_negative_recall_normal_err,
}


@pytest.fixture()
def sample_metric_frame():
    return MetricFrame(
        metrics=metrics_dict,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1)


def test_plotting_output(sample_metric_frame):
    axs = plot_metric_frame(sample_metric_frame).flatten()
    assert len(axs) == 6  # 6 is number of metrics that aren't arrays
    assert isinstance(axs[0], matplotlib.axes.Axes)


def test_ambiguous_error_metric(sample_metric_frame):
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(sample_metric_frame, metrics=["Recall"], errors=[
                          "Recall Error"], conf_intervals=["Recall Bounds"])
        assert str(exc.value) == _GIVEN_BOTH_ERRORS_AND_CONF_INT_ERROR


def test_unequal_error_bars_length(sample_metric_frame):
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(sample_metric_frame, metrics=[
                          "Recall", "Accuracy"], errors=["Recall Error"])
        assert str(exc.value) == _METRIC_AND_ERRORS_NOT_SAME_LENGTH_ERROR


def test_unequal_conf_intervals_length(sample_metric_frame):
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(sample_metric_frame, metrics=[
                          "Recall", "Accuracy"], conf_intervals=["Recall Bounds"])
        assert str(exc.value) == _METRIC_AND_CONF_INTERVALS_NOT_SAME_LENGTH_ERROR


def test_flipped_bounds(sample_metric_frame):
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(sample_metric_frame, metrics=["Recall"],
                          conf_intervals=["Recall Bounds Flipped"])
        assert str(exc.value) == _CONF_INTERVALS_FLIPPED_BOUNDS_ERROR


def test_single_bound(sample_metric_frame):
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(sample_metric_frame, metrics=["Recall"],
                          conf_intervals=["Recall Bounds Single"])
        assert str(exc.value) == _CONF_INTERVALS_MUST_BE_ARRAY


def test_negative_error(sample_metric_frame):
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(sample_metric_frame, metrics=["Recall"],
                          errors=["Recall Error Negative"])
        assert str(exc.value) == _ERRORS_NEGATIVE_VALUE_ERROR


def test_single_ax_input(sample_metric_frame):
    ax = matplotlib.pyplot.subplot()
    ax = plot_metric_frame(sample_metric_frame,
                           metrics=['Recall'],
                           errors=['Recall Error'],
                           ax=ax,
                           kind="bar",
                           colormap="Pastel1")


def test_multi_ax_input(sample_metric_frame):
    fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=2)
    ax[0].set_title("Recall Plot")
    ax[0].set_ylabel("Recall")
    ax[0].set_xlabel("Race")
    ax[0].set_ylim((0, 1))
    ax = plot_metric_frame(sample_metric_frame,
                           metrics=['Recall', 'Accuracy'],
                           errors=['Recall Error', 'Accuracy Error'],
                           ax=ax,
                           kind="bar",
                           colormap="Pastel1")
