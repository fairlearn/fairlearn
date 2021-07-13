# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from fairlearn.experimental.enable_metric_frame_plotting import plot_metric_frame
import matplotlib

from fairlearn.metrics import MetricFrame
from fairlearn.metrics._plotter import (
    _GIVEN_BOTH_ERROR_BARS_AND_CONF_INT_ERROR,
    _METRIC_AND_ERROR_BARS_NOT_SAME_LENGTH_ERROR,
    _METRIC_AND_CONF_INTERVALS_NOT_SAME_LENGTH_ERROR,
    _ERROR_BARS_MUST_BE_TUPLE,
    _CONF_INTERVALS_MUST_BE_TUPLE,
    _ERROR_BARS_NEGATIVE_VALUE_ERROR,
    _CONF_INTERVALS_FLIPPED_BOUNDS_ERROR,
)

from .data_for_test import y_t, y_p, g_1

z_score = 1.959964
digits_of_precision = 4


def wilson(p, n, digits=digits_of_precision, z=z_score):
    """Return lower and upper bound."""
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = np.sqrt((p*(1 - p) + z*z / (4*n)))/np.sqrt(n)
    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    return (round(lower_bound, digits), round(upper_bound, digits))


def compute_error_metric(metric_value, sample_size, z_score):
    """Compute Standard Error Calculation (for Binary Classification).

    Assumes infinitely large population,
    Should be used when the sampling fraction is small.
    For sampling fraction > 5%, may want to use finite population correction
    https://en.wikipedia.org/wiki/Margin_of_error

    Note:
        Returns absolute error (%)
    """
    return z_score*np.sqrt(metric_value*(1.0-metric_value))/np.sqrt(sample_size)


def recall_wilson(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    bounds = wilson(tp/(tp+fn), tp + fn, digits_of_precision, z_score)
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
    error = compute_error_metric(tp/(tp+fn), tp + fn, z_score=z_score)
    return (error, error)


def incorrect_negative_recall_normal_err(y_true, y_pred):
    errors = recall_normal_err(y_true, y_pred)
    return (-1 * errors[0], -1 * errors[1])


def incorrect_single_error_recall_normal_err(y_true, y_pred):
    errors = recall_normal_err(y_true, y_pred)
    return errors[0]


def accuracy_wilson(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    score = accuracy_score(y_true, y_pred)
    bounds = wilson(score, len(y_true), digits_of_precision, z_score)
    return bounds


def accuracy_normal_err(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    score = accuracy_score(y_true, y_pred)
    error = compute_error_metric(score, len(y_true), z_score)
    return (error, error)


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
    'Recall Error Single': incorrect_single_error_recall_normal_err,
}
sample_metric_frame = [
    MetricFrame(
        metrics=metrics_dict,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1)
]


@pytest.mark.parametrize("metric_frame", sample_metric_frame)
def test_plotting_output(metric_frame):
    axs = plot_metric_frame(metric_frame)
    assert len(axs) == 4
    assert isinstance(axs[0], matplotlib.axes.Axes)


@pytest.mark.parametrize("metric_frame", sample_metric_frame)
def test_ambiguous_error_metric(metric_frame):
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(metric_frame, metrics=["Recall"], error_bars=[
                          "Recall Error"], conf_intervals=["Recall Bounds"])
        assert str(exc.value) == _GIVEN_BOTH_ERROR_BARS_AND_CONF_INT_ERROR


@pytest.mark.parametrize("metric_frame", sample_metric_frame)
def test_unequal_length(metric_frame):
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(metric_frame, metrics=[
                          "Recall", "Accuracy"], error_bars=["Recall Error"])
        assert str(exc.value) == _METRIC_AND_ERROR_BARS_NOT_SAME_LENGTH_ERROR
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(metric_frame, metrics=[
                          "Recall", "Accuracy"], conf_intervals=["Recall Bounds"])
        assert str(exc.value) == _METRIC_AND_CONF_INTERVALS_NOT_SAME_LENGTH_ERROR


@pytest.mark.parametrize("metric_frame", sample_metric_frame)
def test_flipped_bounds(metric_frame):
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(metric_frame, metrics=["Recall"],
                          conf_intervals=["Recall Bounds Flipped"])
        assert str(exc.value) == _CONF_INTERVALS_FLIPPED_BOUNDS_ERROR


@pytest.mark.parametrize("metric_frame", sample_metric_frame)
def test_single_bound(metric_frame):
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(metric_frame, metrics=["Recall"],
                          conf_intervals=["Recall Bounds Single"])
        assert str(exc.value) == _CONF_INTERVALS_MUST_BE_TUPLE


@pytest.mark.parametrize("metric_frame", sample_metric_frame)
def test_negative_error(metric_frame):
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(metric_frame, metrics=["Recall"],
                          error_bars=["Recall Error Negative"])
        assert str(exc.value) == _ERROR_BARS_NEGATIVE_VALUE_ERROR


@pytest.mark.parametrize("metric_frame", sample_metric_frame)
def test_single_error(metric_frame):
    with pytest.raises(ValueError) as exc:
        plot_metric_frame(metric_frame, metrics=["Recall"], conf_intervals=["Recall Error Single"])
        assert str(exc.value) == _ERROR_BARS_MUST_BE_TUPLE
