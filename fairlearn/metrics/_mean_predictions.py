# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np

from ._input_manipulations import _convert_to_ndarray_and_squeeze


def mean_prediction(y_true, y_pred, sample_weight=None) -> float:
    r"""Calculate the (weighted) mean prediction.

    The true values are ignored, but required as an argument in order
    to maintain a consistent interface

    Parameters
    ----------
    y_true : array_like
        The true labels (ignored)

    y_pred : array_like
        The predicted labels

    sample_weight : array_like
        Optional array of sample weights
    """
    y_p = _convert_to_ndarray_and_squeeze(y_pred)
    s_w = np.ones(len(y_p))
    if sample_weight is not None:
        s_w = _convert_to_ndarray_and_squeeze(sample_weight)

    return np.dot(y_p, s_w) / s_w.sum()


def _mean_overprediction(y_true, y_pred, sample_weight=None) -> float:
    """Calculate the (weighted) mean overprediction.

    This is the (weighted) mean of the error where any negative
    errors (i.e. underpredictions) are set to zero

    Parameters
    ----------
    y_true : array_like
        The true values

    y_pred : array_like
        The predicted values

    sample_weight : array_like
        Optional array of sample weights
    """
    y_t = _convert_to_ndarray_and_squeeze(y_true)
    y_p = _convert_to_ndarray_and_squeeze(y_pred)
    s_w = np.ones(len(y_p))
    if sample_weight is not None:
        s_w = _convert_to_ndarray_and_squeeze(sample_weight)

    err = y_p - y_t
    err[err < 0] = 0

    return np.dot(err, s_w) / s_w.sum()


def _mean_underprediction(y_true, y_pred, sample_weight=None) -> float:
    """Calculate the (weighted) mean underprediction.

    This is the (weighted) mean of the error where any
    positive errors (i.e. overpredictions) are set to zero.
    The absolute value of the underpredictions is used, so the
    returned value is always positive.

    Parameters
    ----------
    y_true : array_like
        The true values

    y_pred : array_like
        The predicted values

    sample_weight : array_like
        Optional array of sample weights
    """
    y_t = _convert_to_ndarray_and_squeeze(y_true)
    y_p = _convert_to_ndarray_and_squeeze(y_pred)
    s_w = np.ones(len(y_p))
    if sample_weight is not None:
        s_w = _convert_to_ndarray_and_squeeze(sample_weight)

    err = y_p - y_t
    err[err > 0] = 0

    # Error metrics should decrease to 0 so have to flip sign
    return -np.dot(err, s_w) / s_w.sum()
