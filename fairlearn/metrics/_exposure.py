# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
from typing import Any

from fairlearn.metrics._input_manipulations import _convert_to_ndarray_and_squeeze
from ._metric_frame import MetricFrame

_EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE = "Empty {0} passed to exposure function."
_INVALID_RANKING_ERROR_MESSAGE = "Please input a valid complete ranking."


def exposure(y_true,
             y_pred,
             *,
             pos_label: Any = 1,
             sample_weight=None) -> float:
    """Calculate the exposure allocated to the ranking in y_pred.

    For consistency with other metric functions, the ``y_true`` and ``pos_label`` arguments
    are required, but ignored.


    Parameters
    ----------
    y_true : array_like
        The true ranking (ignored)
    y_pred : array_like
        The predicted ranking
    pos_label : Scalar
        The label to treat as the 'good' outcome (ignored)
    sample_weight : array_like
        Optional array of sample weights
    """
    if len(y_pred) == 0:
        raise ValueError(_EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE.format('y_pred'))

    if len(set(y_pred)) != len(y_pred):  # check for repetition in input ranking
        raise ValueError(_INVALID_RANKING_ERROR_MESSAGE)

    y_pred = _convert_to_ndarray_and_squeeze(y_pred)
    v = [1 / np.log2(1 + j) for j in y_pred]  # logarithmic discount

    s_w = np.ones(len(y_pred))
    if sample_weight is not None:
        s_w = np.squeeze(np.asarray(sample_weight))

    return np.dot(v, s_w).sum()


def utility(y_true,
            y_pred,
            *,
            pos_label: Any = 1,
            sample_weight=None) -> float:
    """Calculate the exposure allocated to the ranking in y_pred.

    For consistency with other metric functions, the ``y_true`` and ``pos_label`` arguments
    are required, but ignored.


    Parameters
    ----------
    y_true : array_like
        The utility
    y_pred : array_like
        The predicted ranking
    pos_label : Scalar
        The label to treat as the 'good' outcome (ignored)
    sample_weight : array_like
        Optional array of sample weights
    """
    if len(y_true) == 0:
        raise ValueError(_EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE.format('y_true'))

    u = _convert_to_ndarray_and_squeeze(y_true)

    s_w = np.ones(len(u))
    if sample_weight is not None:
        s_w = np.squeeze(np.asarray(sample_weight))

    return np.dot(u, s_w).sum() / len(u)


def exposure_utility_ratio(
        y_true,
        y_pred,
        *,
        pos_label: Any = 1,
        sample_weight=None) -> float:
    """Calculate the exposure allocated to the ranking in y_pred.

    For consistency with other metric functions, the ``y_true`` and ``pos_label`` arguments
    are required, but ignored.


    Parameters
    ----------
    y_true : array_like
        The utility
    y_pred : array_like
        The predicted ranking
    pos_label : Scalar
        The label to treat as the 'good' outcome (ignored)
    sample_weight : array_like
        Optional array of sample weights
    """
    e = exposure(y_true, y_pred, sample_weight=sample_weight)
    u = utility(y_true, y_pred, sample_weight=sample_weight)
    return e / u


def allocation_harm_in_ranking_difference(
        y_true,
        y_pred,
        *,
        sensitive_features,
        method='between_groups',
        sample_weight=None) -> float:
    """Calculate the difference in exposure allocation.

    The exposure allocation difference is defined as the difference
    between the largest and the smallest group-level exposure,
    The exposure allocation difference of 0 means that all groups have the same exposure.


    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted ranking
    sensitive_features :
        The sensitive features over which demographic parity should be assessed
    method : str
        How to compute the differences. See :func:`fairlearn.metrics.MetricFrame.difference`
        for details.
    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The exposure difference
    """
    exposure_mf = MetricFrame(metrics=exposure,
                              y_true=y_true,
                              y_pred=y_pred,
                              sensitive_features=sensitive_features,
                              sample_params={'sample_weight': sample_weight})
    result = exposure_mf.difference(method=method)
    return result


def allocation_harm_in_ranking_ratio(
        y_true,
        y_pred,
        *,
        sensitive_features,
        method='between_groups',
        sample_weight=None) -> float:
    """Calculate the exposure allocation ratio.

    The exposure allocation ratio is defined as the ratio
    between the largest and the smallest group-level exposure,
    The exposure allocation ratio of 1 means that all groups have the same exposure.


    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted ranking
    sensitive_features :
        The sensitive features over which demographic parity should be assessed
    method : str
        How to compute the differences. See :func:`fairlearn.metrics.MetricFrame.ratio`
        for details.
    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The exposure ratio
    """
    exposure_mf = MetricFrame(metrics=exposure,
                              y_true=y_true,
                              y_pred=y_pred,
                              sensitive_features=sensitive_features,
                              sample_params={'sample_weight': sample_weight})
    result = exposure_mf.ratio(method=method)
    return result


def quality_of_service_harm_in_ranking_difference(
        y_true,
        y_pred,
        *,
        sensitive_features,
        method='between_groups',
        sample_weight=None) -> float:
    """Calculate the quality of service harm ratio between the best and worst serviced groups.

    Quality-of-service is defined as the exposure that a group gets divided by their average
    relevance.
    The quality-of-service ratio of 1 means that all groups have the same exposure.


    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.
    sensitive_features :
        The sensitive features over which demographic parity should be assessed
    method : str
        How to compute the differences. See :func:`fairlearn.metrics.MetricFrame.difference`
        for details.
    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The exposure/utility difference
    """
    mf = MetricFrame(metrics=exposure_utility_ratio,
                     y_true=y_true,
                     y_pred=y_pred,
                     sensitive_features=sensitive_features,
                     sample_params={'sample_weight': sample_weight})
    result = mf.difference(method=method)
    return result


def quality_of_service_harm_in_ranking_ratio(
        y_true,
        y_pred,
        *,
        sensitive_features,
        method='between_groups',
        sample_weight=None) -> float:
    """Calculate the quality of service harm ratio between the best and worst serviced groups.

    Quality-of-service is defined as the exposure that a group gets divided by their average
    relevance.
    The quality-of-service ratio of 1 means that all groups have the same exposure.


    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.
    y_pred : array-like
        Predicted ranking
    sensitive_features :
        The sensitive features over which demographic parity should be assessed
    method : str
        How to compute the differences. See :func:`fairlearn.metrics.MetricFrame.ratio`
        for details.
    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The exposure/utility ratio
    """
    mf = MetricFrame(metrics=exposure_utility_ratio,
                     y_true=y_true,
                     y_pred=y_pred,
                     sensitive_features=sensitive_features,
                     sample_params={'sample_weight': sample_weight})
    result = mf.ratio(method=method)
    return result
