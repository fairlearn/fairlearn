# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np

from fairlearn.metrics._input_manipulations import _convert_to_ndarray_and_squeeze
from ._metric_frame import MetricFrame

_EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE = "Empty {0} passed to exposure function."
_INVALID_RANKING_ERROR_MESSAGE = "Please input a valid complete ranking."
_ZERO_DIVISION_ERROR = "Average utility is 0, which causes a zero division error."


def exposure(y_true,
             y_pred,
             *,
             sample_weight=None) -> float:
    """Calculate the exposure allocated to the ranking in y_pred.

    Exposure is defined as the average logarithmic discount, where
    logarithmic discount equals :math:`1 / log_2 (1 + i)` for i in y_pred, as used
    in discounted cumulative gain (DCG).

    For consistency with other metric functions, the ``y_true`` argument
    is required, but ignored.


    Parameters
    ----------
    y_true : array_like
        Ground truth relevance scores. (ignored)
    y_pred : array_like
        The predicted ranking
    sample_weight : array_like
        Optional array of sample weights
    """
    if len(y_pred) == 0:
        raise ValueError(_EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE.format('y_pred'))

    if len(set(y_pred)) != len(y_pred):  # check for repetition in input ranking
        raise ValueError(_INVALID_RANKING_ERROR_MESSAGE)

    # ranking should start by 1, to prevent zero division.
    if 0 in y_pred:
        y_pred = [x+1 for x in y_pred]

    y_pred = _convert_to_ndarray_and_squeeze(y_pred)
    v = [1 / np.log2(1 + j) for j in y_pred]  # logarithmic discount

    s_w = np.ones(len(y_pred))
    if sample_weight is not None:
        s_w = np.squeeze(np.asarray(sample_weight))

    return np.dot(v, s_w).sum() / len(y_pred)


def utility(y_true,
            y_pred,
            *,
            sample_weight=None) -> float:
    """Calculate the utility of the ranking.

    Utility is defined as the average of y_true.

    The goal of this metric is to be used in the `exposure_utility_ratio` metric. Where we try
    to keep the `exposure` proportional to the `utility`.

    For consistency with other metric functions, the ``y_pred`` argument
    is required, but ignored.


    Parameters
    ----------
    y_true : array_like
        Ground truth relevance scores.
    y_pred : array_like
        The predicted ranking (ignored)
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


def proportional_exposure(
        y_true,
        y_pred,
        *,
        sample_weight=None) -> float:
    """Calculate the proportional exposure of the ranking in y_pred.

    Where we use the two metrics:: func:`fairlearn.metrics.exposure` and
    func:`fairlearn.metrics.utility`.

    Exposure is defined as the average logarithmic discount, where
    logarithmic discount equals :math:`1 / log_2 (1 + i)` for i in y_pred, as used
    in discounted cumulative gain (DCG).

    Utility is defined as the average of y_true.

    The goal of this metric is to keep `utility` and `exposure` proportional to each other. Since
    in ranking problems, an often occurring problem is that small differences in utility lead to
    huge differences in exposure. See `user guide`.

    The proportional exposure is defined as the exposure of ``y_pred`` divided by the utility of
    ``y_true``.


    Parameters
    ----------
    y_true : array_like
        Ground truth relevance scores.
    y_pred : array_like
        The predicted ranking
    sample_weight : array_like
        Optional array of sample weights
    """
    e = exposure(y_true, y_pred, sample_weight=sample_weight)
    u = utility(y_true, y_pred, sample_weight=sample_weight)

    if u == 0:
        raise ZeroDivisionError(_ZERO_DIVISION_ERROR)

    return e / u


def exposure_difference(
        y_true,
        y_pred,
        *,
        sensitive_features,
        method='between_groups',
        sample_weight=None) -> float:
    """Calculate the difference in exposure allocation between groups.

    The exposure allocation difference is defined as the difference
    between the largest and the smallest group-level exposure.
    The exposure allocation difference of 0 means that all groups have the same exposure. A high
    exposure difference can be seen as an indication of allocation harm.


    Parameters
    ----------
    y_true : array-like
        Ground truth relevance scores.
    y_pred : array-like
        Predicted ranking
    sensitive_features :
        The sensitive features over which the allocation harm should be assessed
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


def exposure_ratio(
        y_true,
        y_pred,
        *,
        sensitive_features,
        method='between_groups',
        sample_weight=None) -> float:
    """Calculate the ratio of exposure allocation between groups.

    The exposure ratio is defined as the ratio
    between the largest and the smallest group-level exposure.
    The exposure allocation ratio of 1 means that all groups have the same exposure. A low exposure
    ratio can be seen as an indication of allocation harm.


    Parameters
    ----------
    y_true : array-like
        Ground truth relevance scores.
    y_pred : array-like
        Predicted ranking
    sensitive_features :
        The sensitive features over which the allocation harm should be assessed
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


def proportional_exposure_difference(
        y_true,
        y_pred,
        *,
        sensitive_features,
        method='between_groups',
        sample_weight=None) -> float:
    """Calculate the proportional exposure difference between the best and worst serviced groups.

    Proportional exposure is defined as the exposure that a group gets divided by their average
    relevance.
    The proportional exposure difference of 0 means that all groups have the same exposure. A large
    difference can be seen as an indication of quality-of-service harm.


    Parameters
    ----------
    y_true : array-like
        Ground truth relevance scores.
    y_pred : array-like
        Predicted ranking
    sensitive_features :
        The sensitive features over which the quality-of-service should be assessed
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
    mf = MetricFrame(metrics=proportional_exposure,
                     y_true=y_true,
                     y_pred=y_pred,
                     sensitive_features=sensitive_features,
                     sample_params={'sample_weight': sample_weight})
    result = mf.difference(method=method)
    return result


def proportional_exposure_ratio(
        y_true,
        y_pred,
        *,
        sensitive_features,
        method='between_groups',
        sample_weight=None) -> float:
    """Calculate the proportional exposure ratio between the best and worst serviced groups.

    Proportional exposure is defined as the exposure that a group gets divided by their average
    relevance.
    The proportional exposure ratio of 1 means that all groups have the same exposure. A ratio
    close to 0, can be seen as an indication of quality-of-service harm.


    Parameters
    ----------
    y_true : array-like
        Ground truth relevance scores.
    y_pred : array-like
        Predicted ranking
    sensitive_features :
        The sensitive features over which the quality-of-service should be assessed
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
    mf = MetricFrame(metrics=proportional_exposure,
                     y_true=y_true,
                     y_pred=y_pred,
                     sensitive_features=sensitive_features,
                     sample_params={'sample_weight': sample_weight})
    result = mf.ratio(method=method)
    return result
