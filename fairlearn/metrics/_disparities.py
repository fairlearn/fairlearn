# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Metrics for measuring disparity."""

from ._extra_metrics import selection_rate, true_positive_rate, false_positive_rate
from ._metrics_frame import MetricsFrame


def demographic_parity_difference(
        y_true,
        y_pred,
        *,
        sensitive_features,
        sample_weight=None) -> float:
    """Calculate the demographic parity difference.

    The demographic parity difference is defined as the difference
    between the largest and the smallest group-level selection rate,
    :math:`E[h(X) | A=a]`, across all values :math:`a` of the sensitive feature.
    The demographic parity difference of 0 means that all groups have the same selection rate.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features :
        The sensitive features over which demographic parity should be assessed

    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The demographic parity difference
    """
    sel_rate = MetricsFrame(selection_rate,
                            y_true, y_pred,
                            sensitive_features=sensitive_features,
                            sample_params={'sample_weight': sample_weight})
    result = sel_rate.difference(method='minmax')['selection_rate']
    return result


def demographic_parity_ratio(
        y_true,
        y_pred,
        *,
        sensitive_features,
        sample_weight=None) -> float:
    """Calculate the demographic parity ratio.

    The demographic parity difference is defined as the difference
    between the largest and the smallest group-level selection rate,
    :math:`E[h(X) | A=a]`, across all values :math:`a` of the sensitive feature.
    The demographic parity ratio of 1 means that all groups have the same selection rate.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features :
        The sensitive features over which demographic parity should be assessed

    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The demographic parity ratio
    """
    sel_rate = MetricsFrame(selection_rate,
                            y_true, y_pred,
                            sensitive_features=sensitive_features,
                            sample_params={'sample_weight': sample_weight})
    result = sel_rate.ratio(method='minmax')['selection_rate']
    return result


def equalized_odds_difference(
        y_true,
        y_pred,
        *,
        sensitive_features,
        sample_weight=None) -> float:
    """Calculate the equalized odds difference.

    The greater of two metrics: `true_positive_rate_difference` and
    `false_positive_rate_difference`. The former is the difference between the
    largest and smallest of :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a`
    of the sensitive feature. The latter is defined similarly, but for
    :math:`P[h(X)=1 | A=a, Y=0]`.
    The equalized odds difference of 0 means that all groups have the same
    true positive, true negative, false positive, and false negative rates.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features :
        The sensitive features over which demographic parity should be assessed

    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The equalized odds difference
    """
    eo = _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight)

    return max(eo.difference(method='minmax'))


def equalized_odds_ratio(
        y_true,
        y_pred,
        *,
        sensitive_features,
        sample_weight=None) -> float:
    """Calculate the equalized odds difference.

    The greater of two metrics: `true_positive_rate_difference` and
    `false_positive_rate_difference`. The former is the difference between the
    largest and smallest of :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a`
    of the sensitive feature. The latter is defined similarly, but for
    :math:`P[h(X)=1 | A=a, Y=0]`.
    The equalized odds ratio of 1 means that all groups have the same
    true positive, true negative, false positive, and false negative rates.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features :
        The sensitive features over which demographic parity should be assessed

    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The equalized odds ratio
    """
    eo = _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight)

    return min(eo.ratio(method='minmax'))


def _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight) -> MetricsFrame:
    fns = {'tpr': true_positive_rate, 'fpr': false_positive_rate}
    sw_dict = {'sample_weight': sample_weight}
    sp = {'tpr': sw_dict, 'fpr': sw_dict}
    eo = MetricsFrame(fns,
                      y_true, y_pred,
                      sensitive_features=sensitive_features,
                      sample_params=sp)
    return eo
