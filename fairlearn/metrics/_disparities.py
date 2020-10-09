# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Metrics for measuring disparity."""

from ._extra_metrics import selection_rate, true_positive_rate, false_positive_rate
from ._derived_metrics import make_derived_metric


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
    sel_rate_diff = make_derived_metric('difference',
                                        selection_rate,
                                        sample_param_names=['sample_weight'])
    result = sel_rate_diff(y_true, y_pred,
                           sensitive_features=sensitive_features,
                           sample_weight=sample_weight,
                           method='minmax')
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
    sel_rate_diff = make_derived_metric('ratio',
                                        selection_rate,
                                        sample_param_names=['sample_weight'])
    result = sel_rate_diff(y_true, y_pred,
                           sensitive_features=sensitive_features,
                           sample_weight=sample_weight,
                           method='minmax')
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
    spn = ['sample_weight']
    tpr_diff = make_derived_metric('difference', true_positive_rate, sample_param_names=spn)
    fpr_diff = make_derived_metric('difference', false_positive_rate, sample_param_names=spn)

    tpr_d = tpr_diff(y_true, y_pred,
                     sensitive_features=sensitive_features,
                     sample_weight=sample_weight,
                     method='minmax')
    fpr_d = fpr_diff(y_true, y_pred,
                     sensitive_features=sensitive_features,
                     sample_weight=sample_weight,
                     method='minmax')

    return max(tpr_d, fpr_d)


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
    spn = ['sample_weight']
    tpr_ratio = make_derived_metric('ratio', true_positive_rate, sample_param_names=spn)
    fpr_ratio = make_derived_metric('ratio', false_positive_rate, sample_param_names=spn)

    tpr_r = tpr_ratio(y_true, y_pred,
                      sensitive_features=sensitive_features,
                      sample_weight=sample_weight,
                      method='minmax')
    fpr_r = fpr_ratio(y_true, y_pred,
                      sensitive_features=sensitive_features,
                      sample_weight=sample_weight,
                      method='minmax')

    return min(tpr_r, fpr_r)
