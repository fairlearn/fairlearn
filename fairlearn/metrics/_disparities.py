# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Metrics for measuring disparity."""

from ._metrics_engine import (
    selection_rate_difference,
    selection_rate_ratio,
    true_positive_rate_difference,
    true_positive_rate_ratio,
    false_positive_rate_difference,
    false_positive_rate_ratio)


def demographic_parity_difference(y_true, y_pred, *, sensitive_features, sample_weight=None):
    """Calculate the demographic parity difference.

    :param 1D-array y_true: Ground truth (correct) labels.
    :param 1D-array y_pred: Predicted labels :math:`h(X)` returned by the classifier.
    :param 1D-array sensitive_features: Sensitive features.
    :param 1D-array sample_weight: Sample weights.
    :return: The difference between the largest and the smallest group-level selection rate,
        :math:`E[h(X) | A=a]`, across all values :math:`a` of the sensitive feature.
        The demographic parity difference of 0 means that all groups have the same selection rate.
    """
    return selection_rate_difference(
        y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight)


def demographic_parity_ratio(y_true, y_pred, *, sensitive_features, sample_weight=None):
    """Calculate the demographic parity ratio.

    :param 1D-array y_true: Ground truth (correct) labels.
    :param 1D-array y_pred: Predicted labels :math:`h(X)` returned by the classifier.
    :param 1D-array sensitive_features: Sensitive features.
    :param 1D-array sample_weight: Sample weights.
    :return: The ratio between the smallest and the largest group-level selection rate,
        :math:`E[h(X) | A=a]`, across all values :math:`a` of the sensitive feature.
        The demographic parity ratio of 1 means that all groups have the same selection rate.
    """
    return selection_rate_ratio(
        y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight)


def equalized_odds_difference(y_true, y_pred, *, sensitive_features, sample_weight=None):
    """Calculate the equalized odds difference.

    :param 1D-array y_true: Ground truth (correct) labels :math:`Y`.
    :param 1D-array y_pred: Predicted labels :math:`h(X)` returned by the classifier.
    :param 1D-array sensitive_features: Sensitive features.
    :param 1D-array sample_weight: Sample weights.
    :return: The greater of two metrics: `true_positive_rate_difference` and
        `false_negative_rate_difference`. The former is the difference between the
        largest and smallest of :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a`
        of the sensitive feature. The latter is defined similarly, but for
        :math:`P[h(X)=1 | A=a, Y=0]`.
        The equalized odds difference of 0 means that all groups have the same
        true positive, true negative, false positive, and false negative rates.
    """
    return max(
        true_positive_rate_difference(
            y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight),
        false_positive_rate_difference(
            y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight))


def equalized_odds_ratio(y_true, y_pred, *, sensitive_features, sample_weight=None):
    """Calculate the equalized odds ratio.

    :param 1D-array y_true: Ground truth (correct) labels :math:`Y`.
    :param 1D-array y_pred: Predicted labels :math:`h(X)` returned by the classifier.
    :param 1D-array sensitive_features: Sensitive features.
    :param 1D-array sample_weight: Sample weights.
    :return: The smaller of two metrics: `true_positive_rate_ratio` and
        `false_negative_rate_ratio`. The former is the ratio between the
        smallest and largest of :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a`
        of the sensitive feature. The latter is defined similarly, but for
        :math:`P[h(X)=1 | A=a, Y=0]`.
        The equalized odds ratio of 1 means that all groups have the same
        true positive, true negative, false positive, and false negative rates.
    """
    return min(
        true_positive_rate_ratio(
            y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight),
        false_positive_rate_ratio(
            y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight))
