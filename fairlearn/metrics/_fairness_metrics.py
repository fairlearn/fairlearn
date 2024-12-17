# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Metrics for measuring fairness."""

from typing import Literal

from ._base_metrics import false_positive_rate, selection_rate, true_positive_rate
from ._metric_frame import MetricFrame


def demographic_parity_difference(
    y_true,
    y_pred,
    *,
    sensitive_features,
    method: Literal["between_groups", "to_overall"] = "between_groups",
    sample_weight=None,
) -> float:
    """Calculate the demographic parity difference.

    The demographic parity difference is defined as the difference
    between the largest and the smallest group-level selection rate,
    :math:`E[h(X) | A=a]`, across all values :math:`a` of the sensitive feature(s).
    The demographic parity difference of 0 means that all groups have the same selection rate.

    Read more in the :ref:`User Guide <disparity_metrics>`.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features : array-like
        The sensitive features over which demographic parity should be assessed

    method : string {'between_groups', 'to_overall'}, default :code:`between_groups`
        How to compute the differences. See :func:`fairlearn.metrics.MetricFrame.difference`
        for details.

    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The demographic parity difference
    """
    sel_rate = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params={"sample_weight": sample_weight},
    )
    result = sel_rate.difference(method=method)
    return result


def demographic_parity_ratio(
    y_true,
    y_pred,
    *,
    sensitive_features,
    method: Literal["between_groups", "to_overall"] = "between_groups",
    sample_weight=None,
) -> float:
    """Calculate the demographic parity ratio.

    The demographic parity ratio is defined as the ratio
    between the smallest and the largest group-level selection rate,
    :math:`E[h(X) | A=a]`, across all values :math:`a` of the sensitive feature(s).
    The demographic parity ratio of 1 means that all groups have the same selection rate.

    Read more in the :ref:`User Guide <disparity_metrics>`.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features : array-like
        The sensitive features over which demographic parity should be assessed

    method : string {'between_groups', 'to_overall'}, default :code:`between_groups`
        How to compute the differences. See :func:`fairlearn.metrics.MetricFrame.ratio`
        for details.

    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The demographic parity ratio
    """
    sel_rate = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params={"sample_weight": sample_weight},
    )
    result = sel_rate.ratio(method=method)
    return result


def equalized_odds_difference(
    y_true,
    y_pred,
    *,
    sensitive_features,
    method: Literal["between_groups", "to_overall"] = "between_groups",
    sample_weight=None,
    agg: Literal["worst_case", "mean"] = "worst_case",
) -> float:
    """Calculate the equalized odds difference.

    The greater of two metrics: `true_positive_rate_difference` and
    `false_positive_rate_difference`. The former is the difference between the
    largest and smallest of :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a`
    of the sensitive feature(s). The latter is defined similarly, but for
    :math:`P[h(X)=1 | A=a, Y=0]`.
    The equalized odds difference of 0 means that all groups have the same
    true positive, true negative, false positive, and false negative rates.

    Read more in the :ref:`User Guide <disparity_metrics>`.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features : array-like
        The sensitive features over which equalized odds should be assessed

    method : string {'between_groups', 'to_overall'}, default :code:`between_groups`
        How to compute the differences. See :func:`fairlearn.metrics.MetricFrame.difference`
        for details.

    sample_weight : array-like
        The sample weights

    agg : string {'worst_case', 'mean'}, default :code:`worst_case`
        The aggregation method. One of `"worst_case"` or `"mean"`.
        If `"worst_case"`, the greater one of the false positive rate
        difference and true positive rate difference is returned.
        If `"mean"`, the mean of the differences is returned.

    Returns
    -------
    float
        The equalized odds difference
    """
    if agg not in ["worst_case", "mean"]:
        raise ValueError(f"agg must be one of 'worst_case' or 'mean', got {agg}")

    eo = _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight)

    if agg == "worst_case":
        return max(eo.difference(method=method))
    else:
        return eo.difference(method=method).mean()


def equalized_odds_ratio(
    y_true,
    y_pred,
    *,
    sensitive_features,
    method: Literal["between_groups", "to_overall"] = "between_groups",
    sample_weight=None,
    agg: Literal["worst_case", "mean"] = "worst_case",
) -> float:
    """Calculate the equalized odds ratio.

    The smaller of two metrics: `true_positive_rate_ratio` and
    `false_positive_rate_ratio`. The former is the ratio between the
    smallest and largest of :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a`
    of the sensitive feature(s). The latter is defined similarly, but for
    :math:`P[h(X)=1 | A=a, Y=0]`.
    The equalized odds ratio of 1 means that all groups have the same
    true positive, true negative, false positive, and false negative rates.

    Read more in the :ref:`User Guide <disparity_metrics>`.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features : array-like
        The sensitive features over which equalized odds should be assessed

    method : string {'between_groups', 'to_overall'}, default :code:`between_groups`
        How to compute the differences. See :func:`fairlearn.metrics.MetricFrame.ratio`
        for details.

    sample_weight : array-like
        The sample weights

    agg : string {'worst_case', 'mean'}, default :code:`worst_case`
        The aggregation method. One of `"worst_case"` or `"mean"`.
        If `"worst_case"`, the smaller one of the false positive rate ratio
        and true positive rate ratio is returned.
        If `"mean"`, the mean of the ratios is returned.

    Returns
    -------
    float
        The equalized odds ratio
    """
    if agg not in ["worst_case", "mean"]:
        raise ValueError(f"agg must be one of 'worst_case' or 'mean', got {agg}")

    eo = _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight)

    if agg == "worst_case":
        return min(eo.ratio(method=method))
    else:
        return eo.ratio(method=method).mean()


def _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight) -> MetricFrame:
    fns = {"tpr": true_positive_rate, "fpr": false_positive_rate}
    sw_dict = {"sample_weight": sample_weight}
    sp = {"tpr": sw_dict, "fpr": sw_dict}
    eo = MetricFrame(
        metrics=fns,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params=sp,
    )
    return eo


def equal_opportunity_difference(
    y_true,
    y_pred,
    *,
    sensitive_features,
    method: Literal["between_groups", "to_overall"] = "between_groups",
    sample_weight=None,
) -> float:
    """Calculate the equal opportunity difference.

    The equal opportunity difference is defined as the difference
    between the largest and the smallest group-level true positive rates,
    :math:`E[h(X) | A=a]`, across all values :math:`a` of the sensitive feature(s).
    The equal opportunity difference of 0 means that all groups have the same true positive rate.

    Read more in the :ref:`User Guide <disparity_metrics>`.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features : array-like
        The sensitive features over which equal opportunity should be assessed

    method : string {'between_groups', 'to_overall'}, default :code:`between_groups`
        How to compute the differences. See :func:`fairlearn.metrics.MetricFrame.difference`
        for details.

    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The equal opportunity difference
    """
    tpr = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params={"sample_weight": sample_weight},
    )
    result = tpr.difference(method=method)
    return result


def equal_opportunity_ratio(
    y_true,
    y_pred,
    *,
    sensitive_features,
    method: Literal["between_groups", "to_overall"] = "between_groups",
    sample_weight=None,
) -> float:
    """Calculate the equal opportunity ratio.

    The equal opportunity ratio is defined as the ratio
    between the smallest and the largest group-level true positive rate,
    :math:`E[h(X) | A=a]`, across all values :math:`a` of the sensitive feature(s).
    The equal opportunity ratio of 1 means that all groups have the same true positive rate.

    Read more in the :ref:`User Guide <disparity_metrics>`.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features : array-like
        The sensitive features over which equal opportunity should be assessed

    method : string {'between_groups', 'to_overall'}, default :code:`between_groups`
        How to compute the differences. See :func:`fairlearn.metrics.MetricFrame.ratio`
        for details.

    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The equal opportunity ratio
    """
    tpr = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params={"sample_weight": sample_weight},
    )
    result = tpr.ratio(method=method)
    return result
