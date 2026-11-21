# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Metrics for measuring fairness."""

import warnings
from typing import Literal

import numpy as np

from ._base_metrics import (
    _get_labels_for_confusion_matrix,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from ._metric_frame import MetricFrame

_UNDEFINED_RATE_WARNING = (
    "{rate} is undefined for groups with no {labels} labels, "
    "so these groups are left out: {groups}"
)


def _has_weight(y_true, y_pred, *, positive: bool, sample_weight) -> bool:
    """Report whether the positive (or negative) true labels carry any weight."""
    y_true = np.asarray(y_true)
    _get_labels_for_confusion_matrix(np.vstack((y_true, y_pred)), None)
    mask = (y_true == 1) if positive else (y_true != 1)
    weights = np.ones(len(y_true)) if sample_weight is None else np.asarray(sample_weight)
    return bool(weights[mask].sum() > 0)


def _true_positive_rate_or_nan(y_true, y_pred, sample_weight=None) -> float:
    """Return the true positive rate, or NaN where there are no positive labels."""
    if not _has_weight(y_true, y_pred, positive=True, sample_weight=sample_weight):
        return np.nan
    return true_positive_rate(y_true, y_pred, sample_weight=sample_weight)


def _false_positive_rate_or_nan(y_true, y_pred, sample_weight=None) -> float:
    """Return the false positive rate, or NaN where there are no negative labels."""
    if not _has_weight(y_true, y_pred, positive=False, sample_weight=sample_weight):
        return np.nan
    return false_positive_rate(y_true, y_pred, sample_weight=sample_weight)


def _quietly(aggregate):
    """Run a MetricFrame aggregate without its warning about NaN groups.

    ``_warn_on_undefined_rates`` has already named those groups and said why their rate
    is undefined, so the aggregate's own NaN-group warning (#1828) would repeat it.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r"MetricFrame\.\w+ left out these groups", category=UserWarning
        )
        return aggregate()


def _warn_on_undefined_rates(frame: MetricFrame, *, stacklevel: int) -> None:
    """Warn about the groups whose rate is undefined, which the aggregates leave out."""
    by_group = frame.by_group.to_frame() if frame.by_group.ndim == 1 else frame.by_group
    for column, rate, labels in (
        ("tpr", "The true positive rate", "positive"),
        ("fpr", "The false positive rate", "negative"),
    ):
        if column in by_group and by_group[column].isna().any():
            groups = list(by_group.index[by_group[column].isna()])
            warnings.warn(
                _UNDEFINED_RATE_WARNING.format(rate=rate, labels=labels, groups=groups),
                UserWarning,
                stacklevel=stacklevel + 1,
            )


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

    .. versionadded:: 0.4.6

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

    .. versionadded:: 0.4.6

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

    Depending on the `agg` parameter, this function computes either:

    - The greater of two metrics: `true_positive_rate_difference` and
      `false_positive_rate_difference`, if `agg='worst_case'`. The former is the
      difference between the largest and smallest of
      :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a` of the sensitive
      feature(s). The latter is defined similarly, but for
      :math:`P[h(X)=1 | A=a, Y=0]`.
    - The mean of these two metrics, if `agg='mean'`.

    The equalized odds difference of 0 means that all groups have the same
    true positive, true negative, false positive, and false negative rates.

    A group with no positive labels has no true positive rate, and a group with
    no negative labels has no false positive rate. Such a group is left out of
    that rate's comparison, with a warning, rather than counted as a rate of 0.

    Read more in the :ref:`User Guide <disparity_metrics>`.

    .. versionadded:: 0.4.6

    .. versionchanged:: 0.16.0
        A group whose rate is undefined is left out instead of counted as 0.

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

    Raises
    ------
    ValueError
        If `agg` is not one of ``'worst_case'`` or ``'mean'``.
    """
    if agg not in ["worst_case", "mean"]:
        raise ValueError(f"agg must be one of 'worst_case' or 'mean', got {agg}")

    eo = _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight)

    if agg == "worst_case":
        return _quietly(lambda: eo.difference(method=method)).max()
    else:
        return _quietly(lambda: eo.difference(method=method)).mean()


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

    A group with no positive labels has no true positive rate, and a group with
    no negative labels has no false positive rate. Such a group is left out of
    that rate's comparison, with a warning, rather than counted as a rate of 0.

    Read more in the :ref:`User Guide <disparity_metrics>`.

    .. versionadded:: 0.4.6

    .. versionchanged:: 0.16.0
        A group whose rate is undefined is left out instead of counted as 0.

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

    Raises
    ------
    ValueError
        If `agg` is not one of ``'worst_case'`` or ``'mean'``.
    """
    if agg not in ["worst_case", "mean"]:
        raise ValueError(f"agg must be one of 'worst_case' or 'mean', got {agg}")

    eo = _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight)

    if agg == "worst_case":
        return _quietly(lambda: eo.ratio(method=method)).min()
    else:
        return _quietly(lambda: eo.ratio(method=method)).mean()


def _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight) -> MetricFrame:
    fns = {"tpr": _true_positive_rate_or_nan, "fpr": _false_positive_rate_or_nan}
    sw_dict = {"sample_weight": sample_weight}
    sp = {"tpr": sw_dict, "fpr": sw_dict}
    eo = MetricFrame(
        metrics=fns,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params=sp,
    )
    _warn_on_undefined_rates(eo, stacklevel=3)
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
    :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a` of the sensitive feature(s).
    The equal opportunity difference of 0 means that all groups have the same true positive rate.

    A group with no positive labels has no true positive rate, so it is left out
    of the comparison, with a warning, rather than counted as a rate of 0.

    Read more in the :ref:`User Guide <disparity_metrics>`.

    .. versionadded:: 0.11.0

    .. versionchanged:: 0.16.0
        A group whose rate is undefined is left out instead of counted as 0.

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
        metrics={"tpr": _true_positive_rate_or_nan},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params={"tpr": {"sample_weight": sample_weight}},
    )
    _warn_on_undefined_rates(tpr, stacklevel=2)
    result = _quietly(lambda: tpr.difference(method=method))["tpr"]
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
    :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a` of the sensitive feature(s).
    The equal opportunity ratio of 1 means that all groups have the same true positive rate.

    A group with no positive labels has no true positive rate, so it is left out
    of the comparison, with a warning, rather than counted as a rate of 0.

    Read more in the :ref:`User Guide <disparity_metrics>`.

    .. versionadded:: 0.11.0

    .. versionchanged:: 0.16.0
        A group whose rate is undefined is left out instead of counted as 0.

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
        metrics={"tpr": _true_positive_rate_or_nan},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params={"tpr": {"sample_weight": sample_weight}},
    )
    _warn_on_undefined_rates(tpr, stacklevel=2)
    result = _quietly(lambda: tpr.ratio(method=method))["tpr"]
    return result
