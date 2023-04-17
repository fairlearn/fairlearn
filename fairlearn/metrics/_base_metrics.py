# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""A variety of base metrics useful for assessing fairness.

These are metrics which are not part of `scikit-learn`.
"""

import numpy as np
import sklearn.metrics as skm
from typing import Any

from ._metric_frame import check_consistent_length
from fairlearn.utils._input_manipulations import _convert_to_ndarray_and_squeeze

_EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE = (
    "Empty y_pred passed to selection_rate function."
)
_TOO_MANY_UNIQUE_Y_VALS = "Must have no more than two unique y values"
_RESTRICTED_VALS_IF_POS_LABEL_NONE = (
    "If pos_label is not specified, values must be from {0, 1} or {-1, 1}"  # noqa: E501
)
_NEED_POS_LABEL_IN_Y_VALS = "Must have pos_label in y values"


def _get_labels_for_confusion_matrix(labels, pos_label):
    r"""Figure out the labels argument for skm.confusion_matrix.

    This is an internal method used by the true/false positive/negative
    rate metrics (and hence are restricted to binary data). We compute
    these using the confusion matrix.
    This method prepares the `labels` argument of
    :py:func:`sklearn.metrics.confusion_matrix` based on the
    user's specifications.

    Parameters
    ----------
    labels : array-like
        Labels provided by the user

    pos_label : scalar
        The value in the true and predicted arrays to treat as positive.
        If this is not set, then the unique_labels must be a subset of
        {0, 1} or {-1, 1}, and it will then be set to 1

    Returns
    -------
    list
        A two element list, consisting of the unique labels
        with the positive label listed last. This array will
        always be two elements, even if the unique_labels array
        only has one element.
    """
    unique_labels = list(np.unique(labels))

    # Set pos_label if needed
    if pos_label is None:
        labels01 = frozenset([0, 1])
        labels11 = frozenset([-1, 1])
        if labels01.issuperset(unique_labels) or labels11.issuperset(unique_labels):
            pos_label = 1
        else:
            raise ValueError(_RESTRICTED_VALS_IF_POS_LABEL_NONE)

    # Ensure unique_labels has two elements
    if len(unique_labels) == 1:
        if unique_labels[0] == pos_label:
            unique_labels = [np.iinfo(np.int64).min, pos_label]
        else:
            unique_labels.append(pos_label)
    elif len(unique_labels) == 2:
        if pos_label == unique_labels[0]:
            unique_labels = list(reversed(unique_labels))
        elif pos_label == unique_labels[1]:
            pass
        else:
            raise ValueError(_NEED_POS_LABEL_IN_Y_VALS)
    else:
        raise ValueError(_TOO_MANY_UNIQUE_Y_VALS)

    return unique_labels


def true_positive_rate(y_true, y_pred, sample_weight=None, pos_label=None) -> float:
    r"""Calculate the true positive rate (also called sensitivity, recall, or hit rate).

    Read more in the :ref:`User Guide <custom_fairness_metrics>`.

    Parameters
    ----------
    y_true : array-like
        The list of true values

    y_pred : array-like
        The list of predicted values

    sample_weight : array-like, optional
        A list of weights to apply to each sample. By default all samples are weighted
        equally

    pos_label : scalar, optional
        The value to treat as the 'positive' label in the samples. If `None` (the default)
        then the largest unique value of the y arrays will be used.

    Returns
    -------
    float
        The true positive rate for the data
    """
    unique_labels = _get_labels_for_confusion_matrix(
        np.vstack((y_true, y_pred)), pos_label
    )
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        labels=unique_labels,
        normalize="true",
    ).ravel()
    return tpr


def true_negative_rate(y_true, y_pred, sample_weight=None, pos_label=None) -> float:
    r"""Calculate the true negative rate (also called specificity or selectivity).

    Read more in the :ref:`User Guide <custom_fairness_metrics>`.

    Parameters
    ----------
    y_true : array-like
        The list of true values

    y_pred : array-like
        The list of predicted values

    sample_weight : array-like, optional
        A list of weights to apply to each sample. By default all samples are weighted
        equally

    pos_label : scalar, optional
        The value to treat as the 'positive' label in the samples. If `None` (the default)
        then the largest unique value of the y arrays will be used.

    Returns
    -------
    float
        The true negative rate for the data
    """
    unique_labels = _get_labels_for_confusion_matrix(
        np.vstack((y_true, y_pred)), pos_label
    )
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        labels=unique_labels,
        normalize="true",
    ).ravel()
    return tnr


def false_positive_rate(y_true, y_pred, sample_weight=None, pos_label=None) -> float:
    r"""Calculate the false positive rate (also called fall-out).

    Read more in the :ref:`User Guide <custom_fairness_metrics>`.

    Parameters
    ----------
    y_true : array-like
        The list of true values

    y_pred : array-like
        The list of predicted values

    sample_weight : array-like, optional
        A list of weights to apply to each sample. By default all samples are weighted
        equally

    pos_label : scalar, optional
        The value to treat as the 'positive' label in the samples. If `None` (the default)
        then the largest unique value of the y arrays will be used.

    Returns
    -------
    float
        The false positive rate for the data
    """
    unique_labels = _get_labels_for_confusion_matrix(
        np.vstack((y_true, y_pred)), pos_label
    )
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        labels=unique_labels,
        normalize="true",
    ).ravel()
    return fpr


def false_negative_rate(y_true, y_pred, sample_weight=None, pos_label=None) -> float:
    r"""Calculate the false negative rate (also called miss rate).

    Read more in the :ref:`User Guide <custom_fairness_metrics>`.

    Parameters
    ----------
    y_true : array-like
        The list of true values

    y_pred : array-like
        The list of predicted values

    sample_weight : array-like, optional
        A list of weights to apply to each sample. By default all samples are weighted
        equally

    pos_label : scalar, optional
        The value to treat as the 'positive' label in the samples. If `None` (the default)
        then the largest unique value of the y arrays will be used.

    Returns
    -------
    float
        The false negative rate for the data
    """
    unique_labels = _get_labels_for_confusion_matrix(
        np.vstack((y_true, y_pred)), pos_label
    )
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        labels=unique_labels,
        normalize="true",
    ).ravel()
    return fnr


def count(y_true, y_pred) -> int:
    r"""Calculate the number of data points in each group when working with `MetricFrame`.

    The ``y_true`` argument is used to make this calculation. For consistency with
    other metric functions, the ``y_pred`` argument is required, but ignored.

    Read more in the :ref:`User Guide <assessment>`.

    Parameters
    ----------
    y_true : array_like
        The list of true labels

    y_pred : array_like
        The predicted labels (ignored)

    Returns
    -------
    int
        The number of data points in each group.
    """
    check_consistent_length(y_true, y_pred)
    return len(y_true)


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


def selection_rate(y_true, y_pred, *, pos_label: Any = 1, sample_weight=None) -> float:
    """Calculate the fraction of predicted labels matching the 'good' outcome.

    The argument `pos_label` specifies the 'good' outcome. For consistency with
    other metric functions, the ``y_true`` argument is required, but ignored.

    Read more in the :ref:`User Guide <custom_fairness_metrics>`.

    Parameters
    ----------
    y_true : array_like
        The true labels (ignored)

    y_pred : array_like
        The predicted labels

    pos_label : Scalar
        The label to treat as the 'good' outcome

    sample_weight : array_like
        Optional array of sample weights
    """
    selected = _convert_to_ndarray_and_squeeze(y_pred) == pos_label
    if len(selected) == 0:
        raise ValueError(_EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE)

    s_w = np.ones(len(selected))
    if sample_weight is not None:
        s_w = np.squeeze(np.asarray(sample_weight))

    return np.dot(selected, s_w) / s_w.sum()
