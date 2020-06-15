# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

"""A variety of extra metrics useful for assessing fairness.

These are metrics which are not part of `scikit-learn`.
"""

import numpy as np
import sklearn.metrics as skm

from ._balanced_root_mean_squared_error import _balanced_root_mean_squared_error  # noqa: F401
from ._mean_predictions import mean_prediction, _mean_overprediction, _mean_underprediction  # noqa: F401,E501
from ._selection_rate import selection_rate  # noqa: F401,E501

_TOO_MANY_UNIQUE_Y_VALS = "Must have no more than two unique y values"
_RESTRICTED_VALS_IF_POS_LABEL_NONE = "If pos_label is not specified, values must be take from {0, 1} or {-1, 1}"  # noqa: E501
_NEED_POS_LABEL_IN_Y_VALS = "Must have pos_label in y values"


def _get_labels_for_confusion_matrix(y_true, y_pred, pos_label):
    r"""Figure out the labels argument for skm.confusion_matrix.

    This is an internal method used by the true/false positive/negative
    rate metrics (and hence are restricted to binary data). We compute
    these using the confusion matrix.
    This method prepares the `labels` argument of
    :py:func:`sklearn.metrics.confusion_matrix` based on the
    user's specifications.

    Parameters
    ----------
    y_true : array-like
        The true values for the metric computation

    y_pred : array-like
        The predicted values for the metric computation

    pos_label : scalar
        The value in the true and predicted arrays to treat as positive.
        This must be set if the two y arrays contain a single unique value.
        If set to None, then the larger of the two unique values will be used.

    Returns
    -------
    list
        A two element list, consisting of the unique values of the two arrays
        with the positive label listed last
    """
    unique_labels = list(np.unique(np.concatenate((y_true, y_pred), axis=None)))
    if len(unique_labels) > 2:
        raise ValueError(_TOO_MANY_UNIQUE_Y_VALS)

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
            unique_labels = [0, pos_label]
        else:
            unique_labels.append(pos_label)

    # Check that we have pos_label in the list
    if pos_label not in unique_labels:
        raise ValueError(_NEED_POS_LABEL_IN_Y_VALS)

    # Reverse the list if needed
    if unique_labels[1] != pos_label:
        unique_labels = list(reversed(unique_labels))

    return unique_labels


def true_positive_rate(y_true, y_pred, sample_weight=None, pos_label=None):
    r"""Calculate the true positive rate (also called sensitivity, recall, or hit rate).

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
    unique_labels = _get_labels_for_confusion_matrix(y_true, y_pred, pos_label)
    if len(unique_labels) == 1:
        tpr = float(unique_labels[0] == pos_label)
    else:
        tnr, fpr, fnr, tpr = skm.confusion_matrix(
            y_true, y_pred,
            sample_weight=sample_weight, labels=unique_labels, normalize="true").ravel()
    return tpr


def true_negative_rate(y_true, y_pred, sample_weight=None, pos_label=None):
    r"""Calculate the true negative rate (also called specificity or selectivity).

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
    unique_labels = _get_labels_for_confusion_matrix(y_true, y_pred, pos_label)
    if len(unique_labels) == 1:
        tnr = float(unique_labels[0] != pos_label)
    else:
        tnr, fpr, fnr, tpr = skm.confusion_matrix(
            y_true, y_pred,
            sample_weight=sample_weight, labels=unique_labels, normalize="true").ravel()
    return tnr


def false_positive_rate(y_true, y_pred, sample_weight=None, pos_label=None):
    r"""Calculate the false positive rate (also called fall-out).

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
    unique_labels = _get_labels_for_confusion_matrix(y_true, y_pred, pos_label)
    if len(unique_labels) == 1:
        fpr = 0.0
    else:
        tnr, fpr, fnr, tpr = skm.confusion_matrix(
            y_true, y_pred,
            sample_weight=sample_weight, labels=unique_labels, normalize="true").ravel()
    return fpr


def false_negative_rate(y_true, y_pred, sample_weight=None, pos_label=None):
    r"""Calculate the false negative rate (also called miss rate).

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
    unique_labels = _get_labels_for_confusion_matrix(y_true, y_pred, pos_label)
    if len(unique_labels) == 1:
        fnr = 0.0
    else:
        tnr, fpr, fnr, tpr = skm.confusion_matrix(
            y_true, y_pred,
            sample_weight=sample_weight, labels=unique_labels, normalize="true").ravel()
    return fnr


def _root_mean_squared_error(y_true, y_pred, **kwargs):
    r"""Calculate the root mean squared error."""
    return skm.mean_squared_error(y_true, y_pred, squared=False, **kwargs)
