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
_NEED_POS_LABEL_IN_Y_VALS = "Must have pos_label in y values"


def _get_labels_for_confusion_matrix(y_true, y_pred, pos_label):
    r"""Figure out the labels argument for skm.confusion_matrix.

    This is an internal method used by the true/false positive/negative
    rate metrics (and hence are restricted to binary data). We compute
    these using the confusion matrix.
    In general, we expect the input data to be :math:`\in {0, 1}`
    with 1 being the 'positive' label. However, users might want to have
    a different pair of values, and then specify the positive label
    for themselves.
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
        The value in the true and predicted arrays to treat as positive

    Returns
    -------
    list
        A two element list, consisting of the unique values of the two arrays
        with the positive label listed last
    """
    my_labels = list(np.unique(np.concatenate((y_true, y_pred), axis=None)))
    if len(my_labels) > 2:
        raise ValueError(_TOO_MANY_UNIQUE_Y_VALS)
    if pos_label not in my_labels and len(my_labels) == 2:
        raise ValueError(_NEED_POS_LABEL_IN_Y_VALS)
    if my_labels[1] != pos_label:
        my_labels = list(reversed(my_labels))
    return my_labels


def true_positive_rate(y_true, y_pred, sample_weight=None, pos_label=1):
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
        The value to treat as the 'positive' label in the samples. Defaults to 1.

    Returns
    -------
    float
        The true positive rate for the data
    """
    unique_labels = _get_labels_for_confusion_matrix(y_true, y_pred, pos_label)
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred,
        sample_weight=sample_weight, labels=unique_labels, normalize="true").ravel()
    return tpr


def true_negative_rate(y_true, y_pred, sample_weight=None, pos_label=1):
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
        The value to treat as the 'positive' label in the samples. Defaults to 1.

    Returns
    -------
    float
        The true negative rate for the data
    """
    unique_labels = _get_labels_for_confusion_matrix(y_true, y_pred, pos_label)
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred,
        sample_weight=sample_weight, labels=unique_labels, normalize="true").ravel()
    return tnr


def false_positive_rate(y_true, y_pred, sample_weight=None, pos_label=1):
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
        The value to treat as the 'positive' label in the samples. Defaults to 1.

    Returns
    -------
    float
        The false positive rate for the data
    """
    unique_labels = _get_labels_for_confusion_matrix(y_true, y_pred, pos_label)
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred,
        sample_weight=sample_weight, labels=unique_labels, normalize="true").ravel()
    return fpr


def false_negative_rate(y_true, y_pred, sample_weight=None, pos_label=1):
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
        The value to treat as the 'positive' label in the samples. Defaults to 1.

    Returns
    -------
    float
        The false negative rate for the data
    """
    unique_labels = _get_labels_for_confusion_matrix(y_true, y_pred, pos_label)
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred,
        sample_weight=sample_weight, labels=unique_labels, normalize="true").ravel()
    return fnr


def _root_mean_squared_error(y_true, y_pred, **kwargs):
    r"""Calculate the root mean squared error."""
    return skm.mean_squared_error(y_true, y_pred, squared=False, **kwargs)
