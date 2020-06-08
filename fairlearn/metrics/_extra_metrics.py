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

_NEED_TWO_UNIQUE_Y_VALS = "Must have two unique y values"
_NEED_POS_LABEL_IN_Y_VALS = "Must have pos_label in y values"

def _get_labels_for_confusion_matrix(y_true, y_pred, pos_label):
    """Figure out the labels argument for skm.confusion_matrix.
    This assumes the input arrays are binary, and we need to specify a pos_label.
    Given the way the calling routines work, this is achieved by ensuring that
    the pos_label is last in the (two element) list."""
    my_labels = list(np.unique(np.concatenate((y_true, y_pred), axis=None)))
    if len(my_labels) != 2:
        raise ValueError(_NEED_TWO_UNIQUE_Y_VALS)
    if pos_label not in my_labels:
        raise ValueError(_NEED_TWO_UNIQUE_Y_VALS)
    if my_labels[1] != pos_label:
        my_labels = list(reversed(my_labels))
    return my_labels


def true_positive_rate(y_true, y_pred, sample_weight=None, pos_label=1):
    r"""Calculate the true positive rate (also called sensitivity, recall, or hit rate)."""
    my_labels = _get_labels_for_confusion_matrix(y_true, y_pred, pos_label)
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=my_labels, normalize="true").ravel()
    return tpr


def true_negative_rate(y_true, y_pred, sample_weight=None, pos_label=1):
    r"""Calculate the true negative rate (also called specificity or selectivity)."""
    my_labels = _get_labels_for_confusion_matrix(y_true, y_pred, pos_label)
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=my_labels, normalize="true").ravel()
    return tnr


def false_positive_rate(y_true, y_pred, sample_weight=None, pos_label=1):
    r"""Calculate the false positive rate (also called fall-out)."""
    my_labels = _get_labels_for_confusion_matrix(y_true, y_pred, pos_label)
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=my_labels, normalize="true").ravel()
    return fpr


def false_negative_rate(y_true, y_pred, sample_weight=None, pos_label=1):
    r"""Calculate the false negative rate (also called miss rate)."""
    my_labels = _get_labels_for_confusion_matrix(y_true, y_pred, pos_label)
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=my_labels, normalize="true").ravel()
    return fnr


def _root_mean_squared_error(y_true, y_pred, **kwargs):
    r"""Calculate the root mean squared error."""
    return skm.mean_squared_error(y_true, y_pred, squared=False, **kwargs)
