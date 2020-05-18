# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""A variety of extra metrics useful for assessing fairness.

These are metrics which are not part of `scikit-learn`.
"""

import sklearn.metrics as skm

from ._balanced_root_mean_squared_error import _balanced_root_mean_squared_error  # noqa: F401
from ._mean_predictions import mean_prediction, _mean_overprediction, _mean_underprediction  # noqa: F401,E501
from ._selection_rate import selection_rate  # noqa: F401,E501


def true_positive_rate(y_true, y_pred, sample_weight=None):
    r"""Calculate the true positive rate (also called sensitivity, recall, or hit rate)."""
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=[0, 1], normalize="true").ravel()
    return tpr


def true_negative_rate(y_true, y_pred, sample_weight=None):
    r"""Calculate the true negative rate (also called specificity or selectivity)."""
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=[0, 1], normalize="true").ravel()
    return tnr


def false_positive_rate(y_true, y_pred, sample_weight=None):
    r"""Calculate the false positive rate (also called fall-out)."""
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=[0, 1], normalize="true").ravel()
    return fpr


def false_negative_rate(y_true, y_pred, sample_weight=None):
    r"""Calculate the false negative rate (also called miss rate)."""
    tnr, fpr, fnr, tpr = skm.confusion_matrix(
        y_true, y_pred, sample_weight=sample_weight, labels=[0, 1], normalize="true").ravel()
    return fnr


def _root_mean_squared_error(y_true, y_pred, **kwargs):
    r"""Calculate the root mean squared error."""
    return skm.mean_squared_error(y_true, y_pred, squared=False, **kwargs)
