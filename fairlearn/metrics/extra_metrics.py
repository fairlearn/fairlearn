# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from .balanced_root_mean_squared_error import balanced_root_mean_squared_error  # noqa: F401
from .mean_predictions import mean_prediction, mean_overprediction, mean_underprediction  # noqa: F401,E501
from .selection_rate import selection_rate  # noqa: F401,E501

import sklearn.metrics as skm


def specificity_score(y_true, y_pred, sample_weight=None):
    """
    The specificity score is also known as the True Negative Rate.
    At the present time, this routine only supports binary
    classifiers with labels taken from {0, 1}
    """
    cm = skm.confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    # Taken from
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # This restricts us to binary classification
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn+fp)


def miss_rate(y_true, y_pred, sample_weight=None):
    """
    The miss rate is also known as the False Negative Rate.
    At the present time, this routine only supports binary
    classifiers with labels taken from {0, 1}
    """
    # aka False Negative Rate
    tpr = skm.recall_score(y_true, y_pred, sample_weight=sample_weight)

    # FNR == 1 - TPR
    return 1 - tpr


def fallout_rate(y_true, y_pred, sample_weight=None):
    """
    The fallout rate is also known as the False Positive Rate.
    At the present time, this routine only supports binary
    classifiers with labels taken from {0, 1}
    """
    # aka False Positive Rate
    # Since we use specificity, also restricted to binary classification
    return 1 - specificity_score(y_true, y_pred, sample_weight)
