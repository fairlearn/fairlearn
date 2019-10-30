# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""This module contains a variety of extra metrics which are useful for assessing fairness
which are not available as part of scikit-learn.
"""

import sklearn.metrics as skm

from ._balanced_root_mean_squared_error import balanced_root_mean_squared_error  # noqa: F401
from ._mean_predictions import mean_prediction, mean_overprediction, mean_underprediction  # noqa: F401,E501
from ._selection_rate import selection_rate  # noqa: F401,E501


def specificity_score(y_true, y_pred, sample_weight=None):
    """
    The specificity score is also known as the True Negative Rate.
    At the present time, this routine only supports binary
    classifiers with labels taken from {0, 1}.
    The calculation uses the :any:`sklearn.metrics.confusion_matrix` routine.
    """
    cm = skm.confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    # Taken from
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # This restricts us to binary classification
    tn, fp, _, _ = cm.ravel()
    return tn / (tn + fp)


def miss_rate(y_true, y_pred, sample_weight=None):
    """
    The miss rate is also known as the False Negative Rate.
    At the present time, this routine only supports binary
    classifiers with labels taken from {0, 1}.
    By definition, this is the complement of the True Positive
    Rate, so this routine uses the
    :any:`sklearn.metrics.recall_score` routine.
    """
    # aka False Negative Rate
    tpr = skm.recall_score(y_true, y_pred, sample_weight=sample_weight)

    # FNR == 1 - TPR
    return 1 - tpr


def fallout_rate(y_true, y_pred, sample_weight=None):
    """
    The fallout rate is also known as the False Positive Rate.
    At the present time, this routine only supports binary
    classifiers with labels taken from {0, 1}.
    By definition, this is the complement of the
    Specificity, and so uses :any:`specificity_score` in its
    calculation.
    """
    # aka False Positive Rate
    # Since we use specificity, also restricted to binary classification
    return 1 - specificity_score(y_true, y_pred, sample_weight)
