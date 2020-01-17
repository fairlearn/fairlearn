# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""A variety of extra metrics useful for assessing fairness.

These are metrics which are not part of `scikit-learn`.
"""

import sklearn.metrics as skm

from ._metrics_engine import make_group_metric

from ._balanced_root_mean_squared_error import balanced_root_mean_squared_error  # noqa: F401
from ._mean_predictions import mean_prediction, mean_overprediction, mean_underprediction  # noqa: F401,E501
from ._selection_rate import selection_rate  # noqa: F401,E501


def specificity_score(y_true, y_pred, sample_weight=None):
    r"""Calculate the specificity score (also called the True Negative Rate).

    At the present time, this routine only supports binary
    classifiers with labels :math:`\in {0, 1}`.
    The calculation uses the :py:func:`sklearn.metrics.confusion_matrix` routine.
    """
    cm = skm.confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    # Taken from
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # This restricts us to binary classification
    tn, fp, _, _ = cm.ravel()
    return tn / (tn + fp)


def miss_rate(y_true, y_pred, sample_weight=None):
    r"""Calculate the miss rate (also called the False Negative Rate).

    At the present time, this routine only supports binary
    classifiers with labels :math:`\in {0, 1}`.
    By definition, this is the complement of the True Positive
    Rate, so this routine uses the
    :py:func:`sklearn.metrics.recall_score` routine.
    """
    # aka False Negative Rate
    tpr = skm.recall_score(y_true, y_pred, sample_weight=sample_weight)

    # FNR == 1 - TPR
    return 1 - tpr


def fallout_rate(y_true, y_pred, sample_weight=None):
    r"""Calculate the fallout rate (also called the False Positive Rate).

    At the present time, this routine only supports binary
    classifiers with labels :math:`\in {0, 1}`.
    By definition, this is the complement of the
    Specificity, and so uses :py:func:`specificity_score` in its
    calculation.
    """
    # aka False Positive Rate
    # Since we use specificity, also restricted to binary classification
    return 1 - specificity_score(y_true, y_pred, sample_weight)

# =============================================================
# Group metrics

# Classification metrics


group_fallout_rate = make_group_metric(fallout_rate)
"""A grouped metric for the :py:func:`fallout_rate`
"""

group_miss_rate = make_group_metric(miss_rate)
"""A grouped metric for the :py:func:`miss_rate`
"""
group_specificity_score = make_group_metric(specificity_score)
"""A grouped metric for the :py:func:`specificity_score`
"""

# Regression metrics

group_balanced_root_mean_squared_error = make_group_metric(
    balanced_root_mean_squared_error)
"""A grouped wrapper around the :py:func:`balanced_root_mean_squared_error` routine
"""

group_mean_prediction = make_group_metric(mean_prediction)
"""A grouped wrapper around the :py:func:`mean_prediction` routine
"""

group_mean_overprediction = make_group_metric(mean_overprediction)
"""A grouped wrapper around the :py:func:`mean_overprediction` routine
"""

group_mean_underprediction = make_group_metric(mean_underprediction)
"""A grouped wapper around the :py:func:`mean_underprediction` routine
"""
