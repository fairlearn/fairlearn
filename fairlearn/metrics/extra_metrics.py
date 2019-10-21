# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import sklearn.metrics as skm

from .metrics_engine import metric_by_group


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

# =====


def selection_rate(y_true, y_pred, pos_label=1, sample_weight=None):
    """
    The selection rate is the fraction of predicted labels which
    match the 'good' outcome (as specified by pos_label)
    """
    selected = (np.squeeze(np.asarray(y_pred)) == pos_label)
    s_w = np.ones(len(selected))
    if sample_weight is not None:
        s_w = np.squeeze(np.asarray(sample_weight))

    return np.dot(selected, s_w) / s_w.sum()


def group_selection_rate(y_true, y_pred, group_membership,
                         pos_label=1, sample_weight=None):

    def internal_sel_wrapper(y_true, y_pred, sample_weight=None):
        return selection_rate(y_true, y_pred, pos_label, sample_weight=sample_weight)

    return metric_by_group(internal_sel_wrapper,
                           y_true, y_pred, group_membership,
                           sample_weight=sample_weight)
