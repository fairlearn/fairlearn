# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sklearn.metrics as skm


def specificity_score(y_true, y_pred, sample_weight=None):
    # aka True Negative Rate
    cm = skm.confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    # Taken from
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # This restricts us to binary classification
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn+fp)


def miss_score(y_true, y_pred, sample_weight=None):
    # aka False Negative Rate
    tpr = skm.recall_score(y_true, y_pred, sample_weight=sample_weight)

    # FNR == 1 - TPR
    return 1 - tpr


def fallout_score(y_true, y_pred, sample_weight=None):
    # aka False Positive Rate
    # Since we use specificity, also restricted to binary classification
    return 1 - specificity_score(y_true, y_pred, sample_weight)
