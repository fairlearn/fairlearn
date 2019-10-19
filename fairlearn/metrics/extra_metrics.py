# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sklearn.metrics as skm


def specificity_score(y_true, y_pred, sample_weight=None):
    # aka True Negative Rate
    cm = skm.confusion_matrix(y_true, y_pred, sample_weight)
    # Taken from
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn+fp)
