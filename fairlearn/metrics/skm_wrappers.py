# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sklearn.metrics as skm

from .metrics_engine import metric_by_group


def group_accuracy_score(y_true, y_pred, group_membership, normalize=True, sample_weight=None):

    def internal_wrapper(y_true, y_pred, sample_weight=None):
        return skm.accuracy_score(y_true, y_pred, normalize, sample_weight)

    return metric_by_group(internal_wrapper, y_true, y_pred, group_membership, sample_weight)
