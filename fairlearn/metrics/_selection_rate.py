# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from ._metrics_engine import metric_by_group


def selection_rate(y_true, y_pred, *, pos_label=1, sample_weight=None):
    """Calculate the fraction of predicted labels matching the 'good' outcome.

    The argument `pos_label` specifies the 'good' outcome.
    """
    selected = (np.squeeze(np.asarray(y_pred)) == pos_label)
    s_w = np.ones(len(selected))
    if sample_weight is not None:
        s_w = np.squeeze(np.asarray(sample_weight))

    return np.dot(selected, s_w) / s_w.sum()


def group_selection_rate(y_true, y_pred, group_membership,
                         *, pos_label=1, sample_weight=None):
    """Wrap :func:`selection_rate` as a group metric.

    The arguments are the same, with the addition of the
    `group_membership` array.
    """
    def internal_sel_wrapper(y_true, y_pred, sample_weight=None):
        return selection_rate(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight)

    return metric_by_group(internal_sel_wrapper,
                           y_true, y_pred, group_membership,
                           sample_weight=sample_weight)
