# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np


def selection_rate(y_true, y_pred, *, pos_label=1, sample_weight=None):
    """Calculate the fraction of predicted labels matching the 'good' outcome.

    The argument `pos_label` specifies the 'good' outcome.
    """
    y_pred_arr = np.asarray(y_pred)

    if y_pred_arr.shape == (0,):
        raise ValueError("Empty y_pred passed to selection_rate function.")

    y_pred_arr = np.squeeze(y_pred_arr)
    if y_pred_arr.shape == ():
        # not actually an array, but rather a single entry that was compressed
        # into a scalar by the squeeze call
        return (y_pred_arr == pos_label) * 1.0

    selected = (y_pred_arr == pos_label)
    s_w = np.ones(len(selected))
    if sample_weight is not None:
        s_w = np.squeeze(np.asarray(sample_weight))

    return np.dot(selected, s_w) / s_w.sum()
