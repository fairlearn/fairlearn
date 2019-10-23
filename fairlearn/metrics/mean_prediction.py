# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np


def mean_prediction(y_true, y_pred, sample_weight=None):
    """Returns the (weighted) mean prediction. The true
    values are ignored, but required as an argument in order
    to maintain a consistent interface
    """

    p = np.squeeze(np.asarray(y_pred))
    s_w = np.ones(len(p))
    if sample_weight is not None:
        s_w = np.squeeze(np.asarray(sample_weight))

    return np.dot(p, s_w) / s_w.sum()
