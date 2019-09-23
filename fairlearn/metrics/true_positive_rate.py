# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from . import MetricsResult


def true_positive_rate(y_actual, y_predict, group_id):
    # TODO: Validate y_actual and y_predict are from {0, 1}
    # TODO: Validate that group_id are from {0 ... n}
    result = MetricsResult()

    groups = np.unique(group_id)

    positives_all = 0
    true_positives_all = 0

    positives = np.zeros(len(groups))
    true_positives = np.zeros(len(groups))
    result.group_metric = np.zeros(len(groups))

    for y, yhat, a in zip(y_actual, y_predict, group_id):
        if y == 1:
            positives_all += 1
            positives[a] += 1
            if yhat == 1:
                true_positives_all += 1
                true_positives[a] += 1

    result.metric = true_positives_all / positives_all
    for group in groups:
        result.group_metric[group] = true_positives[group] / positives[group]

    return result
