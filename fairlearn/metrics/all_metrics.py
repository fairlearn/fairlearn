# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from sklearn.metrics import recall_score

from . import MetricsResult


def true_positive_rate(y_actual, y_predict, group_id):
    # TODO: Validate y_actual and y_predict are from {0, 1}
    # TODO: Validate that group_id are from {0 ... n}
    result = MetricsResult()

    groups = np.unique(group_id)
    number_of_groups = np.max(groups) + 1

    positives = np.zeros(number_of_groups)
    true_positives = np.zeros(number_of_groups)
    result.group_metric = np.zeros(number_of_groups)

    for y, yhat, a in zip(y_actual, y_predict, group_id):
        if y == 1:
            positives[a] += 1
            if yhat == 1:
                true_positives[a] += 1

    result.metric = recall_score(y_actual, y_predict)
    for group in groups:
        result.group_metric[group] = true_positives[group] / positives[group]

    return result
