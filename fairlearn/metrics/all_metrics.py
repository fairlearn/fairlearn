# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from sklearn.metrics import recall_score

from . import MetricsResult


def true_positive_rate(y_actual, y_predict, group_id, sample_weight=None):
    return metric_by_groups(recall_score, y_actual, y_predict, group_id, sample_weight)


def selection_rate(y_actual, y_predict, group_id, sample_weight=None):
    if sample_weight is not None:
        raise NotImplementedError("selection_rate and sample_weight")

    def selection_metric(y_a, y_p, sample_weight):
        return np.mean(y_p)

    return metric_by_groups(selection_metric, y_actual, y_predict, group_id, sample_weight)


def metric_by_groups(metric_function, y_actual, y_predict, group_id, sample_weight=None):
    # TODO: Validate y_actual and y_predict are from {0, 1}
    # TODO: Validate that group_id are from {0 ... n}
    result = MetricsResult()

    groups = np.unique(group_id)
    number_of_groups = np.max(groups) + 1

    result.metric = metric_function(y_actual, y_predict, sample_weight=sample_weight)

    # Initialise the group results array
    result.group_metric = np.full(number_of_groups, fill_value=float('nan'))

    # The slicing we use requires Numpy arrays
    y_a = np.array(y_actual)
    y_p = np.array(y_predict)
    for group in groups:
        group_indices = [i for i, elem in enumerate(group_id) if elem == group]
        group_actual = y_a[group_indices]
        group_predict = y_p[group_indices]
        result.group_metric[group] = metric_function(
            group_actual,
            group_predict,
            sample_weight=sample_weight)

    return result
