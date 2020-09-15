# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd


class GroupedMetric:
    def __init__(self, metric_functions,
                 y_true, y_pred, *,
                 sensitive_features,
                 conditional_features=None,
                 sample_param_names=[],
                 params=dict()):
        metric_name = metric_functions.__name__
        self._overall = pd.DataFrame(data=metric_functions(y_true, y_pred, **params),
                                     index=['overall'],
                                     columns=[metric_name])

        groups = np.unique(sensitive_features)
        result_by_group = {}
        for group in groups:
            group_indices = (group == np.asarray(sensitive_features))
            result_by_group[group] = metric_functions(
                y_true[group_indices], y_pred[group_indices],
                **params)

        self._by_group = pd.DataFrame.from_dict(data=result_by_group,
                                                orient='index',
                                                columns=[metric_name])

    @property
    def overall(self):
        return self._overall

    @property
    def by_group(self):
        return self._by_group
