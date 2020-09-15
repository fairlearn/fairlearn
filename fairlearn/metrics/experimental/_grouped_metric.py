# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

from ._sensitive_feature import SensitiveFeature

_BAD_FEATURE_LENGTH = "Received a feature of length {0} when length {1} was expected"
_TOO_MANY_FEATURE_DIMS = "Feature array has too many dimensions"


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

    def _check_feature_length(self, feature, expected_length):
        if len(feature) != expected_length:
            msg = _BAD_FEATURE_LENGTH.format(len(feature), expected_length)
            raise ValueError(msg)

    def _process_features(self, features, expected_length):
        result = []

        if isinstance(features, pd.Series):
            self._check_feature_length(features, expected_length)
            result.append(SensitiveFeature(features, 0, None))
        elif isinstance(features, pd.DataFrame):
            for i in range(len(features.columns)):
                column = features.iloc[:, i]
                self._check_feature_length(column, expected_length)
                result.append(SensitiveFeature(column, i, None))
        else:
            # Need to specify dtype to avoid inadvertent type conversions
            f_arr = np.squeeze(np.asarray(features, dtype=np.object))
            if len(f_arr.shape) == 1:
                self._check_feature_length(f_arr, expected_length)
                result.append(SensitiveFeature(f_arr, 0, None))
            elif len(f_arr.shape) == 2:
                for i in range(f_arr.shape[0]):
                    col = f_arr[i, :]
                    self._check_feature_length(col, expected_length)
                    result.append(SensitiveFeature(col, i, None))
            else:
                raise ValueError(_TOO_MANY_FEATURE_DIMS)

        return result
