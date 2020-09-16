# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

from ._group_feature import GroupFeature

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

        if conditional_features is not None:
            raise NotImplementedError("conditional_features")

        sf_list = self._process_features("SF", sensitive_features, len(y_true))

        sf_index = pd.MultiIndex.from_product([x.classes for x in sf_list],
                                              names=[x.name for x in sf_list])
        self._by_group = pd.DataFrame(columns=[metric_name], index=sf_index)

        metric_results = []
        for sf_curr in sf_index:
            mask = self._mask_from_tuple(sf_curr, sf_list)

            curr_metric = metric_functions(
                y_true[mask], y_pred[mask],
                **params)

            metric_results.append(curr_metric)

        nxt_column = pd.Series(data=metric_results, index=sf_index, name=metric_name)

        self._by_group = pd.concat([nxt_column], axis=1)

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

    def _process_features(self, base_name, features, expected_length):
        result = []

        if isinstance(features, pd.Series):
            self._check_feature_length(features, expected_length)
            result.append(GroupFeature(base_name, features, 0, None))
        elif isinstance(features, pd.DataFrame):
            for i in range(len(features.columns)):
                column = features.iloc[:, i]
                self._check_feature_length(column, expected_length)
                result.append(GroupFeature(base_name, column, i, None))
        else:
            # Need to specify dtype to avoid inadvertent type conversions
            f_arr = np.squeeze(np.asarray(features, dtype=np.object))
            if len(f_arr.shape) == 1:
                self._check_feature_length(f_arr, expected_length)
                result.append(GroupFeature(base_name, f_arr, 0, None))
            elif len(f_arr.shape) == 2:
                for i in range(f_arr.shape[0]):
                    col = f_arr[i, :]
                    self._check_feature_length(col, expected_length)
                    result.append(GroupFeature(base_name, col, i, None))
            else:
                raise ValueError(_TOO_MANY_FEATURE_DIMS)

        return result

    def _mask_from_tuple(self, index_tuple, feature_list):
        assert len(index_tuple) == len(feature_list)

        result = feature_list[0].get_mask_for_class(index_tuple[0])
        for i in range(1, len(index_tuple)):
            result = np.logical_and(
                result,
                feature_list[i].get_mask_for_class(index_tuple[i]))
        return result
