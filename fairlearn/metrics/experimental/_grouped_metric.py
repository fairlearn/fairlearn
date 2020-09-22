# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import copy
import numpy as np
import pandas as pd

from ._function_container import FunctionContainer
from ._group_feature import GroupFeature

_BAD_FEATURE_LENGTH = "Received a feature of length {0} when length {1} was expected"
_TOO_MANY_FEATURE_DIMS = "Feature array has too many dimensions"


class GroupedMetric:
    def __init__(self, metric_functions,
                 y_true, y_pred, *,
                 sensitive_features,
                 conditional_features=None,
                 sample_param_names=None,
                 params=None):
        func_dict = self._process_functions(metric_functions, sample_param_names, params)

        # Now, prepare the sensitive features
        sf_list = self._process_features("SF", sensitive_features, len(y_true))

        cf_list = None
        self._cf_names = None
        if conditional_features is not None:
            cf_list = self._process_features("CF", conditional_features, len(y_true))
            self._cf_names = [x.name for x in cf_list]

        self._overall = self._compute_overall(func_dict, y_true, y_pred, cf_list)
        self._by_group = self._compute_by_group(func_dict, y_true, y_pred, sf_list, cf_list)

    def _compute_overall(self, func_dict, y_true, y_pred, cf_list):
        if cf_list is None:
            result = pd.DataFrame(index=['overall'], columns=func_dict.keys())
            for func_name in func_dict:
                metric_value = func_dict[func_name].evaluate_all(y_true, y_pred)
                result[func_name]['overall'] = metric_value
        else:
            result = self._compute_dataframe_from_rows(func_dict, y_true, y_pred, cf_list)
        return result

    def _compute_by_group(self, func_dict, y_true, y_pred, sf_list, cf_list):
        rows = copy.deepcopy(sf_list)
        if cf_list is not None:
            # Prepend the conditional features, so they are 'higher'
            rows = copy.deepcopy(cf_list) + rows

        return self._compute_dataframe_from_rows(func_dict, y_true, y_pred, rows)

    def _compute_dataframe_from_rows(self, func_dict, y_true, y_pred, rows):
        row_index = pd.MultiIndex.from_product([x.classes for x in rows],
                                               names=[x.name for x in rows])

        result = pd.DataFrame(index=row_index, columns=func_dict.keys())
        for func_name in func_dict:
            for row_curr in row_index:
                mask = self._mask_from_tuple(row_curr, rows)
                curr_metric = func_dict[func_name].evaluate(y_true, y_pred, mask)
                result[func_name][row_curr] = curr_metric
        return result

    @property
    def overall(self):
        return self._overall

    @property
    def by_group(self):
        return self._by_group

    def group_max(self):
        return self.by_group.groupby(level=list(range(len(self._cf_names)))).max()

    def group_min(self):
        if self._cf_names is None:
            result = pd.DataFrame(index=['overall'], columns=self.by_group.columns)
            for c in result.columns:
                min_val = self.by_group[c].min()
                result[c]['overall'] = min_val
        else:
            result = self.by_group.groupby(level=list(range(len(self._cf_names)))).min()
        return result

    def difference(self):
        return self.group_max() - self.group_min()

    def ratio(self):
        pass

    def _check_feature_length(self, feature, expected_length):
        if len(feature) != expected_length:
            msg = _BAD_FEATURE_LENGTH.format(len(feature), expected_length)
            raise ValueError(msg)

    def _process_functions(self, metric_functions, sample_param_names, params):
        func_dict = dict()
        if isinstance(metric_functions, list):
            # Verify the arguments
            spn = np.full(len(metric_functions), fill_value=None)
            if sample_param_names is not None:
                assert isinstance(sample_param_names, list)
                assert len(metric_functions) == len(sample_param_names)
                spn = sample_param_names
            prms = np.full(len(metric_functions), fill_value=None)
            if params is not None:
                assert isinstance(params, list)
                for p in params:
                    assert isinstance(p, dict)
                assert len(metric_functions) == len(params)
                prms = params

            # Iterate
            for i in range(len(metric_functions)):
                fc = FunctionContainer(metric_functions[i], None, spn[i], prms[i])
                assert fc.name not in func_dict
                func_dict[fc.name] = fc
        else:
            fc = FunctionContainer(metric_functions, None, sample_param_names, params)
            func_dict[fc.name] = fc
        return func_dict

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
        elif isinstance(features, list):
            if np.isscalar(features[0]):
                f_arr = np.squeeze(np.asarray(features))
                assert len(f_arr.shape) == 1
                self._check_feature_length(f_arr, expected_length)
                result.append(GroupFeature(base_name, f_arr, 0, None))
            else:
                for i in range(len(features)):
                    if isinstance(features[i], pd.Series):
                        self._check_feature_length(features[i], expected_length)
                        result.append(GroupFeature(base_name, features[i], i, None))
                    else:
                        f_arr = np.squeeze(np.asarray(features[i]))
                        assert len(f_arr.shape) == 1
                        self._check_feature_length(f_arr, expected_length)
                        result.append(GroupFeature(base_name, f_arr, i, None))
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
