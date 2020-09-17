# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

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

        cf_list = None
        if conditional_features is None:
            metrics = {}
            for fc in func_dict.values():
                metrics[fc.name] = fc.evaluate_all(y_true, y_pred)
            self._overall = pd.DataFrame(data=metrics,
                                         index=['overall'],
                                         columns=func_dict.keys())
        else:
            cf_list = self._process_features("CF", conditional_features, len(y_true))
            cf_index = pd.MultiIndex.from_product([x.classes for x in cf_list],
                                                  names=[x.name for x in cf_list])

            self._overall = pd.DataFrame(index=cf_index, columns=[
                                         x.name for x in func_dict.values()])
            for func_name in func_dict:
                for cf_curr in cf_index:
                    mask = self._mask_from_tuple(cf_curr, cf_list)
                    curr_metric = func_dict[func_name].evaluate(y_true, y_pred, mask)

                    self._overall[func_name][cf_curr] = curr_metric

        # Now, prepare the sensitive features
        sf_list = self._process_features("SF", sensitive_features, len(y_true))
        sf_index = pd.MultiIndex.from_product([x.classes for x in sf_list],
                                              names=[x.name for x in sf_list])

        col_lists = [list(func_dict.keys())]
        col_titles = ["Metric Name"]
        if cf_list is not None:
            col_lists = col_lists + [x.classes for x in cf_list]
            col_titles = col_titles + [x.name for x in cf_list]
        columns = pd.MultiIndex.from_product(col_lists, names=col_titles)

        metrics = pd.DataFrame(index=sf_index, columns=columns)

        for col_curr in columns:
            current_function = func_dict[col_curr[0]]
            cf_mask = np.full(len(y_true), fill_value=True)
            if cf_list is not None:
                cf_mask = self._mask_from_tuple(col_curr[1:], cf_list)

            for sf_curr in sf_index:
                sf_mask = self._mask_from_tuple(sf_curr, sf_list)
                mask = np.logical_and(cf_mask, sf_mask)

                curr_metric = current_function.evaluate(y_true, y_pred, mask)

                metrics[col_curr][sf_curr] = curr_metric

        self._by_group = metrics

    @property
    def overall(self):
        return self._overall

    @property
    def by_group(self):
        return self._by_group

    def group_max(self):
        return self._by_group.max()

    def group_min(self):
        pass

    def difference(self):
        pass

    def difference_to_overall(self):
        pass

    def ratio(self):
        pass

    def ratio_to_overall(self):
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
