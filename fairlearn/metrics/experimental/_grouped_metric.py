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
    """Read a placeholder comment."""

    def __init__(self, metric_functions,
                 y_true, y_pred, *,
                 sensitive_features,
                 conditional_features=None,
                 sample_params=None):
        """Read a placeholder comment."""
        func_dict = self._process_functions(metric_functions, sample_params)

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
            result = pd.Series(index=func_dict.keys())
            for func_name in func_dict:
                metric_value = func_dict[func_name].evaluate_all(y_true, y_pred)
                result[func_name] = metric_value
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
        if len(rows) == 1:
            row_index = pd.Index(data=rows[0].classes, name=rows[0].name)
        else:
            row_index = pd.MultiIndex.from_product([x.classes for x in rows],
                                                   names=[x.name for x in rows])

        result = pd.DataFrame(index=row_index, columns=func_dict.keys())
        for func_name in func_dict:
            for row_curr in row_index:
                mask = None
                if len(rows) > 1:
                    mask = self._mask_from_tuple(row_curr, rows)
                else:
                    # Have to force row_curr to be an unary tuple
                    mask = self._mask_from_tuple((row_curr,), rows)
                curr_metric = func_dict[func_name].evaluate(y_true, y_pred, mask)
                result[func_name][row_curr] = curr_metric
        return result

    @property
    def overall(self):
        """Read a placeholder comment."""
        return self._overall

    @property
    def by_group(self):
        """Read a placeholder comment."""
        return self._by_group

    def group_max(self):
        """Read a placeholder comment."""
        if self._cf_names is None:
            result = pd.Series(index=self.by_group.columns)
            for m in result.index:
                max_val = self.by_group[m].max()
                result[m] = max_val
        else:
            result = self.by_group.groupby(level=list(range(len(self._cf_names)))).max()
        return result

    def group_min(self):
        """Read a placeholder comment."""
        if self._cf_names is None:
            result = pd.Series(index=self.by_group.columns)
            for m in result.index:
                min_val = self.by_group[m].min()
                result[m] = min_val
        else:
            result = self.by_group.groupby(level=list(range(len(self._cf_names)))).min()
        return result

    def difference(self, method='minmax'):
        """Read a placeholder comment."""
        subtrahend = np.nan
        if method == 'minmax':
            subtrahend = self.group_min()
        elif method == 'to_overall':
            subtrahend = self.overall
        else:
            raise ValueError("Unrecognised method '{0}' in difference() call".format(method))

        result = None
        if self._cf_names is None:
            result = (self.by_group - subtrahend).abs().max()
        else:
            # It's easiest to give in to the DataFrame columns preference
            cf_levels = list(range(len(self._cf_names)))
            diffs = (self.by_group.unstack(level=cf_levels) -
                     subtrahend.unstack(level=cf_levels)).abs()
            result = diffs.max().unstack(0)

        return result

    def ratio(self):
        """Read a placeholder comment."""
        pass

    def _check_feature_length(self, feature, expected_length):
        if len(feature) != expected_length:
            msg = _BAD_FEATURE_LENGTH.format(len(feature), expected_length)
            raise ValueError(msg)

    def _process_functions(self, metric_functions, sample_params):
        func_dict = dict()
        if isinstance(metric_functions, dict):
            s_p = dict()
            if sample_params is not None:
                assert isinstance(sample_params, dict)
                assert set(sample_params.keys()).issubset(set(metric_functions.keys()))
                s_p = sample_params

            for name, func in metric_functions.items():
                curr_s_p = None
                if name in s_p:
                    curr_s_p = s_p[name]
                fc = FunctionContainer(func, name, curr_s_p)
                func_dict[fc.name_] = fc
        else:
            fc = FunctionContainer(metric_functions, None, sample_params)
            func_dict[fc.name_] = fc
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
        assert isinstance(index_tuple, tuple)
        assert len(index_tuple) == len(feature_list)

        result = feature_list[0].get_mask_for_class(index_tuple[0])
        for i in range(1, len(index_tuple)):
            result = np.logical_and(
                result,
                feature_list[i].get_mask_for_class(index_tuple[i]))
        return result
