# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import copy
import logging
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, Optional, Union
from sklearn.utils import check_consistent_length

from fairlearn.metrics._input_manipulations import _convert_to_ndarray_and_squeeze
from ._function_container import FunctionContainer
from ._group_feature import GroupFeature

logger = logging.getLogger(__name__)

_SUBGROUP_COUNT_WARNING_THRESHOLD = 20

_BAD_FEATURE_LENGTH = "Received a feature of length {0} when length {1} was expected"
_SUBGROUP_COUNT_WARNING = "Found {0} subgroups. Evaluation may be slow"
_TOO_MANY_FEATURE_DIMS = "Feature array has too many dimensions"


class MetricsFrame:
    """Contains a collection of group fairness metrics.

    This class holds grouped metrics calculated for any number of underlying
    metrics. At least one sensitive feature must be supplied, which is used
    to split the data into subgroups. The underlying metric(s) is(are) calculated
    across the entire dataset (made available by the ``overall`` property) and
    for each identified subgroup (made available by the ``by_group`` property).

    Parameters
    ----------
    metric_functions :
        The underlying metric functions which are to be calculated. This
        can either be a single metric function or a dictionary of functions.
        These functions must be callable as
        ``fn(y_true, y_pred, **sample_params)``.
        If there are any other arguments required (such as ``beta`` for
        :func:`sklearn.metrics.fbeta_score`) then
        :func:`functools.partial` must be used.

    y_true : array_like
        The true values

    y_pred : array_like
        The predicted values

    sensitive_features : It's complicated
        The sensitive features which should be used to create the subgroups.
        At least one sensitive feature must be provided. Allowed types are
        pandas Series, DataFrames, lists of Series, numpy arrays, lists of
        numpy arrays and lists of lists (of scalars). If the supplied type
        has associated names (Series or DataFrames) then the corresponding
        sensitive feature will take its name from that. Otherwise, a name
        of the format ``SF [n]`` will be generated, with ``n`` indicating
        the index of the feature.

    conditional_features : It's complicated
        TODO Similar to sensitive_features, but....

    sample_params :
        Parameters for the metric function(s). If there is only one metric function,
        then this is a dictionary of strings and array_likes, which are split
        alongside the ``y_true`` and ``y_pred`` arrays, and passed to the metric function.
        If there are multiple metric functions (passed as a dictioary), then this is
        a nested dictionary, with the first set of string keys identifying the
        metric function name, with the values being the string-to-array_like dictionaries.
    """

    def __init__(self,
                 metric_functions: Union[Callable, Dict[str, Callable]],
                 y_true,
                 y_pred, *,
                 sensitive_features,
                 conditional_features: Optional = None,
                 sample_params: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None):
        """Read a placeholder comment."""
        check_consistent_length(y_true, y_pred)
        y_t = _convert_to_ndarray_and_squeeze(y_true)
        y_p = _convert_to_ndarray_and_squeeze(y_pred)

        func_dict = self._process_functions(metric_functions, sample_params)

        # Now, prepare the sensitive features
        sf_list = self._process_features("SF", sensitive_features, y_t)

        cf_list = None
        self._cf_names = None
        if conditional_features is not None:
            cf_list = self._process_features("CF", conditional_features, y_t)
            self._cf_names = [x.name for x in cf_list]

        self._overall = self._compute_overall(func_dict, y_t, y_p, cf_list)
        self._by_group = self._compute_by_group(func_dict, y_t, y_p, sf_list, cf_list)

    def _compute_overall(self, func_dict, y_true, y_pred, cf_list):
        if cf_list is None:
            result = pd.Series(index=func_dict.keys(), dtype='object')
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

        if len(row_index) > _SUBGROUP_COUNT_WARNING_THRESHOLD:
            msg = _SUBGROUP_COUNT_WARNING.format(len(row_index))
            logger.warning(msg)

        result = pd.DataFrame(index=row_index, columns=func_dict.keys())
        for func_name in func_dict:
            for row_curr in row_index:
                mask = None
                if len(rows) > 1:
                    mask = self._mask_from_tuple(row_curr, rows)
                else:
                    # Have to force row_curr to be an unary tuple
                    mask = self._mask_from_tuple((row_curr,), rows)

                # Only call the metric function if the mask is non-empty
                if sum(mask) > 0:
                    curr_metric = func_dict[func_name].evaluate(y_true, y_pred, mask)
                    result[func_name][row_curr] = curr_metric
        return result

    @ property
    def overall(self) -> Union[pd.Series, pd.DataFrame]:
        """Return the underlying metrics evaluated on the whole dataset.

        If TODO no ``conditional_features`` were specified, then this is
        a :class:`pandas.Series` with one row for each underlying metric.

        If TODO ``conditional_features`` were specified, then this is a
        :class:`pandas.DataFrame`, with columns corresponding to the
        underling metric(s) and the rows indexed by the combinations of
        the TODO conditional features.
        """
        return self._overall

    @ property
    def by_group(self) -> pd.DataFrame:
        """Read a placeholder comment."""
        return self._by_group

    def group_max(self) -> Union[pd.Series, pd.DataFrame]:
        """Read a placeholder comment."""
        if self._cf_names is None:
            result = pd.Series(index=self.by_group.columns, dtype='object')
            for m in result.index:
                max_val = self.by_group[m].max()
                result[m] = max_val
        else:
            lvls = list(range(len(self._cf_names)))
            result = self.by_group.groupby(level=lvls).max()
        return result

    def group_min(self) -> Union[pd.Series, pd.DataFrame]:
        """Read a placeholder comment."""
        if self._cf_names is None:
            result = pd.Series(index=self.by_group.columns, dtype='object')
            for m in result.index:
                min_val = self.by_group[m].min()
                result[m] = min_val
        else:
            lvls = list(range(len(self._cf_names)))
            result = self.by_group.groupby(level=lvls).min()
        return result

    def difference(self,
                   method: str) -> Union[pd.Series, pd.DataFrame]:
        """Read a placeholder comment."""
        subtrahend = np.nan
        if method == 'between_pairs':
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

    def ratio(self,
              method: str) -> Union[pd.Series, pd.DataFrame]:
        """Read a placeholder comment."""
        result = None
        if method == 'between_pairs':
            result = self.group_min() / self.group_max()
        elif method == 'to_overall':
            ratios = self.by_group / self.overall

            def ratio_sub_one(x):
                if x > 1:
                    return 1/x
                else:
                    return x

            ratios = ratios.apply(lambda x: x.transform(ratio_sub_one))
            if self._cf_names is None:
                result = ratios.min()
            else:
                # It's easiest to give in to the DataFrame columns preference
                cf_levels = list(range(len(self._cf_names)))
                result = ratios.unstack(level=cf_levels).min().unstack(0)
        else:
            raise ValueError("Unrecognised method '{0}' in ratio() call".format(method))

        return result

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

    def _process_features(self, base_name, features, sample_array):
        result = []

        if isinstance(features, pd.Series):
            check_consistent_length(features, sample_array)
            result.append(GroupFeature(base_name, features, 0, None))
        elif isinstance(features, pd.DataFrame):
            for i in range(len(features.columns)):
                column = features.iloc[:, i]
                check_consistent_length(column, sample_array)
                result.append(GroupFeature(base_name, column, i, None))
        elif isinstance(features, list):
            if np.isscalar(features[0]):
                f_arr = np.squeeze(np.asarray(features))
                assert len(f_arr.shape) == 1
                check_consistent_length(f_arr, sample_array)
                result.append(GroupFeature(base_name, f_arr, 0, None))
            else:
                for i in range(len(features)):
                    if isinstance(features[i], pd.Series):
                        check_consistent_length(features[i], sample_array)
                        result.append(GroupFeature(base_name, features[i], i, None))
                    else:
                        f_arr = np.squeeze(np.asarray(features[i]))
                        assert len(f_arr.shape) == 1
                        check_consistent_length(f_arr, sample_array)
                        result.append(GroupFeature(base_name, f_arr, i, None))
        else:
            # Need to specify dtype to avoid inadvertent type conversions
            f_arr = np.squeeze(np.asarray(features, dtype=np.object))
            if len(f_arr.shape) == 1:
                check_consistent_length(f_arr, sample_array)
                result.append(GroupFeature(base_name, f_arr, 0, None))
            elif len(f_arr.shape) == 2:
                for i in range(f_arr.shape[0]):
                    col = f_arr[i, :]
                    check_consistent_length(col, sample_array)
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
