# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import copy
import logging
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Union
from sklearn.utils import check_consistent_length

from fairlearn.metrics._input_manipulations import _convert_to_ndarray_and_squeeze
from ._function_container import FunctionContainer, _SAMPLE_PARAMS_NOT_DICT
from ._group_feature import GroupFeature

logger = logging.getLogger(__name__)

_SUBGROUP_COUNT_WARNING_THRESHOLD = 20

_BAD_FEATURE_LENGTH = "Received a feature of length {0} when length {1} was expected"
_SUBGROUP_COUNT_WARNING = "Found {0} subgroups. Evaluation may be slow"
_TOO_MANY_FEATURE_DIMS = "Feature array has too many dimensions"
_SAMPLE_PARAM_KEYS_NOT_IN_FUNC_DICT = \
    "Keys in 'sample_params' do not match those in 'metric_functions'"


class MetricFrame:
    """Contains a collection of group fairness metrics.

    This class holds group fairness metrics calculated for any number of underlying
    metrics. At least one sensitive feature must be supplied, which is used
    to split the data into subgroups. The underlying metric(s) is(are) calculated
    across the entire dataset (made available by the ``overall`` property) and
    for each identified subgroup (made available by the ``by_group`` property).

    The only limitations placed on the metric functions are that:

    * The first two arguments they take must be ``y_true`` and ``y_pred`` arrays
    * Any other arguments must be *sample* based, and will be split up along with
      the ``y_true`` and ``y_pred`` arrays

    The interpretation of the ``y_true`` and ``y_pred`` arrays is up to the
    underlying metric - it is perfectly possible to pass in lists of class
    probability tuples. The return value of the metric is also very flexible;
    there is no problem around returning confusion matrices.

    Once calculated, scalar group fairness metrics can be aggregated over
    sensitive (but not control) features, by taking differences or ratios,
    and then reporting the maximum or minimum respectively.

    This class also supports the concept of 'control features.' Like the sensitive
    features, control features identify subgroups within the data, but
    aggregations are not performed over the control features. Instead, the
    aggregations produce a result for each subgroup identified by the control
    feature(s).

    Parameters
    ----------
    metric_functions : callable or dict
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

    control_features : It's complicated
        Control features can be specified similarly to the sensitive features.
        However, their default names (if none can be identified in the
        input values) are of the format ``CF [n]``.

    sample_params : dict
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
                 control_features: Optional = None,
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
        if control_features is not None:
            cf_list = self._process_features("CF", control_features, y_t)
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

    @property
    def overall(self) -> Union[pd.Series, pd.DataFrame]:
        """Return the underlying metrics evaluated on the whole dataset.

        If no ``control_features`` were specified, then this is
        a :class:`pandas.Series` with one row for each underlying metric.

        If ``control_features`` were specified, then this is a
        :class:`pandas.DataFrame`, with columns corresponding to the
        underling metric(s) and the rows indexed by the subgroups of
        the control features.
        """
        return self._overall

    @property
    def by_group(self) -> pd.DataFrame:
        """Return the collection of metrics evaluated for each subgroup.

        This :class:`pandas.DataFrame` contains the result of evaluating
        each underlying metric for each combination of classes in the
        sensitive and control features. The columns identify
        the underlying metric, while the rows are indexed by the unqiue
        combinations of classes found in the sensitive and control
        features. If control features are present, they are at
        in the 'outer' layer of the row indexing.

        If a particular combination of classes was not present in the dataset
        (likely to occur as more sensitive and control features
        are specified), then the corresponding entry in the DataFrame will
        be NaN.
        """
        return self._by_group

    def group_max(self) -> Union[pd.Series, pd.DataFrame]:
        """Return the maximum value of the metric over the sensitive features.

        This method computes the maximum value over all combinations of
        sensitive features for each underlying metric function in the ``by_groups``
        property (it will only succeed if all the underlying metric
        functions return scalar values). If there are no control
        features, then the result will be a :class:`pandas.Series`,
        with rows corresponding to the underying metrics.

        If there are control features, then the result is a
        :class:`pandas.DataFrame` with columns corresponding to the
        underlying metrics, and rows indexed by the classes identified
        by the control features.
        """
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
        """Return the minimum value of the metric over the sensitive features.

        This method computes the minimum value over all combinations of
        sensitive features for each underlying metric function in the ``by_groups``
        property (it will only succeed if all the underlying metric
        functions return scalar values). If there are no control
        features, then the result will be a :class:`pandas.Series`,
        with rows corresponding to the underying metrics.

        If there are control features, then the result is a
        :class:`pandas.DataFrame` with columns corresponding to the
        underlying metrics, and rows indexed by the classes identified
        by the control features.
        """
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
        """Return the maximum absolute difference between groups for each metric.

        This method calculates a scalar value for each underlying metric by
        finding the maximum absolute difference between the entries in each
        column of the ``by_groups`` DataFrame.

        Similar to other methods, the result will be a :class:`pandas.Series`
        if there are no control features, with rows corresponding
        to each underlying metric. If there are control features,
        then the result will be a :class:`pandas.DataFrame` with a column
        for each underlying metric, and rows corresponding to the subgroups
        identified by the control feature(s).

        There are two allowed values for the ``method=`` parameter. The
        value ``between_groups`` computes the maximum difference between
        any two pairs of groups in the ``by_groups`` DataFrame (i.e.
        ``group_max() - group_min()``). Alternatively, ``to_overall``
        computes the difference between each subgroup and the
        corresponding value from ``overall`` (if there are control
        features, then ``overall`` is multivalued for each metric).
        The result is the absolute maximum of these values.
        """
        subtrahend = np.nan
        if method == 'between_groups':
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
        """Return the minimum ratio between groups for each metric.

        This method calculates a scalar value for each underlying metric by
        finding the minimum ratio (that is, the ratio is forced to be
        less than unity) between the entries in each
        column of the ``by_groups`` DataFrame.

        Similar to other methods, the result will be a :class:`pandas.Series`
        if there are no control features, with rows corresponding
        to each underlying metric. If there are control features,
        then the result will be a :class:`pandas.DataFrame` with a column
        for each underlying metric, and rows corresponding to the subgroups
        identified by the control feature(s).

        There are two allowed values for the ``method=`` parameter. The
        value ``between_groups`` computes the minimum ratio between
        any two pairs of groups in the ``by_groups`` DataFrame (i.e.
        ``group_min() / group_max()``). Alternatively, ``to_overall``
        computes the ratio between each subgroup and the
        corresponding value from ``overall`` (if there are control
        features, then ``overall`` is multivalued for each metric),
        expressing the ratio as a number less than 1.
        The result is the minimum of these values.
        """
        result = None
        if method == 'between_groups':
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

    def _process_functions(self, metric_functions, sample_params) -> Dict[str, FunctionContainer]:
        """Get the underlying metrics into :class:`fairlearn.metrics.FunctionContainer` objects."""
        func_dict = dict()
        if isinstance(metric_functions, dict):
            s_p = dict()
            if sample_params is not None:
                if not isinstance(sample_params, dict):
                    raise ValueError(_SAMPLE_PARAMS_NOT_DICT)

                sp_keys = set(sample_params.keys())
                mf_keys = set(metric_functions.keys())
                if not sp_keys.issubset(mf_keys):
                    raise ValueError(_SAMPLE_PARAM_KEYS_NOT_IN_FUNC_DICT)
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

    def _process_features(self, base_name, features, sample_array) -> List[GroupFeature]:
        """Extract the features into :class:`fairlearn.metrics.GroupFeature` objects."""
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
                assert len(f_arr.shape) == 1  # Sanity check
                check_consistent_length(f_arr, sample_array)
                result.append(GroupFeature(base_name, f_arr, 0, None))
            else:
                for i in range(len(features)):
                    if isinstance(features[i], pd.Series):
                        check_consistent_length(features[i], sample_array)
                        result.append(GroupFeature(base_name, features[i], i, None))
                    else:
                        f_arr = np.squeeze(np.asarray(features[i]))
                        assert len(f_arr.shape) == 1  # Sanity check
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

    def _mask_from_tuple(self, index_tuple, feature_list) -> np.ndarray:
        """Generate a mask for the ``y_true``, ``y_pred`` and ``sample_params`` arrays.

        Given a tuple of feature values (which indexes the ``by_groups``
        DataFrame), generate a mask to select the corresponding samples
        from the input
        """
        # Following are internal sanity checks
        assert isinstance(index_tuple, tuple)
        assert len(index_tuple) == len(feature_list)

        result = feature_list[0].get_mask_for_class(index_tuple[0])
        for i in range(1, len(index_tuple)):
            result = np.logical_and(
                result,
                feature_list[i].get_mask_for_class(index_tuple[i]))
        return result
