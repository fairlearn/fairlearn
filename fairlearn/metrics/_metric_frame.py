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
_FEATURE_LIST_NONSCALAR = "Feature lists must be of scalar types"
_FEATURE_DF_COLUMN_BAD_NAME = "DataFrame column names must be strings. Name '{0}' is of type {1}"
_DUPLICATE_FEATURE_NAME = "Detected duplicate feature name: '{0}'"
_TOO_MANY_FEATURE_DIMS = "Feature array has too many dimensions"
_SAMPLE_PARAM_KEYS_NOT_IN_FUNC_DICT = \
    "Keys in 'sample_params' do not match those in 'metric'"


class MetricFrame:
    """Collection of disaggregated metric values.

    This data structure stores and manipulates disaggregated values for any number of underlying
    metrics. At least one sensitive feature must be supplied, which is used
    to split the data into subgroups. The underlying metric(s) is(are) calculated
    across the entire dataset (made available by the :attr:`.overall` property) and
    for each identified subgroup (made available by the :attr:`.by_group` property).

    The only limitations placed on the metric functions are that:

    * The first two arguments they take must be ``y_true`` and ``y_pred`` arrays
    * Any other arguments must correspond to sample properties (such as sample weights),
      meaning that their first dimension is the same as that of y_true and y_pred. These
      arguments will be split up along with the ``y_true`` and ``y_pred`` arrays

    The interpretation of the ``y_true`` and ``y_pred`` arrays is up to the
    underlying metric - it is perfectly possible to pass in lists of class
    probability tuples. We also support non-scalar return types for the
    metric function (such as confusion matrices) at the current time. However,
    the aggregation functions will not be well defined in this case.

    Group fairness metrics are obtained by methods that implement
    various aggregators over group-level metrics, such such as the
    maximum, minimum, or the worst-case difference or ratio.

    This data structure also supports the concept of 'control features.' Like the sensitive
    features, control features identify subgroups within the data, but
    aggregations are not performed over the control features. Instead, the
    aggregations produce a result for each subgroup identified by the control
    feature(s). The name 'control features' refers to the statistical practice
    of 'controlling' for a variable.

    Parameters
    ----------
    metric : callable or dict
        The underlying metric functions which are to be calculated. This
        can either be a single metric function or a dictionary of functions.
        These functions must be callable as
        ``fn(y_true, y_pred, **sample_params)``.
        If there are any other arguments required (such as ``beta`` for
        :func:`sklearn.metrics.fbeta_score`) then
        :func:`functools.partial` must be used.

        **Note** that the values returned by various members of the class change
        based on whether this argument is a callable or a dictionary of
        callables. This distinction remains *even if* the dictionary only
        contains a single entry.

    y_true : List, pandas.Series, numpy.ndarray, pandas.DataFrame
        The ground-truth labels (for classification) or target values (for regression).

    y_pred : List, pandas.Series, numpy.ndarray, pandas.DataFrame
        The predictions.

    sensitive_features : List, pandas.Series, dict of 1d arrays, numpy.ndarray, pandas.DataFrame
        The sensitive features which should be used to create the subgroups.
        At least one sensitive feature must be provided.
        All names (whether on pandas objects or dictionary keys) must be strings.
        We also forbid DataFrames with column names of ``None``.
        For cases where no names are provided we generate names ``sensitive_feature_[n]``.

    control_features : List, pandas.Series, dict of 1d arrays, numpy.ndarray, pandas.DataFrame
        Control features are similar to sensitive features, in that they
        divide the input data into subgroups.
        Unlike the sensitive features, aggregations are not performed
        across the control features - for example, the ``overall`` property
        will have one value for each subgroup in the control feature(s),
        rather than a single value for the entire data set.
        Control features can be specified similarly to the sensitive features.
        However, their default names (if none can be identified in the
        input values) are of the format ``control_feature_[n]``.

        **Note** the types returned by members of the class vary based on whether
        control features are present.

    sample_params : dict
        Parameters for the metric function(s). If there is only one metric function,
        then this is a dictionary of strings and array-like objects, which are split
        alongside the ``y_true`` and ``y_pred`` arrays, and passed to the metric function.
        If there are multiple metric functions (passed as a dictionary), then this is
        a nested dictionary, with the first set of string keys identifying the
        metric function name, with the values being the string-to-array-like dictionaries.
    """

    def __init__(self,
                 metric: Union[Callable, Dict[str, Callable]],
                 y_true,
                 y_pred, *,
                 sensitive_features,
                 control_features: Optional = None,
                 sample_params: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None):
        """Read a placeholder comment."""
        check_consistent_length(y_true, y_pred)
        y_t = _convert_to_ndarray_and_squeeze(y_true)
        y_p = _convert_to_ndarray_and_squeeze(y_pred)

        func_dict = self._process_functions(metric, sample_params)

        # Now, prepare the sensitive features
        sf_list = self._process_features("sensitive_feature_", sensitive_features, y_t)
        self._sf_names = [x.name for x in sf_list]

        # Prepare the control features
        # Adjust _sf_indices if needed
        cf_list = None
        self._cf_names = None
        if control_features is not None:
            cf_list = self._process_features("control_feature_", control_features, y_t)
            self._cf_names = [x.name for x in cf_list]

        # Check for duplicate feature names
        nameset = set()
        namelist = self._sf_names
        if self._cf_names:
            namelist = namelist + self._cf_names
        for name in namelist:
            if name in nameset:
                raise ValueError(_DUPLICATE_FEATURE_NAME.format(name))
            nameset.add(name)

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
    def overall(self) -> Union[Any, pd.Series, pd.DataFrame]:
        """Return the underlying metrics evaluated on the whole dataset.

        Returns
        -------
        typing.Any or pandas.Series or pandas.DataFrame
            The exact type varies based on whether control featuers were
            provided and how the metric functions were specified.

            ======== ================  =================================
            Metrics  Control Features  Result Type
            ======== ================  =================================
            Callable None              Return type of callable
            -------- ----------------  ---------------------------------
            Callable Provided          Series, indexed by the subgroups
                                       of the conditional feature(s)
            -------- ----------------  ---------------------------------
            Dict     None              Series, indexed by the metric
                                       names
            -------- ----------------  ---------------------------------
            Dict     Provided          DataFrame. Columns are
                                       metric names, rows are subgroups
                                       of conditional feature(s)
            ======== ================  =================================

            The distinction applies even if the dictionary contains a
            single metric function. This is to allow for a consistent
            interface when calling programatically, while also reducing
            typing for those using Fairlearn interactively.
        """
        if self._user_supplied_callable:
            if self.control_levels:
                return self._overall.iloc[:, 0]
            else:
                return self._overall.iloc[0]
        else:
            return self._overall

    @property
    def by_group(self) -> Union[pd.Series, pd.DataFrame]:
        """Return the collection of metrics evaluated for each subgroup.

        The collection is defined by the combination of classes in the
        sensitive and control features. The exact type depends on
        the specification of the metric function.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            When a callable is supplied to the constructor, the result is
            a :class:`pandas.Series`, indexed by the combinations of subgroups
            in the sensitive and control features.

            When the metric functions were specified with a dictionary (even
            if the dictionary only has a single entry), then the result is
            a :class:`pandas.DataFrame` with columns named after the metric
            functions, and rows indexed by the combinations of subgroups
            in the sensitive and control features.

            If a particular combination of subgroups was not present in the dataset
            (likely to occur as more sensitive and control features
            are specified), then the corresponding entry will be NaN.
        """
        if self._user_supplied_callable:
            return self._by_group.iloc[:, 0]
        else:
            return self._by_group

    @property
    def control_levels(self) -> List[str]:
        """Return a list of feature names which are produced by control features.

        If control features are present, then the rows of the :attr:`.by_group`
        property have a :class:`pandas.MultiIndex` index. This property
        identifies which elements of that index are control features.

        Returns
        -------
        List[str] or None
            List of names, which can be used in calls to
            :meth:`pandas.DataFrame.groupby` etc.
        """
        return self._cf_names

    @property
    def sensitive_levels(self) -> List[str]:
        """Return a list of the feature names which are produced by sensitive features.

        In cases where the :attr:`.by_group` property has a :class:`pandas.MultiIndex`
        index, this identifies which elements of the index are sensitive features.

        Returns
        -------
        List[str]
            List of names, which can be used in calls to
            :meth:`pandas.DataFrame.groupby` etc.
        """
        return self._sf_names

    def group_max(self) -> Union[Any, pd.Series, pd.DataFrame]:
        """Return the maximum value of the metric over the sensitive features.

        This method computes the maximum value over all combinations of
        sensitive features for each underlying metric function in the :attr:`.by_group`
        property (it will only succeed if all the underlying metric
        functions return scalar values). The exact return type depends on
        whether control features are present, and whether the metric functions
        were specified as a single callable or a dictionary.

        Returns
        -------
        typing.Any or pandas.Series or pandas.DataFrame
            The maximum value over sensitive features. The exact type
            follows the table in :attr:`.MetricFrame.overall`.
        """
        if not self.control_levels:
            result = pd.Series(index=self._by_group.columns, dtype='object')
            for m in result.index:
                max_val = self._by_group[m].max()
                result[m] = max_val
        else:
            result = self._by_group.groupby(level=self.control_levels).max()

        if self._user_supplied_callable:
            if self.control_levels:
                return result.iloc[:, 0]
            else:
                return result.iloc[0]
        else:
            return result

    def group_min(self) -> Union[Any, pd.Series, pd.DataFrame]:
        """Return the minimum value of the metric over the sensitive features.

        This method computes the minimum value over all combinations of
        sensitive features for each underlying metric function in the :attr:`.by_group`
        property (it will only succeed if all the underlying metric
        functions return scalar values). The exact return type depends on
        whether control features are present, and whether the metric functions
        were specified as a single callable or a dictionary.

        Returns
        -------
        typing.Any pandas.Series or pandas.DataFrame
            The minimum value over sensitive features. The exact type
            follows the table in :attr:`.MetricFrame.overall`.
        """
        if not self.control_levels:
            result = pd.Series(index=self._by_group.columns, dtype='object')
            for m in result.index:
                min_val = self._by_group[m].min()
                result[m] = min_val
        else:
            result = self._by_group.groupby(level=self.control_levels).min()

        if self._user_supplied_callable:
            if self.control_levels:
                return result.iloc[:, 0]
            else:
                return result.iloc[0]
        else:
            return result

    def difference(self,
                   method: str = 'between_groups') -> Union[Any, pd.Series, pd.DataFrame]:
        """Return the maximum absolute difference between groups for each metric.

        This method calculates a scalar value for each underlying metric by
        finding the maximum absolute difference between the entries in each
        combination of sensitive features in the :attr:`.by_group` property.

        Similar to other methods, the result type varies with the
        specification of the metric functions, and whether control features
        are present or not.

        There are two allowed values for the ``method=`` parameter. The
        value ``between_groups`` computes the maximum difference between
        any two pairs of groups in the :attr:`.by_group` property (i.e.
        ``group_max() - group_min()``). Alternatively, ``to_overall``
        computes the difference between each subgroup and the
        corresponding value from :attr:`.overall` (if there are control
        features, then :attr:`.overall` is multivalued for each metric).
        The result is the absolute maximum of these values.

        Parameters
        ----------
        method : str
            How to compute the aggregate. Default is :code:`between_groups`

        Returns
        -------
        typing.Any or pandas.Series or pandas.DataFrame
            The exact type follows the table in :attr:`.MetricFrame.overall`.
        """
        subtrahend = np.nan
        if method == 'between_groups':
            subtrahend = self.group_min()
        elif method == 'to_overall':
            subtrahend = self.overall
        else:
            raise ValueError("Unrecognised method '{0}' in difference() call".format(method))

        return (self.by_group - subtrahend).abs().max(level=self.control_levels)

    def ratio(self,
              method: str = 'between_groups') -> Union[Any, pd.Series, pd.DataFrame]:
        """Return the minimum ratio between groups for each metric.

        This method calculates a scalar value for each underlying metric by
        finding the minimum ratio (that is, the ratio is forced to be
        less than unity) between the entries in each
        column of the :attr:`.by_group` property.

        Similar to other methods, the result type varies with the
        specification of the metric functions, and whether control features
        are present or not.

        There are two allowed values for the ``method=`` parameter. The
        value ``between_groups`` computes the minimum ratio between
        any two pairs of groups in the :attr:`.by_group` property (i.e.
        ``group_min() / group_max()``). Alternatively, ``to_overall``
        computes the ratio between each subgroup and the
        corresponding value from :attr:`.overall` (if there are control
        features, then :attr:`.overall` is multivalued for each metric),
        expressing the ratio as a number less than 1.
        The result is the minimum of these values.

        Parameters
        ----------
        method : str
            How to compute the aggregate. Default is :code:`between_groups`

        Returns
        -------
        typing.Any or pandas.Series or pandas.DataFrame
            The exact type follows the table in :attr:`.MetricFrame.overall`.
        """
        result = None
        if method == 'between_groups':
            result = self.group_min() / self.group_max()
        elif method == 'to_overall':
            if self._user_supplied_callable:
                tmp = self.by_group / self.overall
                result = tmp.transform(lambda x: min(x, 1/x)).min(level=self.control_levels)
            else:
                ratios = None

                if self.control_levels:
                    # It's easiest to give in to the DataFrame columns preference
                    ratios = self.by_group.unstack(level=self.control_levels) /  \
                        self.overall.unstack(level=self.control_levels)
                else:
                    ratios = self.by_group / self.overall

                def ratio_sub_one(x):
                    if x > 1:
                        return 1/x
                    else:
                        return x

                ratios = ratios.apply(lambda x: x.transform(ratio_sub_one))
                if not self.control_levels:
                    result = ratios.min()
                else:
                    result = ratios.min().unstack(0)
        else:
            raise ValueError("Unrecognised method '{0}' in ratio() call".format(method))

        return result

    def _process_functions(self, metric, sample_params) -> Dict[str, FunctionContainer]:
        """Get the underlying metrics into :class:`fairlearn.metrics.FunctionContainer` objects."""
        self._user_supplied_callable = True
        func_dict = dict()
        if isinstance(metric, dict):
            self._user_supplied_callable = False
            s_p = dict()
            if sample_params is not None:
                if not isinstance(sample_params, dict):
                    raise ValueError(_SAMPLE_PARAMS_NOT_DICT)

                sp_keys = set(sample_params.keys())
                mf_keys = set(metric.keys())
                if not sp_keys.issubset(mf_keys):
                    raise ValueError(_SAMPLE_PARAM_KEYS_NOT_IN_FUNC_DICT)
                s_p = sample_params

            for name, func in metric.items():
                curr_s_p = None
                if name in s_p:
                    curr_s_p = s_p[name]
                fc = FunctionContainer(func, name, curr_s_p)
                func_dict[fc.name_] = fc
        else:
            fc = FunctionContainer(metric, None, sample_params)
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
                col_name = features.columns[i]
                if not isinstance(col_name, str):
                    msg = _FEATURE_DF_COLUMN_BAD_NAME.format(col_name, type(col_name))
                    raise ValueError(msg)
                column = features.iloc[:, i]
                check_consistent_length(column, sample_array)
                result.append(GroupFeature(base_name, column, i, None))
        elif isinstance(features, list):
            if np.isscalar(features[0]):
                f_arr = np.atleast_1d(np.squeeze(np.asarray(features)))
                assert len(f_arr.shape) == 1  # Sanity check
                check_consistent_length(f_arr, sample_array)
                result.append(GroupFeature(base_name, f_arr, 0, None))
            else:
                raise ValueError(_FEATURE_LIST_NONSCALAR)
        elif isinstance(features, dict):
            df = pd.DataFrame.from_dict(features)
            for i in range(len(df.columns)):
                col_name = df.columns[i]
                if not isinstance(col_name, str):
                    msg = _FEATURE_DF_COLUMN_BAD_NAME.format(col_name, type(col_name))
                    raise ValueError(msg)
                column = df.iloc[:, i]
                check_consistent_length(column, sample_array)
                result.append(GroupFeature(base_name, column, i, None))
        else:
            # Need to specify dtype to avoid inadvertent type conversions
            f_arr = np.squeeze(np.asarray(features, dtype=np.object))
            if len(f_arr.shape) == 1:
                check_consistent_length(f_arr, sample_array)
                result.append(GroupFeature(base_name, f_arr, 0, None))
            elif len(f_arr.shape) == 2:
                # Work similarly to pd.DataFrame(data=ndarray)
                for i in range(f_arr.shape[1]):
                    col = f_arr[:, i]
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
