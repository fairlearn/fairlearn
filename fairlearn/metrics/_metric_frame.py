# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import copy
import logging
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils import check_consistent_length

from fairlearn.metrics._input_manipulations import _convert_to_ndarray_and_squeeze

from ._annotated_metric_function import AnnotatedMetricFunction
from ._group_feature import GroupFeature

logger = logging.getLogger(__name__)

_VALID_ERROR_STRING = ["raise", "coerce"]
_VALID_GROUPING_FUNCTION = ["min", "max"]

_SF_DICT_CONVERSION_FAILURE = (
    "DataFrame.from_dict() failed on sensitive features. "
    "Please ensure each array is strictly 1-D. "
    "The __cause__ field of this exception may contain further information."
)
_FEATURE_LIST_NONSCALAR = "Feature lists must be of scalar types"
_FEATURE_DF_COLUMN_BAD_NAME = (
    "DataFrame column names must be strings. Name '{0}' is of type {1}"
)
_DUPLICATE_FEATURE_NAME = "Detected duplicate feature name: '{0}'"
_TOO_MANY_FEATURE_DIMS = "Feature array has too many dimensions"
_SAMPLE_PARAMS_NOT_DICT = "Sample parameters must be a dictionary"
_SAMPLE_PARAM_KEYS_NOT_IN_FUNC_DICT = (
    "Keys in 'sample_params' do not match those in 'metric'"
)
_INVALID_ERRORS_VALUE_ERROR_MESSAGE = (
    "Invalid error value specified. Valid values are {0}".format(_VALID_ERROR_STRING)
)
_INVALID_GROUPING_FUNCTION_ERROR_MESSAGE = (
    "Invalid grouping function specified. Valid values are {0}".format(
        _VALID_GROUPING_FUNCTION
    )
)
_MF_CONTAINS_NON_SCALAR_ERROR_MESSAGE = (
    "Metric frame contains non-scalar cells. Please remove non-scalar columns from your"
    " metric frame or use parameter errors='coerce'."
)


def apply_to_dataframe(
    data: pd.DataFrame, metric_functions: Dict[str, AnnotatedMetricFunction]
) -> pd.Series:
    """Apply metric functions to a DataFrame.

    The incoming DataFrame may have been sliced via `groupby()`.
    This function applies each annotated function in turn to the
    supplied DataFrame.
    """
    values = dict()
    for function_name, metric_function in metric_functions.items():
        values[function_name] = metric_function(data)
    result = pd.Series(data=values.values(), index=values.keys())
    return result


def _deprecate_metric_frame_init(new_metric_frame_init):
    """Issue deprecation warnings for the `MetricFrame` constructor.

    Decorator to issue warnings if called with positional arguments
    or with the keyword argument `metric` instead of `metrics`.

    Parameters
    ----------
    new_metric_frame_init : callable
        New MetricFrame constructor.
    """

    @wraps(new_metric_frame_init)
    def compatible_metric_frame_init(self, *args, metric=None, **kwargs):
        positional_names = ["metrics", "y_true", "y_pred"]
        version = "0.10.0"

        positional_dict = dict(zip(positional_names, args))

        # If more than 3 positional arguments are provided (apart from self), show
        # the error message applicable to the new constructor implementation (with `self`
        # being the only positional argument).
        if len(args) > 3:
            raise TypeError(
                f"{new_metric_frame_init.__name__}() takes 1 positional "
                f"argument but {1+len(args)} positional arguments "
                "were given"
            )

        # If 1-3 positional arguments are provided (apart fom self), issue warning.
        if len(args) > 0:
            args_msg = ", ".join([f"'{name}'" for name in positional_dict.keys()])
            warnings.warn(
                f"You have provided {args_msg} as positional arguments. "
                "Please pass them as keyword arguments. From version "
                f"{version} passing them as positional arguments "
                "will result in an error.",
                FutureWarning,
            )

        # If a keyword argument `metric` is provided, issue warning.
        metric_arg_dict = {}
        if metric is not None:
            metric_arg_dict = {"metrics": metric}
            warnings.warn(
                "The positional argument 'metric' has been replaced "
                "by a keyword argument 'metrics'. "
                f"From version {version} passing it as a positional argument "
                "or as a keyword argument 'metric' will result in an error",
                FutureWarning,
            )

        # Call the new constructor with positional arguments passed as keyword arguments
        # and with the `metric` keyword argument renamed to `metrics`.
        new_metric_frame_init(self, **metric_arg_dict, **positional_dict, **kwargs)

    return compatible_metric_frame_init


class MetricFrame:
    r"""Collection of disaggregated metric values.

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
    various aggregators over group-level metrics, such as the
    maximum, minimum, or the worst-case difference or ratio.

    This data structure also supports the concept of 'control features.' Like the sensitive
    features, control features identify subgroups within the data, but
    aggregations are not performed over the control features. Instead, the
    aggregations produce a result for each subgroup identified by the control
    feature(s). The name 'control features' refers to the statistical practice
    of 'controlling' for a variable.

    Read more in the :ref:`User Guide <assessment>`.

    Parameters
    ----------
    metrics : callable or dict
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

    metric : callable or dict
        The underlying metric functions which are to be calculated. This
        can either be a single metric function or a dictionary of functions.
        These functions must be callable as
        ``fn(y_true, y_pred, **sample_params)``.
        If there are any other arguments required (such as ``beta`` for
        :func:`sklearn.metrics.fbeta_score`) then
        :func:`functools.partial` must be used.

        .. deprecated:: 0.7.0
            `metric` will be removed in version 0.10.0, use `metrics` instead.

    Examples
    --------
    >>> from fairlearn.metrics import MetricFrame, selection_rate
    >>> from sklearn.metrics import accuracy_score
    >>> import pandas as pd
    >>> y_true = [1,1,1,1,1,0,0,1,1,0]
    >>> y_pred = [0,1,1,1,1,0,0,0,1,1]
    >>> sex = ['Female']*5 + ['Male']*5
    >>> metrics = {"selection_rate": selection_rate}
    >>> mf1 = MetricFrame(
    ...      metrics=metrics,
    ...      y_true=y_true,
    ...      y_pred=y_pred,
    ...      sensitive_features=sex)

    Access the disaggregated metrics via a pandas Series

    >>> mf1.by_group # doctest: +NORMALIZE_WHITESPACE
                        selection_rate
    sensitive_feature_0
    Female                         0.8
    Male                           0.4

    Access the largest difference, smallest ratio, and worst case performance

    >>> print(f"difference: {mf1.difference()[0]:.3}\t"
    ...      f"ratio: {mf1.ratio()[0]:.3}\t"
    ...      f"max across groups: {mf1.group_max()[0]:.3}")
    ...# doctest: +NORMALIZE_WHITESPACE
    difference: 0.4     ratio: 0.5      max across groups: 0.8

    You can also evaluate multiple metrics by providing a dictionary

    >>> metrics_dict = {"accuracy":accuracy_score, "selection_rate": selection_rate}
    >>> mf2 = MetricFrame(
    ...      metrics=metrics_dict,
    ...      y_true=y_true,
    ...      y_pred=y_pred,
    ...      sensitive_features=sex)

    Access the disaggregated metrics via a pandas DataFrame

    >>> mf2.by_group # doctest: +NORMALIZE_WHITESPACE
                        accuracy selection_rate
    sensitive_feature_0
    Female                   0.8            0.8
    Male                     0.6            0.4

    The largest difference, smallest ratio, and the maximum and minimum values
    across the groups are then all pandas Series, for example:

    >>> mf2.difference()
    accuracy          0.2
    selection_rate    0.4
    dtype: object

    You'll probably want to view them transposed

    >>> pd.DataFrame({'difference': mf2.difference(),
    ...               'ratio': mf2.ratio(),
    ...               'group_min': mf2.group_min(),
    ...               'group_max': mf2.group_max()}).T
               accuracy selection_rate
    difference      0.2            0.4
    ratio          0.75            0.5
    group_min       0.6            0.4
    group_max       0.8            0.8

    More information about plotting metrics can be found in the following section: :ref:`plot`
    """

    # The deprecation decorator does two things:
    # (1) turns first three positional arguments into keyword arguments
    # (2) renames the 'metric' keyword argument into 'metrics'
    @_deprecate_metric_frame_init
    def __init__(
        self,
        *,
        metrics: Union[Callable, Dict[str, Callable]],
        y_true,
        y_pred,
        sensitive_features,
        control_features=None,
        sample_params: Optional[
            Union[Dict[str, Any], Dict[str, Dict[str, Any]]]
        ] = None,
    ):
        """Read a placeholder comment."""
        check_consistent_length(y_true, y_pred)

        y_t = _convert_to_ndarray_and_squeeze(y_true)
        y_p = _convert_to_ndarray_and_squeeze(y_pred)

        all_data = pd.DataFrame.from_dict({"y_true": list(y_t), "y_pred": list(y_p)})

        annotated_funcs = self._process_functions(metrics, sample_params, all_data)

        # Now, prepare the sensitive features
        sf_list = self._process_features("sensitive_feature_", sensitive_features, y_t)
        self._sf_names = [x.name_ for x in sf_list]

        # Prepare the control features
        # Adjust _sf_indices if needed
        cf_list = None
        self._cf_names = None
        if control_features is not None:
            cf_list = self._process_features("control_feature_", control_features, y_t)
            self._cf_names = [x.name_ for x in cf_list]

        # Add sensitive and conditional features to all_data
        for sf in sf_list:
            all_data[sf.name_] = list(sf.raw_feature_)
        if cf_list is not None:
            for cf in cf_list:
                all_data[cf.name_] = list(cf.raw_feature_)

        # Check for duplicate feature names
        nameset = set()
        namelist = self._sf_names
        if self._cf_names:
            namelist = namelist + self._cf_names
        for name in namelist:
            if name in nameset:
                raise ValueError(_DUPLICATE_FEATURE_NAME.format(name))
            nameset.add(name)

        # Create the 'overall' results
        self._overall = self._build_overall_frame(
            all_data, annotated_funcs, cf_list, self._cf_names
        )

        grouping_features = copy.deepcopy(sf_list)
        if cf_list is not None:
            # Prepend the conditional features, so they are 'higher'
            grouping_features = copy.deepcopy(cf_list) + grouping_features

        # Create the 'by group' results
        self._by_group = self._build_by_group_frame(
            all_data, annotated_funcs, grouping_features
        )

    @property
    def overall(self) -> Union[Any, pd.Series, pd.DataFrame]:
        """Return the underlying metrics evaluated on the whole dataset.

           Read more in the :ref:`User Guide <control_features_metrics>`.

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

        Read more in the :ref:`User Guide <control_features_metrics>`.

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

    def __group(
        self, grouping_function: str, errors: str = "raise"
    ) -> Union[Any, pd.Series, pd.DataFrame]:
        """Return the minimum/maximum value of the metric over the sensitive features.

        This is a private method, please use .group_min() or .group_max() instead.

        Parameters
        ----------
        grouping_function: {'min', 'max'}
        errors: {'raise', 'coerce'}, default 'raise'
        if 'raise', then invalid parsing will raise an exception
        if 'coerce', then invalid parsing will be set as NaN

        Returns
        -------
        typing.Any pandas.Series or pandas.DataFrame
            The minimum value over sensitive features. The exact type
            follows the table in :attr:`.MetricFrame.overall`.
        """
        if grouping_function not in _VALID_GROUPING_FUNCTION:
            raise ValueError(_INVALID_GROUPING_FUNCTION_ERROR_MESSAGE)

        if errors not in _VALID_ERROR_STRING:
            raise ValueError(_INVALID_ERRORS_VALUE_ERROR_MESSAGE)

        if not self.control_levels:
            if errors == "raise":
                try:
                    mf = self._by_group
                    if grouping_function == "min":
                        vals = [mf[m].min() for m in mf.columns]
                    else:
                        vals = [mf[m].max() for m in mf.columns]

                    result = pd.Series(
                        vals, index=self._by_group.columns, dtype="object"
                    )
                except ValueError as ve:
                    raise ValueError(_MF_CONTAINS_NON_SCALAR_ERROR_MESSAGE) from ve
            elif errors == "coerce":
                if not self.control_levels:
                    mf = self._by_group
                    # Fill in the possible min/max values, else np.nan
                    if grouping_function == "min":
                        vals = [
                            mf[m].min() if np.isscalar(mf[m].values[0]) else np.nan
                            for m in mf.columns
                        ]
                    else:
                        vals = [
                            mf[m].max() if np.isscalar(mf[m].values[0]) else np.nan
                            for m in mf.columns
                        ]

                    result = pd.Series(vals, index=mf.columns, dtype="object")
        else:
            if errors == "raise":
                try:
                    if grouping_function == "min":
                        result = self._by_group.groupby(level=self.control_levels).min()
                    else:
                        result = self._by_group.groupby(level=self.control_levels).max()
                except ValueError as ve:
                    raise ValueError(_MF_CONTAINS_NON_SCALAR_ERROR_MESSAGE) from ve
            elif errors == "coerce":
                # Fill all impossible columns with NaN before grouping metric frame
                mf = self._by_group.copy()
                mf = mf.applymap(lambda x: x if np.isscalar(x) else np.nan)
                if grouping_function == "min":
                    result = mf.groupby(level=self.control_levels).min()
                else:
                    result = mf.groupby(level=self.control_levels).max()

        if self._user_supplied_callable:
            if self.control_levels:
                return result.iloc[:, 0]
            else:
                return result.iloc[0]
        else:
            return result

    def group_max(self, errors: str = "raise") -> Union[Any, pd.Series, pd.DataFrame]:
        """Return the maximum value of the metric over the sensitive features.

        This method computes the maximum value over all combinations of
        sensitive features for each underlying metric function in the :attr:`.by_group`
        property (it will only succeed if all the underlying metric
        functions return scalar values). The exact return type depends on
        whether control features are present, and whether the metric functions
        were specified as a single callable or a dictionary.

        Read more in the :ref:`User Guide <assessment>`.

        Parameters
        ----------
        errors: {'raise', 'coerce'}, default 'raise'
            if 'raise', then invalid parsing will raise an exception
            if 'coerce', then invalid parsing will be set as NaN

        Returns
        -------
        typing.Any or pandas.Series or pandas.DataFrame
            The maximum value over sensitive features. The exact type
            follows the table in :attr:`.MetricFrame.overall`.
        """
        return self.__group("max", errors)

    def group_min(self, errors: str = "raise") -> Union[Any, pd.Series, pd.DataFrame]:
        """Return the maximum value of the metric over the sensitive features.

        This method computes the minimum value over all combinations of
        sensitive features for each underlying metric function in the :attr:`.by_group`
        property (it will only succeed if all the underlying metric
        functions return scalar values). The exact return type depends on
        whether control features are present, and whether the metric functions
        were specified as a single callable or a dictionary.

        Read more in the :ref:`User Guide <assessment>`.

        Parameters
        ----------
        errors: {'raise', 'coerce'}, default 'raise'
            if 'raise', then invalid parsing will raise an exception
            if 'coerce', then invalid parsing will be set as NaN

        Returns
        -------
        typing.Any or pandas.Series or pandas.DataFrame
            The maximum value over sensitive features. The exact type
            follows the table in :attr:`.MetricFrame.overall`.
        """
        return self.__group("min", errors)

    def difference(
        self, method: str = "between_groups", errors: str = "coerce"
    ) -> Union[Any, pd.Series, pd.DataFrame]:
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

        Read more in the :ref:`User Guide <control_features_metrics>`.

        Parameters
        ----------
        method : str
            How to compute the aggregate. Default is :code:`between_groups`
        errors: {'raise', 'coerce'}, default 'coerce'
            if 'raise', then invalid parsing will raise an exception
            if 'coerce', then invalid parsing will be set as NaN

        Returns
        -------
        typing.Any or pandas.Series or pandas.DataFrame
            The exact type follows the table in :attr:`.MetricFrame.overall`.
        """
        if errors not in _VALID_ERROR_STRING:
            raise ValueError(_INVALID_ERRORS_VALUE_ERROR_MESSAGE)

        if method == "between_groups":
            subtrahend = self.group_min(errors=errors)
        elif method == "to_overall":
            subtrahend = self.overall
        else:
            raise ValueError(
                "Unrecognised method '{0}' in difference() call".format(method)
            )

        mf = self.by_group.copy()
        # Can assume errors='coerce', else error would already have been raised in .group_min
        # Fill all non-scalar values with NaN
        if isinstance(mf, pd.Series):
            mf = mf.map(lambda x: x if np.isscalar(x) else np.nan)
        else:
            mf = mf.applymap(lambda x: x if np.isscalar(x) else np.nan)

        if self.control_levels is None:
            result = (mf - subtrahend).abs().max()
        else:
            result = (mf - subtrahend).abs().groupby(level=self.control_levels).max()

        return result

    def ratio(
        self, method: str = "between_groups", errors: str = "coerce"
    ) -> Union[Any, pd.Series, pd.DataFrame]:
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

        Read more in the :ref:`User Guide <assessment>`.

        Parameters
        ----------
        method : str
            How to compute the aggregate. Default is :code:`between_groups`
        errors: {'raise', 'coerce'}, default 'coerce'
            if 'raise', then invalid parsing will raise an exception
            if 'coerce', then invalid parsing will be set as NaN

        Returns
        -------
        typing.Any or pandas.Series or pandas.DataFrame
            The exact type follows the table in :attr:`.MetricFrame.overall`.
        """

        def ratio_sub_one(x):
            if x > 1:
                return 1 / x
            else:
                return x

        if errors not in _VALID_ERROR_STRING:
            raise ValueError(_INVALID_ERRORS_VALUE_ERROR_MESSAGE)

        result = None
        if method == "between_groups":
            result = self.group_min(errors=errors) / self.group_max(errors=errors)
        elif method == "to_overall":
            if self._user_supplied_callable:
                tmp = self.by_group / self.overall
                if self.control_levels:
                    result = (
                        tmp.transform(ratio_sub_one)
                        .groupby(level=self.control_levels)
                        .min()
                    )
                else:
                    result = tmp.transform(ratio_sub_one).min()
            else:
                ratios = None

                if self.control_levels:
                    # It's easiest to give in to the DataFrame columns preference
                    ratios = self.by_group.unstack(
                        level=self.control_levels
                    ) / self.overall.unstack(level=self.control_levels)
                else:
                    ratios = self.by_group / self.overall

                ratios = ratios.apply(lambda x: x.transform(ratio_sub_one))
                if not self.control_levels:
                    result = ratios.min()
                else:
                    result = ratios.min().unstack(0)
        else:
            raise ValueError("Unrecognised method '{0}' in ratio() call".format(method))

        return result

    def _process_functions(
        self, metric, sample_params, all_data: pd.DataFrame
    ) -> Dict[str, AnnotatedMetricFunction]:
        """Get the underlying metrics into :class:`fairlearn.metrics.AnnotatedMetricFunction`."""
        self._user_supplied_callable = True
        func_dict = dict()

        # The supplied 'metric' may be a dictionary of functions
        if isinstance(metric, dict):
            self._user_supplied_callable = False
            s_p = dict()

            if sample_params is not None:
                # If we have sample_params, they had better be a dictionary
                if not isinstance(sample_params, dict):
                    raise ValueError(_SAMPLE_PARAMS_NOT_DICT)

                # The keys of the sample_params dictionary must be a
                # subset of our supplied metric functions
                sp_keys = set(sample_params.keys())
                mf_keys = set(metric.keys())
                if not sp_keys.issubset(mf_keys):
                    raise ValueError(_SAMPLE_PARAM_KEYS_NOT_IN_FUNC_DICT)
                s_p = sample_params

            for name, func in metric.items():
                curr_s_p = None
                if name in s_p:
                    curr_s_p = s_p[name]

                amf = self._process_one_function(func, name, curr_s_p, all_data)
                func_dict[amf.name] = amf
        else:
            # This is the case where the user has supplied a single metric function
            amf = self._process_one_function(metric, None, sample_params, all_data)
            func_dict[amf.name] = amf
        return func_dict

    def _process_one_function(
        self,
        func: Callable,
        name: str,
        sample_parameters: Dict[str, Any],
        all_data: pd.DataFrame,
    ) -> AnnotatedMetricFunction:
        # Deal with the sample parameters
        _sample_param_arrays = dict()
        if sample_parameters is not None:
            if not isinstance(sample_parameters, dict):
                raise ValueError(_SAMPLE_PARAMS_NOT_DICT)
            for k, v in sample_parameters.items():
                if v is not None:
                    # Coerce any sample_params to being ndarrays for easy masking
                    _sample_param_arrays[k] = np.asarray(v)

        # Build the kwargs
        kwarg_dict = dict()
        for param_name, param_values in _sample_param_arrays.items():
            col_name = f"{name}_{param_name}"
            all_data[col_name] = param_values
            kwarg_dict[param_name] = col_name

        # Construct the return object
        amf = AnnotatedMetricFunction(
            func=func,
            name=name,
            postional_argument_names=["y_true", "y_pred"],
            kw_argument_mapping=kwarg_dict,
        )

        return amf

    def _process_features(
        self, base_name, features, sample_array
    ) -> List[GroupFeature]:
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
            try:
                df = pd.DataFrame.from_dict(features)
            except ValueError as ve:
                raise ValueError(_SF_DICT_CONVERSION_FAILURE) from ve
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
            f_arr = np.squeeze(np.asarray(features, dtype=object))
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

    def _build_overall_frame(self, data, metric_funcs, cf_list, cf_names):
        """Build the 'overall' result during construction."""
        if cf_names is None:
            return apply_to_dataframe(data, metric_functions=metric_funcs)
        else:
            temp = data.groupby(by=cf_names).apply(
                apply_to_dataframe, metric_functions=metric_funcs
            )
            # If there are multiple control features, might have missing combinations
            if len(cf_names) > 1:
                all_indices = pd.MultiIndex.from_product(
                    [x.classes_ for x in cf_list], names=[x.name_ for x in cf_list]
                )

                return temp.reindex(index=all_indices)
            else:
                return temp

    def _build_by_group_frame(self, data, metric_funcs, grouping_features):
        """Build the 'by_group' result during construction."""
        temp = data.groupby([x.name_ for x in grouping_features]).apply(
            apply_to_dataframe, metric_functions=metric_funcs
        )
        if len(grouping_features) > 1:
            # We might have missing combinations in the input, so expand to fill
            all_indices = pd.MultiIndex.from_product(
                [x.classes_ for x in grouping_features],
                names=[x.name_ for x in grouping_features],
            )

            return temp.reindex(index=all_indices)
        else:
            return temp
