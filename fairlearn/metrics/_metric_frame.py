# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils import check_consistent_length

from fairlearn.utils._input_manipulations import _convert_to_ndarray_and_squeeze

from ._annotated_metric_function import AnnotatedMetricFunction
from ._bootstrap import calculate_pandas_quantiles, generate_bootstrap_samples
from ._disaggregated_result import (
    _INVALID_ERRORS_VALUE_ERROR_MESSAGE,
    _VALID_ERROR_STRING,
    DisaggregatedResult,
)
from ._group_feature import GroupFeature

logger = logging.getLogger(__name__)

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

_COMPARE_METHODS = ["between_groups", "to_overall"]
_INVALID_COMPARE_METHOD = "Unrecognised comparison method: {0}"

_BOOTSTRAP_NEED_N_AND_CI = "Must specify both n_boot and ci_quantiles"
_BOOTSTRAP_N_BOOT_INT_GT_ZERO = "Must have n_boot be a positive integer"
_BOOTSTRAP_CI_INVALID = "Must have all ci_quantiles be floats in (0, 1)"


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
        See the :ref:`section on intersecting groups <assessment_intersecting_groups>`
        in the User Guide to learn how to use control levels.

        **Note** the types returned by members of the class vary based on whether
        control features are present.

    sample_params : dict
        Parameters for the metric function(s). If there is only one metric function,
        then this is a dictionary of strings and array-like objects, which are split
        alongside the ``y_true`` and ``y_pred`` arrays, and passed to the metric function.
        If there are multiple metric functions (passed as a dictionary), then this is
        a nested dictionary, with the first set of string keys identifying the
        metric function name, with the values being the string-to-array-like dictionaries.

    n_boot : Optional[int]
        If set to a postive integer, generate this number of bootstrap samples of the
        supplied data, and use to estimate confidence intervals for all of the metrics.
        Must be set with `ci_quantiles`.

    ci_quantiles : Optional[List[float]]
        A list of confidence interval quantiles to extract from the bootstrap samples.
        For example, the list `[0.159, 0.5, 0.841]` would extract the median and
        standard deviations.

    random_state : Optional[Union[int, np.random.RandomState]]
        Used to control the generation of the bootstrap samples

    Examples
    --------
    We will now go through some simple examples (see the :ref:`User Guide <assessment>` for
    a more in-depth discussion):

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

    >>> print(f"difference: {mf1.difference()[0]:.3}   "
    ...      f"ratio: {mf1.ratio()[0]:.3}   "
    ...      f"max across groups: {mf1.group_max()[0]:.3}")
    difference: 0.4   ratio: 0.5   max across groups: 0.8

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
    dtype: float64

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

    More information about plotting metrics can be found in the
    :ref:`plotting section <plot_metricframe>` of the User Guide.
    """

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
        n_boot: Optional[int] = None,
        ci_quantiles: Optional[List[float]] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
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

        self._result_cache = dict()

        # Create the basic results
        result = DisaggregatedResult.create(
            data=all_data,
            annotated_functions=annotated_funcs,
            sensitive_feature_names=self._sf_names,
            control_feature_names=self._cf_names,
        )
        # Build into cache
        self._populate_results(result)

        # Handle bootstrapping
        self._ci_quantiles = None
        if n_boot is not None and ci_quantiles is not None and len(ci_quantiles) > 0:
            if not isinstance(n_boot, int) or n_boot < 1:
                raise ValueError(_BOOTSTRAP_N_BOOT_INT_GT_ZERO)
            for _ci in ci_quantiles:
                if not isinstance(_ci, float) or _ci <= 0 or _ci >= 1:
                    raise ValueError(_BOOTSTRAP_CI_INVALID)
            self._ci_quantiles = ci_quantiles

            _bootstrap_samples = generate_bootstrap_samples(
                n_samples=n_boot,
                random_state=random_state,
                data=all_data,
                annotated_functions=annotated_funcs,
                sensitive_feature_names=self._sf_names,
                control_feature_names=self._cf_names,
            )

            self._populate_results_ci(_bootstrap_samples, ci_quantiles)
        elif (n_boot is not None) ^ (
            (ci_quantiles is not None) and (len(ci_quantiles)) > 0
        ):
            raise ValueError(_BOOTSTRAP_NEED_N_AND_CI)

    def _extract_result(self, underlying_result, no_control_levels: bool):
        """
        Change result types for those who dislike consistency.

        The `no_control_levels` parameter determines whether the presence
        of control levels will affect the result. This is the case for
        overall, but not the other cases.
        """
        if self._user_supplied_callable:
            if self.control_levels or no_control_levels:
                return underlying_result.iloc[:, 0]
            else:
                return underlying_result.iloc[0]
        else:
            return underlying_result

    def _none_to_nan(
        self, target: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Convert Nones to NaNs."""
        # Ideally, we wouldn't care about Series vs DataFrame
        # However, DataFrame.map() didn't appear until Pandas 2.1
        # Before, it was DataFrame.applymap() which then got deprecated
        if isinstance(target, pd.Series):
            result = target.map(lambda x: x if x is not None else np.nan)
        else:
            result = target.apply(
                lambda x: x.apply(lambda y: y if np.isscalar(y) else np.nan)
            )
        return result

    def _populate_results(self, raw_result: DisaggregatedResult):
        """
        Populate the :code:`_result_cache`.

        We cache all possible results from :class:`~MetricFrame` internally
        (i.e. all possible calls to :meth:`~MetricFrame.difference()` etc.)
        and this method is responsible for performing those computations.
        The :code:`_result_cache` is a nested dictionary, with keys given by
        the method names and arguments being cached.

        Note that if exceptions are thrown, we cache those, and they are thrown
        if the user calls the corresponding method (and arguments).
        """
        # Start with overall
        self._result_cache["overall"] = self._extract_result(
            raw_result.overall, no_control_levels=False
        )

        # Now do by_group
        self._result_cache["by_group"] = self._extract_result(
            raw_result.by_group, no_control_levels=True
        )

        # Next up, group_min and group_max
        group_functions = {"group_min": "min", "group_max": "max"}
        for k, v in group_functions.items():
            self._result_cache[k] = dict()
            for err_string in _VALID_ERROR_STRING:
                try:
                    self._result_cache[k][err_string] = self._group(
                        raw_result, v, err_string
                    )
                except Exception as e:  # noqa: B902
                    # Store any exception for later
                    self._result_cache[k][err_string] = e

        # Differences and ratios
        for c_t in ["difference", "ratio"]:
            self._result_cache[c_t] = dict()
            for c_m in _COMPARE_METHODS:
                self._result_cache[c_t][c_m] = dict()
                for err_string in _VALID_ERROR_STRING:
                    try:
                        if c_t == "difference":
                            tmp = raw_result.difference(
                                self.control_levels, method=c_m, errors=err_string
                            )
                        else:
                            tmp = raw_result.ratio(
                                self.control_levels, method=c_m, errors=err_string
                            )

                        result = self._none_to_nan(tmp)

                        self._result_cache[c_t][c_m][err_string] = self._extract_result(
                            result, no_control_levels=False
                        )
                    except Exception as e:  # noqa: B902
                        # Store any exception for later
                        self._result_cache[c_t][c_m][err_string] = e

    def _populate_results_ci(
        self, bootstrap_samples: List[DisaggregatedResult], ci_quantiles: List[float]
    ):
        """Similar to _populate_results, but computes confidence intervals from bootstrap.

        Most of the work is done in :meth:`calculate_pandas_quantiles`.
        """
        result_overall = calculate_pandas_quantiles(
            ci_quantiles, [x.overall for x in bootstrap_samples]
        )
        self._result_cache["overall_ci"] = [
            self._extract_result(x, no_control_levels=False) for x in result_overall
        ]

        result_group = calculate_pandas_quantiles(
            ci_quantiles, [x.by_group for x in bootstrap_samples]
        )
        self._result_cache["by_group_ci"] = [
            self._extract_result(x, no_control_levels=True) for x in result_group
        ]

        group_functions = {"group_min_ci": "min", "group_max_ci": "max"}
        for k, v in group_functions.items():
            self._result_cache[k] = self._result_cache[k] = self._group_ci(
                bootstrap_samples=bootstrap_samples,
                ci_quantiles=ci_quantiles,
                grouping_function=v,
            )

        # Differences and ratios
        for c_t in ["difference_ci", "ratio_ci"]:
            self._result_cache[c_t] = dict()
            for c_m in _COMPARE_METHODS:
                if c_t == "difference_ci":
                    raw_samples = [
                        r.difference(
                            self.control_levels,
                            method=c_m,
                            errors="raise",
                        )
                        for r in bootstrap_samples
                    ]
                else:
                    raw_samples = [
                        r.ratio(
                            self.control_levels,
                            method=c_m,
                            errors="raise",
                        )
                        for r in bootstrap_samples
                    ]

                samples = [self._none_to_nan(x) for x in raw_samples]

                raw_result = calculate_pandas_quantiles(
                    quantiles=ci_quantiles, bootstrap_samples=samples
                )

                result = [
                    self._extract_result(x, no_control_levels=False) for x in raw_result
                ]

                self._result_cache[c_t][c_m] = result

    @property
    def overall(
        self,
    ) -> Union[Any, pd.Series, pd.DataFrame,]:
        """Return the underlying metrics evaluated on the whole dataset.

        Read more in the :ref:`User Guide <assessment_quantify_harms>`.

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
        return self._result_cache["overall"]

    @property
    def overall_ci(
        self,
    ) -> List[Union[Any, pd.Series, pd.DataFrame,]]:
        """Return the underlying bootstrapped metrics evaluated on the whole dataset.

        When bootstrapping has been activated (by `n_boot` and `ci_quantiles` in the
        constructor), this property will be available.
        The contents will be a list of the same underlying type as that returned by
        :attr:`MetricFrame.overall` property.
        The elements of the list are indexed by the `ci_quantiles` array supplied
        to the constructor.
        """
        return self._result_cache["overall_ci"]

    @property
    def by_group(
        self,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Return the collection of metrics evaluated for each subgroup.

        The collection is defined by the combination of classes in the
        sensitive and control features. The exact type depends on
        the specification of the metric function.

        Read more in the :ref:`User Guide <assessment_quantify_harms>`.

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
        return self._result_cache["by_group"]

    @property
    def by_group_ci(self) -> Union[List[pd.Series], List[pd.DataFrame]]:
        """Return the confidence intervals for the metrics, evaluated on each subgroup.

        When bootstrapping has been activated (by `n_boot` and `ci_quantiles` in the
        constructor), this property will be available.
        The contents will be a list, with each element having the same type as that
        returned by the :attr:`MetricFrame.by_group` property.
        The elements of the list are indexed by the `ci_quantiles` array supplied
        to the constructor.
        """
        return self._result_cache["by_group_ci"]

    @property
    def control_levels(self) -> Optional[List[str]]:
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

        Read more in the :ref:`User Guide <assessment_quantify_harms>`.

        Returns
        -------
        List[str]
            List of names, which can be used in calls to
            :meth:`pandas.DataFrame.groupby` etc.
        """
        return self._sf_names

    @property
    def ci_quantiles(self) -> Optional[List[float]]:
        """Return the quantiles specified for bootstrapping."""
        return self._ci_quantiles

    def _group(
        self,
        disagg_result: DisaggregatedResult,
        grouping_function: str,
        errors: str = "raise",
    ) -> Union[Any, pd.Series, pd.DataFrame]:
        """Return the minimum/maximum value of the metric over the sensitive features.

        This is a private method, please use .group_min() or .group_max() instead.

        Parameters
        ----------
        disagg_result: The DisaggregatedResult containing all the metrics
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
        result = disagg_result.apply_grouping(
            grouping_function, self.control_levels, errors=errors
        )

        return self._extract_result(result, no_control_levels=False)

    def _group_ci(
        self,
        bootstrap_samples: List[DisaggregatedResult],
        ci_quantiles: List[float],
        grouping_function: str,
    ) -> Union[List[Any], List[pd.Series], List[pd.DataFrame]]:
        # There is no 'errors' argument because everything must have been a scalar for
        # np.quantiles
        samples = [
            r.apply_grouping(grouping_function, self.control_levels, errors="raise")
            for r in bootstrap_samples
        ]

        raw_result = calculate_pandas_quantiles(
            quantiles=ci_quantiles, bootstrap_samples=samples
        )

        result = [self._extract_result(x, no_control_levels=False) for x in raw_result]
        return result

    def group_max(self, errors: str = "raise") -> Union[Any, pd.Series, pd.DataFrame]:
        """Return the maximum value of the metric over the sensitive features.

        This method computes the maximum value over all combinations of
        sensitive features for each underlying metric function in the :attr:`.by_group`
        property (it will only succeed if all the underlying metric
        functions return scalar values). The exact return type depends on
        whether control features are present, and whether the metric functions
        were specified as a single callable or a dictionary.

        Read more in the :ref:`User Guide <assessment_compare_harms>`.

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
        if errors not in _VALID_ERROR_STRING:
            raise ValueError(_INVALID_ERRORS_VALUE_ERROR_MESSAGE)

        value = self._result_cache["group_max"][errors]
        if isinstance(value, Exception):
            raise value
        else:
            return value

    def group_max_ci(self) -> Union[List[Any], List[pd.Series], List[pd.DataFrame]]:
        """Return the bootstrapped confidence intervals for :attr:`MetricFrame.group_max`.

        When bootstrapping has been activated (by `n_boot` and `ci_quantiles` in the
        constructor), this property will be available.
        The contents will be a list, with each element having the same type as that
        returned by the :meth:`MetricFrame.group_max` function.
        The elements of the list are indexed by the `ci_quantiles` array supplied
        to the constructor.

        Unlike :meth:`MetricFrame.group_max` there is no :code:`errors` parameter, because
        a bootstrapped :class:`MetricFrame` requires all the metrics to return scalars.
        """
        value = self._result_cache["group_max_ci"]
        return value

    def group_min(self, errors: str = "raise") -> Union[Any, pd.Series, pd.DataFrame]:
        """Return the maximum value of the metric over the sensitive features.

        This method computes the minimum value over all combinations of
        sensitive features for each underlying metric function in the :attr:`.by_group`
        property (it will only succeed if all the underlying metric
        functions return scalar values). The exact return type depends on
        whether control features are present, and whether the metric functions
        were specified as a single callable or a dictionary.

        Read more in the :ref:`User Guide <assessment_compare_harms>`.

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
        if errors not in _VALID_ERROR_STRING:
            raise ValueError(_INVALID_ERRORS_VALUE_ERROR_MESSAGE)

        value = self._result_cache["group_min"][errors]
        if isinstance(value, Exception):
            raise value
        else:
            return value

    def group_min_ci(self) -> Union[List[Any], List[pd.Series], List[pd.DataFrame]]:
        """Return the bootstrapped confidence intervals for :attr:`MetricFrame.group_min`.

        When bootstrapping has been activated (by `n_boot` and `ci_quantiles` in the
        constructor), this property will be available.
        The contents will be a list, with each element having the same type as that
        returned by the :meth:`MetricFrame.group_min` function.
        The elements of the list are indexed by the `ci_quantiles` array supplied
        to the constructor.

        Unlike :meth:`MetricFrame.group_min` there is no :code:`errors` parameter, because
        a bootstrapped :class:`MetricFrame` requires all the metrics to return scalars.
        """
        value = self._result_cache["group_min_ci"]
        return value

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

        Read more in the :ref:`User Guide <assessment_compare_harms>`.

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

        if method not in _COMPARE_METHODS:
            raise ValueError(_INVALID_COMPARE_METHOD.format(method))

        value = self._result_cache["difference"][method][errors]
        if isinstance(value, Exception):
            raise value
        else:
            return value

    def difference_ci(
        self, method: str = "between_groups"
    ) -> Union[List[Any], List[pd.Series], List[pd.DataFrame]]:
        """Return the bootstrapped confidence intervals for :meth:`MetricFrame.difference`.

        When bootstrapping has been activated (by `n_boot` and `ci_quantiles` in the
        constructor), this property will be available.
        The contents will be a list, with each element having the same type as that
        returned by the :func:`MetricFrame.difference` function.
        The elements of the list are indexed by the `ci_quantiles` array supplied
        to the constructor.

        Unlike :meth:`MetricFrame.difference` there is no :code:`errors` parameter, because
        a bootstrapped :class:`MetricFrame` requires all the metrics to return scalars.
        """
        if method not in _COMPARE_METHODS:
            raise ValueError(_INVALID_COMPARE_METHOD.format(method))

        value = self._result_cache["difference_ci"][method]
        return value

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

        Read more in the :ref:`User Guide <assessment_compare_harms>`.

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

        if method not in _COMPARE_METHODS:
            raise ValueError(_INVALID_COMPARE_METHOD.format(method))

        value = self._result_cache["ratio"][method][errors]
        if isinstance(value, Exception):
            raise value
        else:
            return value

    def ratio_ci(
        self, method: str = "between_groups"
    ) -> Union[List[Any], List[pd.Series], List[pd.DataFrame]]:
        """Return the bootstrapped confidence intervals for :meth:`MetricFrame.ratio`.

        When bootstrapping has been activated (by `n_boot` and `ci_quantiles` in the
        constructor), this property will be available.
        The contents will be a list, with each element having the same type as that
        returned by the :func:`MetricFrame.ratio` function.
        The elements of the list are indexed by the `ci_quantiles` array supplied
        to the constructor.

        Unlike :meth:`MetricFrame.ratio` there is no :code:`errors` parameter, because
        a bootstrapped :class:`MetricFrame` requires all the metrics to return scalars.
        """
        if method not in _COMPARE_METHODS:
            raise ValueError(_INVALID_COMPARE_METHOD.format(method))

        value = self._result_cache["ratio_ci"][method]
        return value

    def _process_functions(
        self,
        metric: Union[Callable, Dict[str, Callable]],
        sample_params,
        all_data: pd.DataFrame,
    ) -> Dict[str, AnnotatedMetricFunction]:
        """Get the metrics into :class:`fairlearn.metrics.AnnotatedMetricFunction`."""
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
        name: Optional[str],
        sample_parameters: Optional[Dict[str, Any]],
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
            positional_argument_names=["y_true", "y_pred"],
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
