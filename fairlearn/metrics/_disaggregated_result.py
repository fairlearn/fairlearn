# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ._annotated_metric_function import AnnotatedMetricFunction

logger = logging.getLogger(__name__)

_VALID_ERROR_STRING = ["raise", "coerce"]
_VALID_GROUPING_FUNCTION = ["min", "max"]

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


def extract_unique_classes(
    data: pd.DataFrame, feature_list: List[str]
) -> Dict[str, np.ndarray]:
    """Compute unique values in a given set of columns."""
    result = dict()
    for feature in feature_list:
        result[feature] = np.unique(data[feature])
    return result


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
    # correctly handle zero provided metrics
    if len(values) == 0:
        result = pd.Series(dtype=float)
    else:
        result = pd.Series(values)
    return result


class DisaggregatedResult:
    """Pickier version of MetricFrame.

    This holds the internal result from a disaggregated metric
    computation, and provides `apply_grouping()` (to cover min
    and max), `difference()` and `ratio()` methods.

    The main difference to the results computed by MetricFrame
    is that no account is made of whether the user supplied
    a bare function or a dictionary. Hence the results are
    always Series or DataFrame.

    Parameters
    ----------
    overall: Series or DataFrame
        The metric function(s) computed on the entire dataset, split by
        control features if supplied
    by_group: Series or DataFrame
        The metric function(s) computed on each subgroup identified by
        the sensitive and control features
    """

    def __init__(self, overall: Union[pd.Series, pd.DataFrame], by_group: pd.DataFrame):
        """Construct an object."""
        self._overall = overall
        assert isinstance(by_group, pd.DataFrame)
        self._by_group = by_group

    @property
    def overall(self) -> Union[pd.Series, pd.DataFrame]:
        """Return overall metrics."""
        return self._overall

    @property
    def by_group(self) -> pd.DataFrame:
        """Return the metrics by group."""
        return self._by_group

    def apply_grouping(
        self,
        grouping_function: str,
        control_feature_names: Optional[List[str]] = None,
        errors: str = "raise",
    ) -> Union[pd.Series, pd.DataFrame]:
        """Compute mins or maxes.

        Parameters
        ----------
        grouping_function: string
            Must be 'min' or 'max'
        control_feature_names: Optional[List[str]]
            Names of the control features. Must appear in the index of the `overall`
            and `by_group` properties
        errors: string {'raise', 'coerce'}, default 'raise'
            How to deal with any errors. Either coerce to `np.nan` or wrap the
            exception and reraise

        Returns
        -------
        Series or DataFrame
            Contains the desired mins or maxes
        """
        if grouping_function not in _VALID_GROUPING_FUNCTION:
            raise ValueError(_INVALID_GROUPING_FUNCTION_ERROR_MESSAGE)

        if errors not in _VALID_ERROR_STRING:
            raise ValueError(_INVALID_ERRORS_VALUE_ERROR_MESSAGE)

        if not control_feature_names:
            if errors == "raise":
                try:
                    mf = self.by_group
                    if grouping_function == "min":
                        vals = [mf[m].min() for m in mf.columns]
                    else:
                        vals = [mf[m].max() for m in mf.columns]

                    result = pd.Series(
                        vals, index=self.by_group.columns, dtype="object"
                    )
                except ValueError as ve:
                    raise ValueError(_MF_CONTAINS_NON_SCALAR_ERROR_MESSAGE) from ve
            elif errors == "coerce":
                if not control_feature_names:
                    mf = self.by_group
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
                        result = self.by_group.groupby(
                            level=control_feature_names
                        ).min()
                    else:
                        result = self.by_group.groupby(
                            level=control_feature_names
                        ).max()
                except ValueError as ve:
                    raise ValueError(_MF_CONTAINS_NON_SCALAR_ERROR_MESSAGE) from ve
            elif errors == "coerce":
                # Fill all impossible columns with NaN before grouping metric frame
                mf = self.by_group.copy()
                mf = mf.applymap(lambda x: x if np.isscalar(x) else np.nan)
                if grouping_function == "min":
                    result = mf.groupby(level=control_feature_names).min()
                else:
                    result = mf.groupby(level=control_feature_names).max()

        assert isinstance(result, pd.Series) or isinstance(result, pd.DataFrame)

        return result

    def difference(
        self,
        control_feature_names: List[str],
        method: str = "between_groups",
        errors: str = "coerce",
    ) -> Union[pd.Series, pd.DataFrame]:
        """Return the maximum absolute difference between groups for each metric.

        This method calculates a scalar value for each underlying metric by
        finding the maximum absolute difference between the entries in each
        combination of sensitive features in the :attr:`.by_group` property.

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
        errors: {'raise', 'coerce'}, default 'coerce'
            if 'raise', then invalid parsing will raise an exception
            if 'coerce', then invalid parsing will be set as NaN

        Returns
        -------
        pandas.Series or pandas.DataFrame
        """
        if errors not in _VALID_ERROR_STRING:
            raise ValueError(_INVALID_ERRORS_VALUE_ERROR_MESSAGE)

        if method == "between_groups":
            subtrahend = self.apply_grouping(
                "min", control_feature_names, errors=errors
            )
        elif method == "to_overall":
            subtrahend = self.overall
        else:
            raise ValueError(
                "Unrecognised method '{0}' in difference() call".format(method)
            )

        mf = self.by_group.copy()
        # Can assume errors='coerce', else error would already have been raised in .group_min
        # Fill all non-scalar values with NaN
        mf = mf.applymap(lambda x: x if np.isscalar(x) else np.nan)

        if control_feature_names is None:
            result = (mf - subtrahend).abs().max()
        else:
            result = (mf - subtrahend).abs().groupby(level=control_feature_names).max()

        assert isinstance(result, pd.Series) or isinstance(result, pd.DataFrame)

        return result

    def ratio(
        self,
        control_feature_names: List[str],
        method: str = "between_groups",
        errors: str = "coerce",
    ) -> Union[pd.Series, pd.DataFrame]:
        """Return the minimum ratio between groups for each metric.

        This method calculates a scalar value for each underlying metric by
        finding the minimum ratio (that is, the ratio is forced to be
        less than unity) between the entries in each
        column of the :attr:`.by_group` property.

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
        errors: {'raise', 'coerce'}, default 'coerce'
            if 'raise', then invalid parsing will raise an exception
            if 'coerce', then invalid parsing will be set as NaN

        Returns
        -------
        typing.Any or pandas.Series or pandas.DataFrame
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
            result = self.apply_grouping(
                "min", control_feature_names, errors=errors
            ) / self.apply_grouping("max", control_feature_names, errors=errors)
        elif method == "to_overall":
            ratios = None

            if control_feature_names is not None:
                # It's easiest to give in to the DataFrame columns preference
                ratios = self.by_group.unstack(
                    level=control_feature_names
                ) / self.overall.unstack(level=control_feature_names)
            else:
                ratios = self.by_group / self.overall

            ratios = ratios.apply(lambda x: x.transform(ratio_sub_one))
            if not control_feature_names:
                result = ratios.min()
            else:
                result = ratios.min().unstack(0)
        else:
            raise ValueError("Unrecognised method '{0}' in ratio() call".format(method))

        assert isinstance(result, pd.Series) or isinstance(result, pd.DataFrame)

        return result

    @staticmethod
    def create(
        *,
        data: pd.DataFrame,
        annotated_functions: Dict[str, AnnotatedMetricFunction],
        sensitive_feature_names: List[str],
        control_feature_names: Optional[List[str]],
    ) -> "DisaggregatedResult":
        """Manufacture a DisaggregatedResult.

        This is essentially a more restricted version of the MetricFrame
        constructor.

        All of the `data` have to be supplied as a DataFrame.
        The metric functions have to be supplied as a dictionary of
        AnnotatedMetricFunction.
        The latter class contains the metric function itself, and
        mappings between the metric function arguments and the columns
        of the `data` DataFrame.
        The sensitive and (optional) control features are lists of
        column names in `data`.

        Parameters
        ----------
        data : DataFrame
            A DataFrame containing all of the columns required to compute the metrics
        annotated_functions: Dict[str, AnnotatedMetricFunction]
            A dictionary of metric functions, each of which is annotated with the
            mapping of columns in `data` to argument names in the function
        sensitive_feature_names: List[str]
            The list of columns in `data` which correspond to the sensitive feature(s)
        control_feature_names: Optional[List[str]]
            Optional list of columns in `data` which correspond to the control features,
            if any

        Returns
        -------
        DisaggregatedResult
            Freshly constructed instance of this class
        """
        # Calculate the 'overall' values
        if control_feature_names is None:
            overall = apply_to_dataframe(data, metric_functions=annotated_functions)
        else:
            temp = data.groupby(by=control_feature_names).apply(
                apply_to_dataframe, metric_functions=annotated_functions
            )
            # If there are multiple control features, might have missing combinations
            if len(control_feature_names) > 1:
                cf_classes = extract_unique_classes(data, control_feature_names)
                all_indices = pd.MultiIndex.from_product(
                    cf_classes.values(), names=cf_classes.keys()
                )

                overall = temp.reindex(index=all_indices)
            else:
                overall = temp

        # Calculate the 'by_group' values
        all_grouping_names = [x for x in sensitive_feature_names]
        if control_feature_names is not None:
            # Note that we prepend the control feature names
            all_grouping_names = control_feature_names + all_grouping_names

        temp = data.groupby(all_grouping_names).apply(
            apply_to_dataframe, metric_functions=annotated_functions
        )
        if len(all_grouping_names) > 1:
            # We might have missing combinations in the input, so expand to fill
            all_classes = extract_unique_classes(data, all_grouping_names)
            all_indices = pd.MultiIndex.from_product(
                all_classes.values(),
                names=all_classes.keys(),
            )

            by_group = temp.reindex(index=all_indices)
        else:
            by_group = temp

        return DisaggregatedResult(overall, by_group)
