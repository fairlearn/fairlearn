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


class DisaggregatedResult:
    def __init__(
        self,
        overall: Union[pd.Series, pd.DataFrame],
        by_group: Union[pd.Series, pd.DataFrame],
    ):
        self._overall = overall
        self._by_group = by_group

    @property
    def overall(self) -> Union[pd.Series, pd.DataFrame]:
        return self._overall

    @property
    def by_group(self) -> Union[pd.Series, pd.DataFrame]:
        return self._by_group

    def apply_grouping(self, grouping_function: str,
                       control_feature_names: Optional[List[str]],
                       errors: str = "raise"
                       ) -> Union[pd.Series, pd.DataFrame]:
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

        return result


def extract_unique_classes(data: pd.DataFrame, feature_list: List[str]):
    result = dict()
    for feature in feature_list:
        result[feature] = np.unique(data[feature])
    return result


def calculate_disaggregated_metrics(
    data: pd.DataFrame,
    annotated_functions: Dict[str, AnnotatedMetricFunction],
    sensitive_feature_names: List[str],
    control_feature_names: Optional[List[str]],
) -> DisaggregatedResult:
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
