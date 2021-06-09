# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Utility class for plotting metrics with error ranges."""

from logging import error
from typing import Any, Dict

import matplotlib

from ._metric_frame import MetricFrame

_ALLOWED_ERROR_KEY_TYPES = ["symmetric_error", "lower_error", "upper_error", "lower_bound", "upper_bound"]

_ERROR_MAPPING_DICT_ERROR = "error_mapping should be a dictionary"
_ERROR_MAPPING_ZERO_ITEMS_ERROR = "error_mapping should have non-zero amount of items"
_ERROR_MAPPING_METRIC_NOT_IN_MF_ERROR = "Metric name {0} should be a valid metric in provided MetricFrame"
_ERROR_MAPPING_KEY_TYPE_NOT_VALID_ERROR = "Received metric key_type of {0}, but should be one of: {1}"
_ERROR_MAPPING_KEY_TYPE_VALUE_NOT_IN_MF_ERROR = "Error metric value indexed by key {0} for Metric {1} should be a valid metric in provided MetricFrame"

_METRIC_NOT_STRING_ERROR = "Metric should be string of the metric in provided MetricFrame, but {0} was provided"
_NEG_ERROR_OR_FLIPPED_BOUNDS_ERROR = "upper_error, lower_error, and symmetric_error must be positive and upper_bound must be greater than lower_bound"
_INVALID_ERROR_MAPPING = "Invalid error mapping, each metric must have all the key-value pairs that make up an Error Format"

class ErrorPlotter:
    """Visualization tool for metrics with statistical error bounds

    This class takes in a :class:`fairlearn.metrics.MetricFrame` with precomputed metrics and metric errors
    and a `error_mapping` dict to interpret the columns (??) of the `MetricFrame`.
    
    Parameters
    ----------
    metric_frame : fairlearn.metrics.MetricFrame
        The collection fo disaggregated metric values, along with the metric errors.
    
    error_mapping : dict
        The mapping between metrics and their corresponding metric errors.
        
        Note:
            This class references columns from the `fairlearn.metrics.MetricFrame`
            by their column (??) name.

        The supported formats of errors are below:
        +--------------------+-------------------+----------+-------------------+
        | Error Format       |  Key_Type         | Example  | Resulting Bounds  |
        +====================+===================+==========+===================+
        | Symmetric Error (±)|  symmetric_error  | 0.01     | [X-0.01, X+0.01]  |
        +--------------------+-------------------+----------+-------------------+
        | Asymmetric Error   |  lower_error      | 0.02     | [X-0.02, X+0.03]  |
        |                    +-------------------+----------+                   |
        |                    |  upper_error      | 0.03     |                   |
        +--------------------+-------------------+----------+-------------------+
        | Bounds (symmetric  |  lower_bound      | 0.805    | [0.805, 0.815]    |
        | or asymmetric)     +-------------------+----------+                   |
        |                    |  upper_bound      | 0.815    |                   |
        +--------------------+-------------------+----------+-------------------+

        The format for `error_mapping` is:
            {
                "Metric Name A": {
                    "Key_Type 1": "Metric Error Name"
                    "Key_Type 2": "Metric Error Name"
                },
                "Metric Name B": {
                    "Key_Type 1": "Metric Error Name"
                    "Key_Type 2": "Metric Error Name"
                },
            }

        Example:
            {
                "Recall": {
                    "upper_bound": "Recall upper bound",
                    "lower_bound": "Recall lower bound"
                },
                "Accuracy": {
                    "symmetric_error": "Accuracy error"
                }
            }
    """

    def __init__(self,
                 metric_frame: MetricFrame,
                 error_mapping: Dict[str, Dict[str, Any]]):
        """Read a placeholder comment."""
        # Verify valid `error_mapping`
        assert isinstance(error_mapping, dict), _ERROR_MAPPING_DICT_ERROR
        assert len(error_mapping) > 0, _ERROR_MAPPING_ZERO_ITEMS_ERROR
        for metric_name in error_mapping.keys():
            assert metric_name in metric_frame.by_group, _ERROR_MAPPING_METRIC_NOT_IN_MF_ERROR.format(metric_name)
            for key_type in error_mapping[metric_name].keys():
                assert key_type in _ALLOWED_ERROR_KEY_TYPES, _ERROR_MAPPING_KEY_TYPE_NOT_VALID_ERROR.format(key_type, _ALLOWED_ERROR_KEY_TYPES)
                assert error_mapping[metric_name][key_type] in metric_frame.by_group, _ERROR_MAPPING_KEY_TYPE_VALUE_NOT_IN_MF_ERROR.format(key_type, metric_name)

        self.metric_frame = metric_frame
        self.error_mapping = error_mapping

    def plot_with_error(self, plot_type: str, metric: str, ax: matplotlib.axes._axes.Axes=None, plot_error_labels: bool=True,
                        text_precision_digits: int=4, text_fontsize: int=8, text_color: str="black",
                        text_ha: str="center", **kwargs):
        """
        Plots a given metric and its given error (as described by the `error_mapping`)

        Note:
            If multiple valid error_mappings are provided, this function will prioritize in the following order:
            1. upper_bound and lower_bound
            2. upper_error and lower_error
            3. symmetric_error
        
        Parameters
        ----------
        plot_type : str
            The type of plot to display. i.e. "bar", "line", etc

        metric : str
            The name of the metric to plot. Should match a column from the given `MetricFrame`

        ax : matplotlib.axes._axes.Axes, optional
            Custom Matplotlib axes to which this function can plot

        plot_error_labels : bool, optional
            Whether or not to plot numerical labels for the error bounds

        text_precision_digits : int, optional
            The number of digits of precision to show for error labels

        text_font_size : int, optional
            The font size to use for error labels

        text_color : str, optional
            The font color to use for error labels

        text_ha : str, optional
            The horizontal alignment modifier to use for error labels

        **kwargs
            Other optional keyword arguments to pass to matplotlib plotting method

        Returns
        -------
        matplotlib.axes._axes.Axes
        """
        assert isinstance(metric, str), _METRIC_NOT_STRING_ERROR.format(type(metric))
        assert metric in self.error_mapping.keys()

        df = self.metric_frame.by_group

        if "upper_bound" in self.error_mapping[metric].keys() and \
           "lower_bound" in self.error_mapping[metric].keys():
            lower_bound = self.error_mapping[metric]["lower_bound"]
            upper_bound = self.error_mapping[metric]["upper_bound"]

            df_lower_error = df[metric] - df[lower_bound]
            df_upper_error = df[upper_bound] - df[metric]
            df_error = [df_lower_error, df_upper_error]

            if plot_error_labels:
                df_lower_bound = df[lower_bound]
                df_upper_bound = df[upper_bound]
        elif "upper_error" in self.error_mapping[metric].keys() and \
             "lower_error" in self.error_mapping[metric].keys():
            lower_error = self.error_mapping[metric]["lower_error"]
            upper_error = self.error_mapping[metric]["upper_error"]

            df_lower_error = df[lower_error]
            df_upper_error = df[upper_error]
            df_error = [df_lower_error, df_upper_error]

            if plot_error_labels:
                df_lower_bound = df[metric] - df_lower_error
                df_upper_bound = df[metric] + df_upper_error
        elif "symmetric_error" in self.error_mapping[metric].keys():
            symmetric_error = self.error_mapping[metric]["symmetric_error"]
            df_error = df[symmetric_error]

            if plot_error_labels:
                df_lower_bound = df[metric] - df_error
                df_upper_bound = df[metric] + df_error
        else:
            raise AssertionError(_INVALID_ERROR_MAPPING)
        
        assert (df_error >= 0).all(), _NEG_ERROR_OR_FLIPPED_BOUNDS_ERROR

        ax = df.plot(kind=plot_type, y=metric, yerr=df_error, ax=ax, **kwargs)

        # TODO: Check assumption of plotting items in the vertical direction
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        if plot_error_labels:
            for i, item in enumerate(ax.patches):
                ax.text(i, df_lower_bound[i] - 0.05 * y_range, round(df_lower_bound[i],
                        text_precision_digits), fontsize=text_fontsize, color=text_color,
                        ha=text_ha)
                ax.text(i, df_upper_bound[i] + 0.01 * y_range, round(df_upper_bound[i],
                        text_precision_digits), fontsize=text_fontsize, color=text_color,
                        ha=text_ha)
        return ax

    def update_error_mapping():
        pass
