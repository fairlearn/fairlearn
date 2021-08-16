# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""Utility class for plotting metrics with error ranges."""

from typing import List, Union
import pandas as pd
import numpy as np

from ._metric_frame import MetricFrame
from matplotlib.lines import Line2D
from matplotlib.axes._axes import Axes

_GIVEN_BOTH_ERRORS_AND_CONF_INT_ERROR = \
    "Ambiguous Error Metric. Please only provide one of: errors or conf_intervals."
_METRIC_AND_ERRORS_NOT_SAME_LENGTH_ERROR = \
    "The list of metrics and list of errors are not the same length."
_METRIC_LENGTH_ZERO_ERROR = \
    "No metrics were provided to plot. A nonzero number of metrics is required."
_METRIC_AND_CONF_INTERVALS_NOT_SAME_LENGTH_ERROR = \
    "The list of metrics and list of conf_intervals are not the same length."

_METRICS_NOT_LIST_OR_STR_ERROR = \
    """Metric should be a string of a single metric or a list of the metrics
     in provided MetricFrame, but {0} was provided"""
_ERRORS_MUST_BE_SCALAR_OR_ARRAY = "Calculated errors must be a scalar or a array of length 2"
_CONF_INTERVALS_MUST_BE_ARRAY = "Calculated conf_intervals must be array of length 2"
_ERRORS_NEGATIVE_VALUE_ERROR = "Calculated relative errors must be positive"
_CONF_INTERVALS_FLIPPED_BOUNDS_ERROR = \
    "Calculated conf_intervals' upper bound cannot be less than lower bound"


def _is_arraylike(error):
    return (isinstance(error, np.ndarray) or isinstance(error, list))


def _check_if_metrics_and_error_metrics_same_length(metrics, errors, conf_intervals):
    if errors is not None:
        if len(errors) != len(metrics):
            raise ValueError(_METRIC_AND_ERRORS_NOT_SAME_LENGTH_ERROR)
    elif conf_intervals is not None:
        if len(conf_intervals) != len(metrics):
            raise ValueError(_METRIC_AND_CONF_INTERVALS_NOT_SAME_LENGTH_ERROR)


def _check_if_metrics_length_zero(metrics):
    if len(metrics) == 0:
        raise ValueError(_METRIC_LENGTH_ZERO_ERROR)


def _check_for_ambiguous_error_metric(errors, conf_intervals):
    # ambiguous error metric if both are provided
    if errors is not None and conf_intervals is not None:
        raise ValueError(_GIVEN_BOTH_ERRORS_AND_CONF_INT_ERROR)


def _check_for_valid_metrics_format(metrics):
    # ensure metrics is either list, str, or None
    if not (isinstance(metrics, list) or isinstance(metrics, str) or metrics is None):
        raise ValueError(_METRICS_NOT_LIST_OR_STR_ERROR.format(type(metrics)))


def _convert_scalar_error_to_array_error(df_errors):
    if all((isinstance(error, float) or isinstance(error, int)) for error in df_errors):
        return df_errors.apply(lambda x: [x, x])
    if all((_is_arraylike(error) and len(error) == 2) for error in df_errors):
        return df_errors
    else:
        raise ValueError(_ERRORS_MUST_BE_SCALAR_OR_ARRAY)


def _check_valid_errors(df_errors):
    for tup in df_errors:
        if not all(ele >= 0 for ele in tup):
            raise ValueError(_ERRORS_NEGATIVE_VALUE_ERROR)


def _check_valid_conf_interval(df_conf_intervals):
    for tup in df_conf_intervals:
        if not _is_arraylike(tup) or len(tup) != 2:
            raise ValueError(_CONF_INTERVALS_MUST_BE_ARRAY)
        if tup[0] > tup[1]:
            raise ValueError(_CONF_INTERVALS_FLIPPED_BOUNDS_ERROR)


def _plot_df(df, metrics, kind, subplots, legend_label, df_all_errors=None, **kwargs):
    if df_all_errors is not None:
        yerr = np.array(
            [np.array([[*row] for row in df_all_errors[metric]]).T for metric in metrics]
        )
        if kind == "scatter":
            axs = df[metrics].plot(linestyle='', marker='o', yerr=yerr,
                                   subplots=subplots, **kwargs)
        else:
            axs = df[metrics].plot(kind=kind, yerr=yerr, subplots=subplots, **kwargs)

        if isinstance(axs, np.ndarray):
            for ax in axs.flatten():
                if kind == "scatter":
                    color = ax.lines[0].get_color()
                else:
                    color = "black"

                # extend legend with 95% CI text
                handles, labels = ax.get_legend_handles_labels()
                custom_line = [Line2D([0], [0], color=color, label=legend_label)]
                handles.extend(custom_line)
                ax.legend(handles=handles)
        else:
            if kind == "scatter":
                color = axs.lines[0].get_color()
            else:
                color = "black"

            # extend legend with 95% CI text
            handles, labels = axs.get_legend_handles_labels()
            custom_line = [Line2D([0], [0], color=color, label=legend_label)]
            handles.extend(custom_line)
            axs.legend(handles=handles)

    else:
        if kind == "scatter":
            axs = df[metrics].plot(linestyle='', marker='o', subplots=subplots, **kwargs)
        else:
            axs = df[metrics].plot(kind=kind, subplots=subplots, **kwargs)

    return axs


def plot_metric_frame(metric_frame: MetricFrame, *,
                      kind: str = "scatter",
                      metrics: Union[List[str], str] = None,
                      errors: Union[List[str], str] = None,
                      conf_intervals: Union[List[str], str] = None,
                      subplots: bool = True,
                      plot_error_labels: bool = False,
                      error_labels_precision: int = 4,
                      error_labels_fontsize: int = 8,
                      error_labels_color: str = "black",
                      error_labels_ha: str = "center",
                      legend_label: str = "95% CI",
                      **kwargs
                      ) -> Union[Axes, List[Axes]]:
    """Visualization for metrics with statistical error bounds.

    Plots a given metric and its given error (as described by the `errors` or `conf_intervals`)

    This function takes in a :class:`fairlearn.metrics.MetricFrame` with precomputed metrics
    and metric errors and a `errors` or `conf_intervals` array to interpret the columns
    of the :class:`fairlearn.metrics.MetricFrame`.

    The items at each index of the given `metrics` array and given `errors` or `conf_intervals`
    array should correspond to a pair of the same metric and metric error, respectively.

    Parameters
    ----------
    metric_frame : fairlearn.metrics.MetricFrame
        The collection of disaggregated metric values, along with the metric errors.

    kind : str, default="scatter"
        The type of plot to display. i.e. "bar", "line", etc.
        List of options is detailed in :meth:`pandas.DataFrame.plot`

    metrics : str or list of str
        The name of the metrics to plot.
        Should match columns from the given :class:`fairlearn.metrics.MetricFrame`.

    errors : str or list of str
        The name of the error metrics to plot.
        Should match columns from the given :class:`fairlearn.metrics.MetricFrame`.
        Errors quantify the bounds relative to the base metric.
        Errors must be non-negative.

        Example:
            If the error for a certain column is :code:`0.01`
            and the metric is 0.6,
            then the plotted bounds will be :code:`[0.59, 0.61]`

        Note:
            The return of the error function should be an scalar of the
            symmetric errors. i.e. :code:`0.01`

    conf_intervals : str or list of str
        The name of the confidence intervals to plot.
        Should match columns from the given :class:`fairlearn.metrics.MetricFrame`.

        Note:
            The return of the error function should be an array of the lower
            and upper bounds. i.e. :code:`[0.59, 0.62]`

    subplots : bool, default=True
        Whether or not to plot metrics on separate subplots

    plot_error_labels : bool, default=False
        Whether or not to plot numerical labels for the error bounds

    error_labels_precision : int, default=4
        The number of digits of precision to show for error labels

    error_labels_fontsize : int, default=8
        The font size to use for error labels

    error_labels_color : str, default="black"
        The font color to use for error labels

    error_labels_ha : str, default="center"
        The horizontal alignment modifier to use for error labels

    Returns
    -------
    :class:`matplotlib.axes.Axes` or :class:`numpy.ndarray` of them
    """
    _check_for_ambiguous_error_metric(errors, conf_intervals)
    _check_for_valid_metrics_format(metrics)

    metrics = [metrics] if isinstance(metrics, str) else metrics
    conf_intervals = [conf_intervals] if isinstance(conf_intervals, str) else conf_intervals
    errors = [errors] if isinstance(errors, str) else errors

    df = metric_frame.by_group

    # only plot metrics that aren't arrays (filters out metric errors)
    if metrics is None:
        metrics = []
        for metric in list(df):
            if not _is_arraylike(df[metric][0]):
                metrics.append(metric)

    _check_if_metrics_and_error_metrics_same_length(metrics, errors, conf_intervals)
    _check_if_metrics_length_zero(metrics)

    # plotting without errors or confidence intervals
    # Note: Returns early
    if errors is None and conf_intervals is None:
        axs = _plot_df(df, metrics, kind, subplots, legend_label, **kwargs)
        return axs

    df_all_errors = pd.DataFrame([])
    df_all_bounds = pd.DataFrame([])
    # plotting with confidence intervals:
    if conf_intervals is not None:
        for metric, conf_interval in zip(metrics, conf_intervals):
            _check_valid_conf_interval(df[conf_interval])
            df_temp = pd.DataFrame([])
            df_temp[['lower', 'upper']] = pd.DataFrame(df[conf_interval].tolist(), index=df.index)
            df_temp['error'] = list(
                zip(df[metric] - df_temp['lower'], df_temp['upper'] - df[metric]))
            df_all_errors[metric] = df_temp['error']

            if plot_error_labels:
                df_all_bounds[metric] = df[conf_interval]
    # plotting with relative errors
    elif errors is not None:
        for metric, error_bar in zip(metrics, errors):
            df[error_bar] = _convert_scalar_error_to_array_error(df[error_bar])
            _check_valid_errors(df[error_bar])
            df_all_errors[metric] = df[error_bar]

            if plot_error_labels:
                df_error = pd.DataFrame([])
                df_error[['lower', 'upper']] = pd.DataFrame(df[error_bar].tolist(), index=df.index)
                df_error['error'] = list(
                    zip(df[metric] - df_error['lower'], df[metric] + df_error['upper']))
                df_all_bounds[metric] = df_error['error']

    axs = _plot_df(df, metrics, kind, subplots, legend_label, df_all_errors, **kwargs)

    # Error labels don't get plotted when subplots=False
    if plot_error_labels and kind == "bar" and subplots:
        for j, metric in enumerate(metrics):
            temp_axs = axs.flatten() if isinstance(axs, np.ndarray) else np.array([axs])
            y_min, y_max = temp_axs[j].get_ylim()
            y_range = y_max - y_min

            for i in range(len(temp_axs[j].patches)):
                temp_axs[j].text(i,
                                 df_all_bounds[metric][i][0] - 0.05 * y_range,
                                 round(df_all_bounds[metric][i][0], error_labels_precision),
                                 fontsize=error_labels_fontsize,
                                 color=error_labels_color,
                                 ha=error_labels_ha)
                temp_axs[j].text(i,
                                 df_all_bounds[metric][i][1] + 0.01 * y_range,
                                 round(df_all_bounds[metric][i][1], error_labels_precision),
                                 fontsize=error_labels_fontsize,
                                 color=error_labels_color,
                                 ha=error_labels_ha)

    return axs
