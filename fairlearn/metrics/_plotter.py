# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""Utility class for plotting metrics with error ranges."""

from typing import List, Union
import pandas as pd

from ._metric_frame import MetricFrame

_MATPLOTLIB_IMPORT_ERROR_MESSAGE = "Please make sure to install matplotlib to use " \
                                   "the plotting for MetricFrame."

_GIVEN_BOTH_ERROR_BARS_AND_CONF_INT_ERROR = \
    "Ambiguous Error Metric. Please only provide one of: error_bars or conf_intervals."
_METRIC_AND_ERROR_BARS_NOT_SAME_LENGTH_ERROR = \
    "The list of metrics and list of error_bars are not the same length."
_METRIC_AND_CONF_INTERVALS_NOT_SAME_LENGTH_ERROR = \
    "The list of metrics and list of conf_intervals are not the same length."

_METRICS_NOT_LIST_OR_STR_ERROR = \
    """Metric should be a string of a single metric or a list of the metrics
     in provided MetricFrame, but {0} was provided"""
_ERROR_BARS_MUST_BE_TUPLE = "Calculated error_bars must be tuple of length 2"
_CONF_INTERVALS_MUST_BE_TUPLE = "Calculated conf_intervals must be tuple of length 2"
_ERROR_BARS_NEGATIVE_VALUE_ERROR = "Calculated relative errors from error_bars must be positive"
_CONF_INTERVALS_FLIPPED_BOUNDS_ERROR = \
    "Calculated conf_intervals' upper bound cannot be less than lower bound"


def _check_if_metrics_and_error_metrics_same_length(metrics, error_bars, conf_intervals):
    if error_bars is not None:
        if len(error_bars) != len(metrics):
            raise ValueError(_METRIC_AND_ERROR_BARS_NOT_SAME_LENGTH_ERROR)
    elif conf_intervals is not None:
        if len(conf_intervals) != len(metrics):
            raise ValueError(_METRIC_AND_CONF_INTERVALS_NOT_SAME_LENGTH_ERROR)


def _check_for_ambiguous_error_metric(error_bars, conf_intervals):
    # ambiguous error metric if both are provided
    if error_bars is not None and conf_intervals is not None:
        raise ValueError(_GIVEN_BOTH_ERROR_BARS_AND_CONF_INT_ERROR)


def _check_for_valid_metrics_format(metrics):
    # ensure metrics is either list, str, or None
    if not (isinstance(metrics, list) or isinstance(metrics, str) or metrics is None):
        raise ValueError(_METRICS_NOT_LIST_OR_STR_ERROR.format(type(metrics)))


def _check_valid_error_bars(df_error_bars):
    for tup in df_error_bars:
        if not isinstance(tup, tuple) or len(tup) != 2:
            raise ValueError(_ERROR_BARS_MUST_BE_TUPLE)
        if not all(ele >= 0 for ele in tup):
            raise ValueError(_ERROR_BARS_NEGATIVE_VALUE_ERROR)


def _check_valid_conf_interval(df_conf_intervals):
    for tup in df_conf_intervals:
        if not isinstance(tup, tuple) or len(tup) != 2:
            raise ValueError(_CONF_INTERVALS_MUST_BE_TUPLE)
        if tup[0] > tup[1]:
            raise ValueError(_CONF_INTERVALS_FLIPPED_BOUNDS_ERROR)


def plot_metric_frame(metric_frame: MetricFrame,
                      kind: str = "scatter",
                      metrics: Union[List[str], str] = None,
                      error_bars: Union[List[str], str] = None,
                      conf_intervals: Union[List[str], str] = None,
                      axs=None,
                      show_plot=True,
                      plot_error_labels: bool = False,
                      text_precision_digits: int = 4,
                      text_fontsize: int = 8,
                      text_color: str = "black",
                      text_ha: str = "center",
                      capsize=10,
                      colormap="Pastel1",
                      subplot_shape=None,
                      figsize=None
                      ):
    """Visualization for metrics with statistical error bounds.

    Plots a given metric and its given error (as described by the `error_bars` or `conf_intervals`)

    This function takes in a :class:`fairlearn.metrics.MetricFrame` with precomputed metrics
    and metric errors and a `error_bars` or `conf_intervals` array to interpret the columns
    of the `MetricFrame`.

    The items at each index of the given `metrics` array and given `error_bars` or `conf_intervals`
    array should correspond to a pair of the same metric and metric error, respectively.

    Note:
        If given many metrics to plot, this function plots them in only 1 row
        with the number of subplot columns matching the number of metrics

    Parameters
    ----------
    metric_frame : fairlearn.metrics.MetricFrame
        The collection fo disaggregated metric values, along with the metric errors.

    kind : str, optional
        The type of plot to display. i.e. "bar", "line", etc.
        List of options is detailed in `pandas.DataFrame.plot`

    metrics : str or list of str
        The name of the metrics to plot. Should match columns from the given `MetricFrame`.

    error_bars : str or list of str
        The name of the error metrics to plot. Should match columns from the given `MetricFrame`.
        Error bars quantify the bounds relative to the base metric.

        Example:
            If the error bar for a certain column is [0.01, 0.02]
            and the metric is 0.6
            Then the plotted bounds will be [0.59, 0.62]

        Note:
            The return of the error bar function should be a tuple of the lower
            and upper errors

    conf_intervals : str or list of str
        The name of the confidence intervals to plot.
        Should match columns from the given `MetricFrame`.

        Note:
            The return of the error bar function should be a tuple of the lower
            and upper bounds

    axs : matplotlib.axes._axes.Axes or list of matplotlib.axes._axes.Axes, optional
        Custom Matplotlib axes to which this function can plot

    show_plot : bool, optional
        Whether to show the plot or not

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

    Returns
    -------
    matplotlib.axes._axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError(_MATPLOTLIB_IMPORT_ERROR_MESSAGE)

    _check_for_ambiguous_error_metric(error_bars, conf_intervals)
    _check_for_valid_metrics_format(metrics)

    metrics = [metrics] if isinstance(metrics, str) else metrics
    conf_intervals = [conf_intervals] if isinstance(conf_intervals, str) else conf_intervals
    error_bars = [error_bars] if isinstance(error_bars, str) else error_bars

    df = metric_frame.by_group

    # only plot metrics that aren't tuples (filters out metric errors)
    if metrics is None:
        metrics = []
        for metric in list(df):
            if not isinstance(df[metric][0], tuple):
                metrics.append(metric)

    _check_if_metrics_and_error_metrics_same_length(metrics, error_bars, conf_intervals)

    if axs is None:
        if subplot_shape is None:
            if len(metrics) > 0:
                subplot_shape = (1, len(metrics))
            else:
                subplot_shape = (1, 1)
        fig, axs = plt.subplots(*subplot_shape, squeeze=False, figsize=figsize)
        axs = axs.flatten()

    # plotting without error bars or confidence intervals
    # Note: Returns early
    df['_index'] = df.index
    if error_bars is None and conf_intervals is None:
        for metric, ax in zip(metrics, axs):
            previous_xlabel = ax.get_xlabel()
            ax = df.plot(x="_index", y=metric, kind=kind, ax=ax, colormap=colormap)
            ax.set_xlabel(previous_xlabel)

        if show_plot:
            plt.show()

        del df['_index']
        return axs

    df_all_errors = pd.DataFrame([])
    df_all_bounds = pd.DataFrame([])
    # plotting with confidence intervals:
    if conf_intervals is not None:
        for metric, conf_interval in zip(metrics, conf_intervals):
            _check_valid_conf_interval(df[conf_interval])
            df_all_errors[metric] = abs(df[metric] - df[conf_interval])

            if plot_error_labels:
                df_all_bounds[metric] = df[conf_interval]
    # plotting with relative errors
    elif error_bars is not None:
        for metric, error_bar in zip(metrics, error_bars):
            _check_valid_error_bars(df[error_bar])
            df_all_errors[metric] = df[error_bar]

            if plot_error_labels:
                df_error = pd.DataFrame([])
                df_error[['lower', 'upper']] = pd.DataFrame(df[error_bar].tolist(), index=df.index)
                df_error['error'] = list(
                    zip(df[metric] - df_error['lower'], df[metric] + df_error['upper']))
                df_all_bounds[metric] = df_error['error']

    for metric, ax in zip(metrics, axs):
        previous_xlabel = ax.get_xlabel()
        ax = df.plot(x="_index", y=metric, kind=kind,
                     yerr=df_all_errors, ax=ax, colormap=colormap)
        ax.set_xlabel(previous_xlabel)

    # TODO: Check assumption of plotting items in the vertical direction
    if plot_error_labels:
        for j, metric in enumerate(metrics):
            y_min, y_max = axs[j].get_ylim()
            y_range = y_max - y_min

            # TODO: Figure out if works for other plotting modes (besides bar)
            for i in range(len(axs[j].patches)):
                axs[j].text(i,
                            df_all_bounds[metric][i][0] - 0.05 * y_range,
                            round(df_all_bounds[metric][i][0], text_precision_digits),
                            fontsize=text_fontsize,
                            color=text_color,
                            ha=text_ha)
                axs[j].text(i,
                            df_all_bounds[metric][i][1] + 0.01 * y_range,
                            round(df_all_bounds[metric][i][1], text_precision_digits),
                            fontsize=text_fontsize,
                            color=text_color,
                            ha=text_ha)

    if show_plot:
        plt.show()

    del df['_index']
    return axs
