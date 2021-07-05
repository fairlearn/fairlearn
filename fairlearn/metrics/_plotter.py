# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Utility class for plotting metrics with error ranges."""

from logging import error
from typing import Any, Dict, List, Union
from matplotlib.pyplot import subplot, title
from numpy.lib.arraysetops import isin
import pandas as pd
import numpy as np

from matplotlib.axes._axes import Axes

from ._metric_frame import MetricFrame

_MATPLOTLIB_IMPORT_ERROR_MESSAGE = "Please make sure to install matplotlib to use " \
                                   "the plotting for MetricFrame."

_GIVEN_BOTH_ERROR_BARS_AND_CONF_INT_ERROR = "Ambiguous Error Metric. Please only provide one of: error_bars and conf_intervals."

_METRICS_NOT_LIST_OR_STR_ERROR = "Metric should be a string of a single metric or a list of the metrics in provided MetricFrame, but {0} was provided"
_NEG_ERROR_OR_FLIPPED_BOUNDS_ERROR = "upper_error, lower_error, and symmetric_error must be positive and upper_bound must be greater than lower_bound"
_INVALID_ERROR_MAPPING = "Invalid error mapping, each metric must have all the key-value pairs that make up an Error Format"

def plot_metric_frame(metric_frame: MetricFrame,
                     plot_type: str="scatter",
                     metrics: Union[List[str], str]=None,
                     error_bars: Union[List[str], str]=None,
                     conf_intervals: Union[List[str], str]=None,
                     axs=None,
                     show_plot=True,
                     plot_error_labels: bool=True,
                     text_precision_digits: int=4,
                     text_fontsize: int=8,
                     text_color: str="black",
                     text_ha: str="center",
                     capsize=10,
                     colormap="Pastel1",
                     subplot_shape=None,
                     figsize=None
                     ):
    """Visualization for metrics with statistical error bounds.

    Plots a given metric and its given error (as described by the `error_mapping`)

    This function takes in a :class:`fairlearn.metrics.MetricFrame` with precomputed metrics and metric errors
    and a `error_mapping` dict to interpret the columns (??) of the `MetricFrame`.
    
    Parameters
    ----------
    metric_frame : fairlearn.metrics.MetricFrame
        The collection fo disaggregated metric values, along with the metric errors.

    plot_type : str
        The type of plot to display. i.e. "bar", "line", etc

    metrics : str or list of str
        The name of the metrics to plot. Should match columns from the given `MetricFrame`

    axs : matplotlib.axes._axes.Axes, optional
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

    # ambiguous error metric if both are provided
    if error_bars is not None and conf_intervals is not None:
        raise ValueError(_GIVEN_BOTH_ERROR_BARS_AND_CONF_INT_ERROR)
    # TODO: Check if metrics and error_bars/conf_intervals same length 

    if not (isinstance(metrics, list) or isinstance(metrics, str) or metrics is None):
        raise ValueError(_METRICS_NOT_LIST_OR_STR_ERROR.format(type(metrics)))
    
    metrics = [metrics] if isinstance(metrics, str) else metrics
    conf_intervals = [conf_intervals] if isinstance(conf_intervals, str) else conf_intervals
    error_bars = [error_bars] if isinstance(error_bars, str) else error_bars

    df = metric_frame.by_group

    # only plot metrics that aren't tuples
    if metrics is None:
        metrics = []
        for metric in list(df):
            if not isinstance(df[metric][0], tuple) and metric != "_index":
                metrics.append(metric)


    if axs is None:
        if subplot_shape is None:
            subplot_shape = (1, len(metrics))
        fig, axs = plt.subplots(*subplot_shape, squeeze=False, figsize=figsize)
        axs = axs.flatten()

    # plotting without error
    df['_index'] = df.index
    if error_bars is None and conf_intervals is None:
        for metric, ax in zip(metrics, axs):
            previous_xlabel = ax.get_xlabel()
            ax = df.plot(x="_index", y=metric, kind=plot_type, ax=ax, colormap=colormap)
            ax.set_xlabel(previous_xlabel)
        del df['_index']
        return axs

    df_all_errors = pd.DataFrame([])
    df_all_bounds = pd.DataFrame([])
    # plotting with confidence intervals:
    if conf_intervals is not None:
        for metric, conf_interval in zip(metrics, conf_intervals):
            df_all_errors[metric] = abs(df[metric] - df[conf_interval])

            # assert bounds are valid
            #assert (df_lower_error >= 0).all(), _NEG_ERROR_OR_FLIPPED_BOUNDS_ERROR

            if plot_error_labels:
                df_all_bounds[metric] = df[conf_interval]
    # plotting with relative errors
    elif error_bars is not None:
        for metric, error_bar in zip(metrics, error_bars):
            df_all_errors[metric] = df[error_bar]

            # assert bounds are valid
            #assert (df_error >= 0).all(), _NEG_ERROR_OR_FLIPPED_BOUNDS_ERROR

            if plot_error_labels:
                df_error = pd.DataFrame([])
                df_error[['lower', 'upper']] = pd.DataFrame(df[error_bar].tolist(), index=df.index)
                df_error['error'] = list(zip(df[metric] - df_error['lower'], df[metric] + df_error['upper']))
                df_all_bounds[metric] = df_error['error']
    
    for metric, ax in zip(metrics, axs):
        previous_xlabel = ax.get_xlabel()
        ax = df.plot(x="_index", y=metric, kind=plot_type, yerr=df_all_errors, ax=ax, colormap=colormap)
        ax.set_xlabel(previous_xlabel)
    # TODO: Check assumption of plotting items in the vertical direction

    if plot_error_labels:
        for j, metric in enumerate(metrics):
            y_min, y_max = axs[j].get_ylim()
            y_range = y_max - y_min

            # TODO: Figure out if works for other plotting modes (besides bar)
            for i in range(len(axs[j].patches)):
                axs[j].text(i, df_all_bounds[metric][i][0] - 0.05 * y_range, round(df_all_bounds[metric][i][0],
                        text_precision_digits), fontsize=text_fontsize, color=text_color,
                        ha=text_ha)
                axs[j].text(i, df_all_bounds[metric][i][1] + 0.01 * y_range, round(df_all_bounds[metric][i][1],
                        text_precision_digits), fontsize=text_fontsize, color=text_color,
                        ha=text_ha)

    if show_plot:
        plt.show()

    del df['_index']
    return axs

