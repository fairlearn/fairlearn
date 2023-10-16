# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""Utility class for plotting metrics with and without confidence interval ranges."""

from typing import List, Union

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.utils import check_consistent_length

from ._metric_frame import MetricFrame

_METRIC_FRAME_INVALID_ERROR = "Input metric_frame should be of type MetricFrame."
_METRIC_LENGTH_ZERO_ERROR = (
    "No metrics were provided to plot. A nonzero number of metrics is required."
)
_METRICS_NOT_LIST_OR_STR_ERROR = (
    "Metric should be a string of a single metric or a list of the metrics"
    "in provided MetricFrame, but {0} was provided"
)
_CONF_INTERVALS_MUST_BE_ARRAY = "Calculated conf_intervals must be array of length 2"
_CONF_INTERVALS_FLIPPED_BOUNDS_ERROR = (
    "Calculated conf_intervals' upper bound cannot be less than lower bound"
)


def _is_arraylike(input_):
    return isinstance(input_, np.ndarray) or isinstance(input_, list)


def _build_legend(ax, kind, legend_label):
    """Take an axis and builds a custom legend.

    Adds a legend item based off of `legend_label` for confidence intervals

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        The Axes for which the legend will be modified.

    kind : str
        The type of plot to display, e.g., "point", "bar", "line", etc.
        The supported values are "point" and those listed in :meth:`pandas.DataFrame.plot`

    legend_label : str
        The label corresponding to the confidence interval bars
    """
    color = ax.lines[0].get_color() if kind == "point" else "black"

    # extend legend with user-provided legend_label
    handles, labels = ax.get_legend_handles_labels()
    custom_line = [Line2D([0], [0], color=color, label=legend_label)]
    handles.extend(custom_line)
    ax.legend(handles=handles)


def _plot_df(df, metrics, kind, subplots, legend_label, df_all_errors=None, **kwargs):
    r"""Plot the data with or without errors.

    Parameters
    ----------
    df : pd.DataFrame
        The collection of disaggregated metric values, along with the metric errors.

    metrics : str or list of str
        The name of the metrics to plot.
        Should match columns from the given :class:`fairlearn.metrics.MetricFrame`.

    kind : str
        The type of plot to display, e.g., "point", "bar", "line", etc.
        The supported values are "point" and those listed in :meth:`pandas.DataFrame.plot`

    subplots : bool
        Whether or not to plot metrics on separate subplots

    legend_label : str
        The label corresponding to the confidence interval bars

    df_all_errors : pd.DataFrame, default=None
        Optional argument for DataFrame that indicates the errors to plot

    \*\*kwargs
        Keyword arguments that are passed in to :meth:`pandas.DataFrame.plot`.

    Returns
    -------
    :class:`matplotlib.axes.Axes` or :class:`numpy.ndarray` of them
    """
    if df_all_errors is not None:
        yerr = np.array(
            [
                np.array([[*row] for row in df_all_errors[metric]]).T
                for metric in metrics
            ]
        )
        if kind == "point":
            axs = df[metrics].plot(
                linestyle="", marker="o", yerr=yerr, subplots=subplots, **kwargs
            )
        else:
            axs = df[metrics].plot(kind=kind, yerr=yerr, subplots=subplots, **kwargs)

        if isinstance(axs, np.ndarray):
            for ax in axs.flatten():
                _build_legend(ax, kind, legend_label)
        else:
            _build_legend(axs, kind, legend_label)

    else:
        if kind == "point":
            axs = df[metrics].plot(
                linestyle="", marker="o", subplots=subplots, **kwargs
            )
        else:
            axs = df[metrics].plot(kind=kind, subplots=subplots, **kwargs)

    return axs


def plot_metric_frame(
    metric_frame: MetricFrame,
    *,
    kind: str = "point",
    metrics: Union[List[str], str] = None,
    conf_intervals: Union[List[str], str] = None,
    subplots: bool = True,
    plot_ci_labels: bool = False,
    ci_labels_precision: int = 4,
    ci_labels_fontsize: int = 8,
    ci_labels_color: str = "black",
    ci_labels_ha: str = "center",
    ci_labels_legend: str = "Conf. Intervals",
    **kwargs,
):
    r"""Visualization for metrics with and without confidence intervals.

    Plots a given metric and its given error (as described by `conf_intervals`)

    This function takes in a :class:`fairlearn.metrics.MetricFrame` with precomputed metrics
    and metric errors and a `conf_intervals` array to interpret the columns
    of the :class:`fairlearn.metrics.MetricFrame`.

    The items at each index of the given `metrics` array and given `errors` or `conf_intervals`
    array should correspond to a pair of the same metric and metric error, respectively.

    Parameters
    ----------
    metric_frame : fairlearn.metrics.MetricFrame
        The collection of disaggregated metric values, along with the metric errors.

    kind : str, default="point"
        The type of plot to display, e.g., "point", "bar", "line", etc.
        The supported values are "point" and those listed in :meth:`pandas.DataFrame.plot`

    metrics : str or list of str
        The name of the metrics to plot.
        Should match columns from the given :class:`fairlearn.metrics.MetricFrame`.

    conf_intervals : str or list of str
        The name of the confidence intervals to plot.
        Should match columns from the given :class:`fairlearn.metrics.MetricFrame`.

        Note:
            The return of the error function should be an array of the lower
            and upper bounds. e.g. :code:`[0.59, 0.62]`

    subplots : bool, default=True
        Whether or not to plot metrics on separate subplots

    plot_ci_labels : bool, default=False
        Whether or not to plot numerical labels for the confidence intervals

    ci_labels_precision : int, default=4
        The number of digits of precision to show for confidence interval labels

    ci_labels_fontsize : int, default=8
        The font size to use for confidence interval labels

    ci_labels_color : str, default="black"
        The font color to use for confidence interval labels

    ci_labels_ha : str, default="center"
        The horizontal alignment modifier to use for confidence interval labels

    ci_labels_legend : str, default="Conf. Intervals"
        The label corresponding to the confidence interval bars

    \*\*kwargs
        Keyword arguments that are passed in to :meth:`pandas.DataFrame.plot`

    Returns
    -------
    :class:`matplotlib.axes.Axes` or :class:`numpy.ndarray` of them
    """
    # ensure metric_frame is a MetricFrame
    if not isinstance(metric_frame, MetricFrame):
        raise (ValueError(_METRIC_FRAME_INVALID_ERROR))
    # ensure metrics is either list, str, or None
    if not (isinstance(metrics, list) or isinstance(metrics, str) or metrics is None):
        raise ValueError(_METRICS_NOT_LIST_OR_STR_ERROR.format(type(metrics)))

    metrics = [metrics] if isinstance(metrics, str) else metrics
    conf_intervals = (
        [conf_intervals] if isinstance(conf_intervals, str) else conf_intervals
    )

    df = metric_frame.by_group

    # only plot metrics that aren't arrays (filters out metric errors)
    if metrics is None:
        metrics = []
        for metric in list(df):
            if not _is_arraylike(df[metric].iloc[0]):
                metrics.append(metric)

    check_consistent_length(metrics, conf_intervals)
    if len(metrics) == 0:
        raise ValueError(_METRIC_LENGTH_ZERO_ERROR)

    # plotting without confidence intervals
    # Note: Returns early
    if conf_intervals is None:
        axs = _plot_df(df, metrics, kind, subplots, ci_labels_legend, **kwargs)
        return axs

    # check for valid confidence intervals
    for conf_interval in conf_intervals:
        for tup in df[conf_interval]:
            if not _is_arraylike(tup) or len(tup) != 2:
                raise ValueError(_CONF_INTERVALS_MUST_BE_ARRAY)
            if tup[0] > tup[1]:
                raise ValueError(_CONF_INTERVALS_FLIPPED_BOUNDS_ERROR)

    df_all_errors = pd.DataFrame([])
    df_all_bounds = pd.DataFrame([])
    # plotting with confidence intervals:
    for metric, conf_interval in zip(metrics, conf_intervals):
        df_temp = pd.DataFrame([])
        df_temp[["lower", "upper"]] = pd.DataFrame(
            df[conf_interval].tolist(), index=df.index
        )
        df_temp["error"] = list(
            zip(df[metric] - df_temp["lower"], df_temp["upper"] - df[metric])
        )
        df_all_errors[metric] = df_temp["error"]

        if plot_ci_labels:
            df_all_bounds[metric] = df[conf_interval]

    axs = _plot_df(
        df, metrics, kind, subplots, ci_labels_legend, df_all_errors, **kwargs
    )

    # Confidence interval labels don't get plotted when subplots=False
    if plot_ci_labels and kind == "bar" and subplots:
        for j, metric in enumerate(metrics):
            temp_axs = axs.flatten() if isinstance(axs, np.ndarray) else np.array([axs])
            y_min, y_max = temp_axs[j].get_ylim()
            y_range = y_max - y_min

            for i in range(len(temp_axs[j].patches)):
                temp_axs[j].text(
                    i,
                    df_all_bounds[metric][i][0] - 0.05 * y_range,
                    round(df_all_bounds[metric][i][0], ci_labels_precision),
                    fontsize=ci_labels_fontsize,
                    color=ci_labels_color,
                    ha=ci_labels_ha,
                )
                temp_axs[j].text(
                    i,
                    df_all_bounds[metric][i][1] + 0.01 * y_range,
                    round(df_all_bounds[metric][i][1], ci_labels_precision),
                    fontsize=ci_labels_fontsize,
                    color=ci_labels_color,
                    ha=ci_labels_ha,
                )

    return axs
