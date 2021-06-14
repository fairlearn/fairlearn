# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Utility class for plotting metrics with error ranges."""

from logging import error
from typing import Any, Dict, List, Union
import pandas as pd

from matplotlib.axes._axes import Axes

from ._metric_frame import MetricFrame

_MATPLOTLIB_IMPORT_ERROR_MESSAGE = "Please make sure to install matplotlib to use " \
                                   "the plotting for MetricFrame."

_GIVEN_BOTH_ERROR_BARS_AND_CONF_INT_ERROR = "Ambiguous Error Metric. Please only provide one of: error_bars and conf_intervals."

_METRICS_NOT_LIST_OR_STR_ERROR = "Metric should be a string of a single metric or a list of the metrics in provided MetricFrame, but {0} was provided"
_NEG_ERROR_OR_FLIPPED_BOUNDS_ERROR = "upper_error, lower_error, and symmetric_error must be positive and upper_bound must be greater than lower_bound"
_INVALID_ERROR_MAPPING = "Invalid error mapping, each metric must have all the key-value pairs that make up an Error Format"

def plot_metric_frame(plot_type: str, metric_frame: MetricFrame, 
                    metrics: Union[List[str], str]=None, error_bars: Union[List[str], str]=None, conf_intervals: Union[List[str], str]=None,
                    ax=None, show_plot=True, plot_error_labels: bool=True,
                    text_precision_digits: int=4, text_fontsize: int=8, text_color: str="black", text_ha: str="center",
                    title=None, capsize=10, colormap="Pastel1", TEMP_plot_with_df=True):
    """Visualization for metrics with statistical error bounds.

    Plots a given metric and its given error (as described by the `error_mapping`)

    This function takes in a :class:`fairlearn.metrics.MetricFrame` with precomputed metrics and metric errors
    and a `error_mapping` dict to interpret the columns (??) of the `MetricFrame`.

    Note:
        If multiple valid error_mappings are provided, this function will prioritize in the following order:
        1. upper_bound and lower_bound
        2. upper_error and lower_error
        3. symmetric_error
    
    Parameters
    ----------
    plot_type : str
        The type of plot to display. i.e. "bar", "line", etc

    metric_frame : fairlearn.metrics.MetricFrame
        The collection fo disaggregated metric values, along with the metric errors.

    metric : str
        The name of the metric to plot. Should match a column from the given `MetricFrame`

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

    ax : matplotlib.axes._axes.Axes, optional
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

    # ambiguous error metric
    if error_bars is not None and conf_intervals is not None:
        raise RuntimeError(_GIVEN_BOTH_ERROR_BARS_AND_CONF_INT_ERROR)

    if ax is None:
        ax = plt.axes(title=title)
    df = metric_frame.by_group

    assert isinstance(metrics, list) or isinstance(metrics, str) or metrics is None, _METRICS_NOT_LIST_OR_STR_ERROR.format(type(metrics))
    
    metrics = [metrics] if isinstance(metrics, str) else metrics
    conf_intervals = [conf_intervals] if isinstance(conf_intervals, str) else conf_intervals
    error_bars = [error_bars] if isinstance(error_bars, str) else error_bars

    # plotting without error
    if error_bars is None and conf_intervals is None:
        # ax.bar(x=df[metrics].index, height=df[metrics])
        ax = df.plot(kind=plot_type, y=metrics, ax=ax, colormap=colormap, capsize=capsize, 
                 title=title)
        return ax

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
    else:
        raise AssertionError(_INVALID_ERROR_MAPPING)
    
    ax = df.plot(kind=plot_type, y=metrics, yerr=df_all_errors, ax=ax, colormap=colormap, capsize=capsize, 
                     title=title, subplots=True)

    # # External API work with axes (sci-kit learn compatible)
    # # array of axes
    # if TEMP_plot_with_df:
    #     ax = df.plot(kind=plot_type, y=metric, yerr=df_error, ax=ax, colormap=colormap, capsize=capsize, 
    #                  title=title)
    # else:
    #     ax.bar(x=df[metric].index, height=df[metric])
    #     ax.errorbar(x=df[metric].index, y=df[metric], yerr=df["Recall Error"], ecolor="black", capsize=capsize)

    # TODO: Check assumption of plotting items in the vertical direction

    print(df_all_bounds)

    if plot_error_labels:
        for j, metric in enumerate(metrics):
            y_min, y_max = ax[j].get_ylim()
            y_range = y_max - y_min

            for i in range(len(ax[j].patches)):
                ax[j].text(i, df_all_bounds[metric][i][0] - 0.05 * y_range, round(df_all_bounds[metric][i][0],
                        text_precision_digits), fontsize=text_fontsize, color=text_color,
                        ha=text_ha)
                ax[j].text(i, df_all_bounds[metric][i][1] + 0.01 * y_range, round(df_all_bounds[metric][i][1],
                        text_precision_digits), fontsize=text_fontsize, color=text_color,
                        ha=text_ha)

    if show_plot:
        plt.show()

    return ax

