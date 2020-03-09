# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Utilities for plotting curves."""

from ._constants import _MATPLOTLIB_IMPORT_ERROR_MESSAGE, DEMOGRAPHIC_PARITY, EQUALIZED_ODDS
from ._threshold_optimizer import ThresholdOptimizer, _SUPPORTED_CONSTRAINTS


def _get_debug_color(key):
    try:
        import matplotlib.cm as cm
        import matplotlib.colors
    except ImportError:
        raise RuntimeError(_MATPLOTLIB_IMPORT_ERROR_MESSAGE)
    tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=7)
    tab10_scalarMap = cm.ScalarMappable(norm=tab10_norm, cmap='Dark2')
    debug_colors = [tab10_scalarMap.to_rgba(x) for x in range(10)]
    debug_ncolors = len(debug_colors)
    debug_colormap = {}

    if key not in debug_colormap:
        color = debug_colors[len(debug_colormap) % debug_ncolors]
        debug_colormap[key] = color
    return debug_colormap[key]


def _plot_solution(ax, x_best, y_best, solution_label, xlabel, ylabel):
    """Plot the given solution with appropriate labels."""
    if y_best is None:
        ax.axvline(x=x_best, label=solution_label, ls='--')
    else:
        ax.plot(x_best, y_best, 'm*', ms=10, label=solution_label)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _plot_overlap(ax, x_grid, y_min):
    """Plot the overlap region."""
    highlight_color = [0.95, 0.90, 0.40]
    line, = ax.plot(x_grid, y_min, color=highlight_color, lw=8, label='overlap')
    line.zorder -= 1


def _plot_curve(ax, sensitive_feature, x_col, y_col, points):
    """Plot the given curve with labels."""
    color = _get_debug_color(sensitive_feature)
    ax.plot(points[x_col], points[y_col], c=color, ls='-', lw=2.0,
            label='sensitive feature value ' + str(sensitive_feature))


def _raise_if_not_threshold_optimizer(obj):
    if not isinstance(obj, ThresholdOptimizer):
        raise ValueError("Argument {} needs to be of type {}."
                         .format(obj.__name__, ThresholdOptimizer.__name__))


def plot_threshold_optimizer(threshold_optimizer, ax=None, show_plot=True):
    """Plot the chosen solution of the threshold optimizer.

    For `fairlearn.postprocessing.ThresholdOptimizer` objects that have their
    constraint set to `'demographic_parity'` this will result in a
    selection/error curve plot. For `fairlearn.postprocessing.ThresholdOptimizer`
    objects that have their constraint set to `'equalized_odds'` this will
    result in a ROC curve plot.

    :param threshold_optimizer: the `ThresholdOptimizer` instance for which the
        results should be illustrated.
    :type threshold_optimizer: fairlearn.postprocessing.ThresholdOptimizer
    :param ax: a custom `matplotlib.axes.Axes` object to use for the plots, default None
    :type ax: `matplotlib.axes.Axes`
    :param show_plot: whether or not the generated plot should be shown, default True
    :type show_plot: bool
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError(_MATPLOTLIB_IMPORT_ERROR_MESSAGE)

    _raise_if_not_threshold_optimizer(threshold_optimizer)

    if threshold_optimizer.constraints == DEMOGRAPHIC_PARITY:
        for sensitive_feature_value in threshold_optimizer._selection_error_curve.keys():
            _plot_curve(ax, sensitive_feature_value, 'selection', 'error',
                        threshold_optimizer._selection_error_curve[sensitive_feature_value])

        if ax is None:
            ax = plt.figure()
        _plot_solution(ax, threshold_optimizer._x_best, None, "DP solution",
                       "selection rate", "error")
    elif threshold_optimizer.constraints == EQUALIZED_ODDS:
        for sensitive_feature_value in threshold_optimizer._roc_curve.keys():
            _plot_curve(ax, sensitive_feature_value, 'x', 'y',
                        threshold_optimizer._roc_curve[sensitive_feature_value])

        if ax is None:
            ax = plt.figure()
        _plot_overlap(ax, threshold_optimizer._x_grid, threshold_optimizer._y_min)
        _plot_solution(ax, threshold_optimizer._x_best, threshold_optimizer._y_best,
                       'EO solution', "$P[\\hat{Y}=1|Y=0]$", "$P[\\hat{Y}=1|Y=1]$")
    else:
        raise ValueError("The plot can only be generated for a ThresholdOptimizer "
                         "object with constraints {}."
                         .format(" or ".join(_SUPPORTED_CONSTRAINTS)))

    if show_plot:
        plt.show()
