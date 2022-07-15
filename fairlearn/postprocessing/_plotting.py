# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Utilities for plotting curves."""

import math
import pandas as pd
import numpy as np
from ._constants import _MATPLOTLIB_IMPORT_ERROR_MESSAGE
from ._threshold_optimizer import ThresholdOptimizer
from ..utils._input_validation import _merge_columns
from sklearn.utils.validation import check_is_fitted, check_array

_debug_colors = None
_debug_ncolors = 10
_debug_colormap = {}


def _get_debug_color(key):
    global _debug_colors, _debug_ncolors, _debug_colormap
    try:
        import matplotlib.cm as cm
        import matplotlib.colors
    except ImportError:
        raise RuntimeError(_MATPLOTLIB_IMPORT_ERROR_MESSAGE)
    if _debug_colors is None:
        tab_norm = matplotlib.colors.Normalize(vmin=0, vmax=7)
        tab_scalarMap = cm.ScalarMappable(norm=tab_norm, cmap="Dark2")
        _debug_colors = [tab_scalarMap.to_rgba(x) for x in range(_debug_ncolors)]

    if key not in _debug_colormap:
        color = _debug_colors[len(_debug_colormap) % _debug_ncolors]
        _debug_colormap[key] = color
    return _debug_colormap[key]


def _plot_solution(ax, x_best, y_best, solution_label, xlabel, ylabel):
    """Plot the given solution with appropriate labels."""
    if y_best is None:
        ax.axvline(x=x_best, label=solution_label, ls="--")
    else:
        ax.plot(x_best, y_best, "m*", ms=10, label=solution_label)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _plot_overlap(ax, x_grid, y_min):
    """Plot the overlap region."""
    highlight_color = [0.95, 0.90, 0.40]
    (line,) = ax.plot(x_grid, y_min, color=highlight_color, lw=8, label="overlap")
    line.zorder -= 1


def _plot_curve(ax, sensitive_feature, x_col, y_col, points):
    """Plot the given curve with labels."""
    color = _get_debug_color(sensitive_feature)
    ax.plot(
        points[x_col],
        points[y_col],
        c=color,
        ls="-",
        lw=2.0,
        label="sensitive feature = " + str(sensitive_feature),
    )


def _raise_if_not_threshold_optimizer(obj):
    if not isinstance(obj, ThresholdOptimizer):
        raise ValueError(
            "Argument {} needs to be of type {}.".format(
                obj.__name__, ThresholdOptimizer.__name__
            )
        )


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
    check_is_fitted(threshold_optimizer)

    if ax is None:
        ax = plt.axes()

    for sensitive_feature_value in threshold_optimizer._tradeoff_curve.keys():
        _plot_curve(
            ax,
            sensitive_feature_value,
            "x",
            "y",
            threshold_optimizer._tradeoff_curve[sensitive_feature_value],
        )

    if threshold_optimizer.constraints == "equalized_odds":
        _plot_overlap(ax, threshold_optimizer._x_grid, threshold_optimizer._y_min)
        _plot_solution(
            ax,
            threshold_optimizer._x_best,
            threshold_optimizer._y_best,
            "solution",
            "$P[\\hat{Y}=1|Y=0]$",
            "$P[\\hat{Y}=1|Y=1]$",
        )
    else:
        _plot_solution(
            ax,
            threshold_optimizer._x_best,
            None,
            "solution",
            threshold_optimizer.x_metric_,
            threshold_optimizer.y_metric_,
        )

    if show_plot:
        plt.show()


"""Utilities to plot information about the output of an estimator"""


def _check_sensitive_features(sensitive_features):
    """Return in sensitive_features pd.Series format.

    If there are multiple sensitive features, merge into a single column.

    Parameters
    ----------
    sensitive_features : numpy.ndarray, list, pandas.DataFrame,\
        or pandas.Series
        Sensitive features to identify groups by

    Returns
    -------
    pd.Series
        The (merged) sensitive feature column
    """
    sensitive_features = check_array(
        sensitive_features, ensure_2d=False, dtype=None)

    if len(sensitive_features.shape) > 1 and sensitive_features.shape[1] > 1:
        sensitive_features = _merge_columns(sensitive_features)

    return pd.Series(sensitive_features.squeeze())


def plot_proba_distribution(sensitive_features, y_prob, specified_groups_to_plot=None):
    """Create a stacked barplot showing the distribution of probabilities.

    The created plot is grouped by sensitive feature value.

    Parameters
    ----------
    sensitive_features : numpy.ndarray, list, pandas.DataFrame,\
        or pandas.Series
        Sensitive features to identify groups by

    y_prob : numpy.ndarray
        Probabilities returned by a classifier

    specified_groups_to_plot : list or None, default=None
        If None, plot information about all known sensitive groups.
        If list, plot information about sensitive groups in list
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError(_MATPLOTLIB_IMPORT_ERROR_MESSAGE)

    sensitive_features = _check_sensitive_features(sensitive_features)

    sf_to_plot = specified_groups_to_plot if specified_groups_to_plot else np.unique(
        sensitive_features).tolist()

    samples_each_sf = sensitive_features.value_counts()

    n_rows = len(sensitive_features)

    # create dict to keep track of proba distribution for each sf, by creating array of the form
    # [#of probas in interval 0 - .1, ... , #of probas in interval .9 - 1]
    statistics_dict = {name: np.zeros(10) for name in sf_to_plot}

    # loop through A and y_prob simultaneously, and only once
    for i in range(n_rows):
        sf = sensitive_features.iloc[i]
        if sf in statistics_dict:
            stats_array = statistics_dict[sf]
            # append predict proba to correct interval
            stats_array[math.floor(y_prob[i]*10)] += 1

    # create arrays used for making the plot
    stats_arrays = statistics_dict.values()
    nmbr_sf_to_plot = len(sf_to_plot)
    bottom = np.zeros(nmbr_sf_to_plot)
    colorbrewer = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c',
                   '#fdbf6f', '#ff7f00', '#e31a1c', '#fb9a99', '#6a3d9a', '#cab2d6']
    labels = []
    handles = []

    # Start the making plot
    plt.figure(figsize=(nmbr_sf_to_plot, 6))
    for i in range(10):

        # Calculate percentages of probas in each interval for each group
        nmbr_of_probas_in_interval = np.array([stats_array[i] for stats_array in stats_arrays])
        nmbr_of_each_group = np.array([samples_each_sf[sf] for sf in sf_to_plot])
        nmbr_probas_percentages = nmbr_of_probas_in_interval/nmbr_of_each_group*100

        # Plot next bar
        handle = plt.bar(sf_to_plot, nmbr_probas_percentages,
                         bottom=bottom, color=colorbrewer[i], alpha=.8)

        # Keep track of handles and labels for the legend
        handles.insert(0, handle)
        labels.insert(0, '{} - {}'.format(i/10, (i+1)/10))

        # Create text annotation
        for xpos, ypos, yval in zip(sf_to_plot, bottom+nmbr_probas_percentages/2,
                                    nmbr_probas_percentages):
            if yval >= 5:
                plt.text(xpos, ypos, '{}%'.format(round(yval, 1)), ha="center", va="center")

        # Update y-positions for next bars
        bottom += nmbr_probas_percentages

    plt.ylabel('distribution of predict_proba')
    plt.xlabel('Sensitive feature')
    plt.xticks(rotation=45, ha='right')
    plt.legend(labels=labels, handles=handles, bbox_to_anchor=(1, .9))
    plt.show()


def plot_positive_predictions(sensitive_features, y_pred, specified_groups_to_plot=None):
    """Create barplot showing the percentage of positive predictions.

    The created plot is grouped by sensitive feature value.

    Parameters
    ----------
    sensitive_features : numpy.ndarray, list, pandas.DataFrame,\
        or pandas.Series
        Sensitive features to identify groups by

    y_pred : numpy.ndarray
        Binary predictions returned by a classifier

    specified_groups_to_plot : list or None, default=None
        If None, plot information about all known sensitive groups.
        If list, plot information about sensitive groups in list.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError(_MATPLOTLIB_IMPORT_ERROR_MESSAGE)

    sensitive_features = _check_sensitive_features(sensitive_features)

    sf_to_plot = specified_groups_to_plot if specified_groups_to_plot else np.unique(
        sensitive_features).tolist()

    samples_each_sf = sensitive_features.value_counts()

    n_rows = len(sensitive_features)

    # get statistics_dict to keep track the amount of postive predictions of each group
    statistics_dict = {name: 0 for name in sf_to_plot}

    # Keep track of the number of postive predictions for all groups,
    # by looping through y_pred only once
    for i in range(n_rows):
        sf = sensitive_features.iloc[i]
        if sf in statistics_dict:
            statistics_dict[sf] += y_pred[i]

    # calculate percentages
    percentages = np.array([round(statistics_dict[sf]/samples_each_sf[sf]*100, 1)
                            for sf in sf_to_plot])

    # create barplot
    plt.figure(figsize=(len(sf_to_plot), 6))
    plt.bar(sf_to_plot, percentages, color='#2364D2', alpha=.8)

    # Create text annotation
    for xpos, ypos, yval in zip(sf_to_plot, percentages/2, percentages):
        plt.text(xpos, ypos if ypos else 1, '{}%'.format(yval), ha="center", va="center")

    plt.ylabel('% positive (=1) predictions')
    plt.xlabel('Sensitive feature')
    plt.xticks(rotation=45, ha='right')
    plt.show()


def plot_histograms_per_group(sensitive_features, y_pred, specified_groups_to_plot=None):
    """Create hist of distribution of :code:`y_pred` for each sensitive feature value.

    Parameters
    ----------
    sensitive_features : numpy.ndarray, list, pandas.DataFrame,\
        or pandas.Series
        Sensitive features to identify groups by

    y_pred : numpy.ndarray
        Continuous output returned by an estimator. This can be
        regression output, probabilities obtained by calling
        :code:`predict_proba()`, or the decision function obtained by
        calling :code:`decision_function()`

    specified_groups_to_plot : list or None, default=None
        If None, plot information about all known sensitive groups.
        If list, plot information about sensitive groups in list.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError(_MATPLOTLIB_IMPORT_ERROR_MESSAGE)

    sensitive_features = _check_sensitive_features(sensitive_features)

    sf_to_plot = specified_groups_to_plot if specified_groups_to_plot else np.unique(
        sensitive_features).tolist()
    nmbr_sf_to_plot = len(sf_to_plot)

    n_rows = len(sensitive_features)

    # Create statistics_dict to keep track of all values for each group
    statistics_dict = {name: np.array([]) for name in sf_to_plot}

    # Keep track of all values for each group simultaneously,
    # by looping through y_pred only once
    for i in range(n_rows):
        sf = sensitive_features.iloc[i]
        if sf in statistics_dict:
            statistics_dict[sf] = np.append(statistics_dict[sf], y_pred[i])

    # Depending on the number of sf to plot, create a figure with subplots
    if nmbr_sf_to_plot > 2:
        nmbr_rows = math.ceil(nmbr_sf_to_plot / 2)
        fig, axs = plt.subplots(nmbr_rows, 2, figsize=(nmbr_rows*2+2, nmbr_rows*2+2))

        row = 0
        col = 0
        for sf, arrays in statistics_dict.items():

            axs[row, col].hist(arrays, bins=25, color='#2364D2', alpha=.8)
            axs[row, col].set_title(sf)

            if col:
                row += 1
                col -= 1
            else:
                col += 1

        if nmbr_sf_to_plot % 2:
            fig.delaxes(axs[nmbr_rows-1, 1])
        fig.tight_layout()

    elif nmbr_sf_to_plot == 2:
        fig, (ax1, ax2) = plt.subplots(2, figsize=(6, 6))
        ax1.hist(statistics_dict[sf_to_plot[0]], bins=25, color='#2364D2', alpha=.8)
        ax1.set_title(sf_to_plot[0])

        ax2.hist(statistics_dict[sf_to_plot[1]], bins=25, color='#2364D2', alpha=.8)
        ax2.set_title(sf_to_plot[1])

        fig.tight_layout()

    else:
        plt.hist(statistics_dict[sf_to_plot[0]], bins=25, color='#2364D2', alpha=.8)
        plt.title(sf_to_plot[0])
        plt.show()
