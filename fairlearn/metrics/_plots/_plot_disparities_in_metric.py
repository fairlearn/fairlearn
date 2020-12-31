# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from fairlearn.metrics import MetricFrame


def plot_disparities_in_metric(metric, y_true, y_pred, sensitive_features, show_plot=True):
    """Plot disparities in a metric between groups defined by sensitive features.

    This method visualizes the metric values for all groups as identified
    by sensitive features.

    Parameters
    ----------
    metric : callable
        A metric function that takes arguments `y_true` and `y_pred`
    y_true : array-like
        The list of true values
    y_pred : array-like
        The list of predicted values
    sensitive_features : array-like
        the sensitive attributes

    """
    # compute
    metric_frame = MetricFrame(metric, y_true, y_pred,
                               sensitive_features=sensitive_features)

    # chart text for localization
    metric_text = metric.__name__.replace('_', ' ')
    title_text = f'Disparity in {metric_text}'
    ylabel_text = 'By group'
    overall_performance_text = 'Overall'
    diff_text = 'min-max difference'
    ratio_text = 'min-max ratio'
    # chart styles
    figsize = (12, 4)
    plt.rc('font', size=12)
    height = 0.4
    bar_color = '#666'  # grey
    overall_vertical_line_color = '#333'
    label_padding = 2

    # bars
    fig, ax = plt.subplots(figsize=figsize)
    n_groups = len(metric_frame.by_group)
    ax.barh(list(range(n_groups)[::-1]),
            metric_frame.by_group,
            height,
            tick_label=metric_frame.by_group.index,
            color=[bar_color],
            left=0)

    # labels
    height_index = n_groups - 1
    for _, group_metric in metric_frame.by_group.items():
        plt.annotate(f"{group_metric:.1%}",
                     (group_metric, height_index),
                     textcoords="offset points",
                     xytext=(label_padding, 0),
                     ha="left")
        height_index -= 1

    # axes, titles, legend, etc
    plt.title(f"\n{title_text}\n", fontsize=24, loc="left")
    right_title = f"{overall_performance_text}:\n" \
        f"{metric_frame.overall:.1%} {metric_text}\n" \
        f"{metric_frame.difference():.1%} {diff_text}\n" \
        f"{metric_frame.ratio():.1%} {ratio_text}\n"
    plt.title(right_title, fontsize=12, loc="right")
    ax.set_ylabel(ylabel_text)
    ax.set_xlabel(metric_text)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None))
    ax.axvline(linewidth=1, color=overall_vertical_line_color, x=metric_frame.overall)
    plt.gca().set_xlim(0, 1)

    # ensure everything is within the boundaries of the plot using tight layout
    plt.tight_layout()

    if show_plot:
        plt.show()

    return None
