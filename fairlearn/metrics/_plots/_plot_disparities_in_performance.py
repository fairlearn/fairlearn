# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, false_positive_rate, false_negative_rate


def plot_disparities_in_performance(*, y_true, y_pred, sensitive_features, show_plot=True,
                                    ax=None):
    """Plot disparities in performance for a binary classifier across a single sensitive feature.

    This helps visualize differences in overall prediction error, as well as false positive
    and false negative error rates across the values of a single sensitive attribute
    (eg, group membership).

    Parameters
    ----------
    y_true : array-like
        The list of true values

    y_pred : array-like
        The list of predicted values

    sensitive_features : array-like
        The sensitive features

    show_plot : bool, default=True
        Whether to show the plot. By default, the plot is shown using the
        :func:`matplotlib.pyplot.show` function.

    ax : :class:`matplotlib.axes.Axes`, default=None
        :class:`matplotlib.axes.Axes` object to plot on. If `None`, a new
        figure and axes is created.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        axes object to configure further if desired

    """
    # compute
    accuracy_frame = MetricFrame(
        accuracy_score,
        y_true,
        y_pred,
        sensitive_features=sensitive_features)
    fp_frame = MetricFrame(
        false_positive_rate,
        y_true,
        y_pred,
        sensitive_features=sensitive_features)
    fn_frame = MetricFrame(
        false_negative_rate,
        y_true,
        y_pred,
        sensitive_features=sensitive_features)

    # chart text for localization
    title_text = 'Disparity in performance'
    ylabel_text = 'By group'
    overall_performance_text = 'Overall performance'
    accuracy_text = 'accuracy'
    diff_text = 'min-max difference'
    ratio_text = 'min-max ratio'
    fn_legend_text = "False negative rate"
    fp_legend_text = "False positive rate"

    # chart styles
    plt.rc('font', size=12)
    height = 0.4
    zero_vertical_line_color = '#333'
    fnr_color = '#FF7F0E'  # orange
    fpr_color = '#2077B4'  # blue
    label_padding = 2

    # bars
    labels = []
    for sensitive_feature_value in accuracy_frame.by_group.index:
        labels.append(f"{sensitive_feature_value}\n"
                      f"{accuracy_frame.by_group[sensitive_feature_value]:.1%}"
                      f"\n{accuracy_text}")

    if ax is None:
        _, ax = plt.subplots()

    ax.barh(
        labels,
        fn_frame.by_group,
        height,
        color=[fnr_color],
        left=fn_frame.by_group.map(lambda fn: -1 * fn),
        label=fn_legend_text)
    ax.barh(
        labels,
        fp_frame.by_group,
        height,
        color=[fpr_color],
        left=0,
        label=fp_legend_text)

    # labels
    for index, fp_rate in enumerate(fp_frame.by_group):
        plt.annotate(
            f"{fp_rate:.1%}",
            (fp_rate, index),
            textcoords="offset points",
            xytext=(label_padding, 0),
            ha="left")
    for index, fn_rate in enumerate(fn_frame.by_group):
        plt.annotate(
            f"{fn_rate:.1%}",
            (-1*fn_rate, index),
            textcoords="offset points",
            xytext=(-1*label_padding, 0),
            ha="right")

    # axes, titles, legend, etc
    plt.title("\n{}\n".format(title_text), fontsize=24, loc="left")
    right_title = f"{overall_performance_text}:\n" \
        f"{accuracy_frame.overall:.1%} {accuracy_text}\n" \
        f"{accuracy_frame.difference():.1%} {diff_text}\n" \
        f"{accuracy_frame.ratio():.1%} {ratio_text}\n"
    plt.title(right_title, fontsize=12, loc="right")
    ax.set_ylabel(ylabel_text)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None))
    ax.axvline(linewidth=1, color=zero_vertical_line_color)
    plt.gca().set_xlim(-1, 1)

    # position legend at the horizontally below the chart
    ax.legend(bbox_to_anchor=(0.5, -0.12), loc="upper center", ncol=2, fontsize=10)

    # ensure everything is within the boundaries of the plot using tight layout
    plt.tight_layout()

    if show_plot:
        plt.show()

    return ax
