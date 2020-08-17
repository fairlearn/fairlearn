# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import accuracy_score
from fairlearn.metrics import group_summary, false_positive_rate, false_negative_rate


def plot_disparities_in_performance(y_true, y_pred, sensitive_features):
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
        the sensitive attributes

    Returns
    -------
    None
        The function renders a matplotlib plot
    """
    # compute
    accuracy_summary = group_summary(
        accuracy_score,
        y_true,
        y_pred,
        sensitive_features=sensitive_features)
    fp_summary = group_summary(
        false_positive_rate,
        y_true,
        y_pred,
        sensitive_features=sensitive_features)
    fn_summary = group_summary(
        false_negative_rate,
        y_true,
        y_pred,
        sensitive_features=sensitive_features)
    sensitive_values = sensitive_features.unique()
    overpredictions = []
    underpredictions = []
    for sensitive_value in sensitive_values:
        overpredictions.append(fp_summary['by_group'][sensitive_value])
        underpredictions.append(fn_summary['by_group'][sensitive_value])
    disparity = abs(
        accuracy_summary['by_group'][sensitive_values[0]]
        - accuracy_summary['by_group'][sensitive_values[1]])

    # chart text for localization
    title_text = 'Disparity in performance'
    xlabel_text = 'Prediction error rate'
    ylabel_text = 'By group'
    overall_performance_text = 'Overall performance'
    accuracy_text = 'accuracy'
    disparity_text = 'disparity'
    underprediction_legend_text = "Underprediction\n  predicted=0\n  true=1"
    overprediction_legend_text = "Overprediction\n  predicted=1\n  true=0"

    # chart styles
    figsize = (12, 4)
    plt.rc('font', size=12)
    height = 0.4
    zero_vertical_line_color = '#333'
    underprediction_color = '#FF7F0E'  # orange
    overprediction_color = '#2077B4'  # blue
    label_padding = 2

    # bars
    labels = []
    for sensitive_value in sensitive_values:
        labels.append("{}\n{:.1%}\n{}".format(
            sensitive_value,
            accuracy_summary['by_group'][sensitive_value],
            accuracy_text))
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        labels,
        underpredictions,
        height,
        color=[underprediction_color],
        left=list(-1*p for p in underpredictions),
        label=underprediction_legend_text)
    ax.barh(
        labels,
        overpredictions,
        height,
        color=[overprediction_color],
        left=0,
        label=overprediction_legend_text)

    # labels
    for index, overprediction in enumerate(overpredictions):
        plt.annotate(
            "{:.1%}".format(overprediction),
            (overprediction, index),
            textcoords="offset points",
            xytext=(label_padding, 0),
            ha="left")
    for index, underprediction in enumerate(underpredictions):
        plt.annotate(
            "{:.1%}".format(underprediction),
            (-1*underprediction, index),
            textcoords="offset points",
            xytext=(-1*label_padding, 0),
            ha="right")

    # axes, titles, legend, etc
    plt.title("\n{}\n".format(title_text), fontsize=24, loc="left")
    right_title = "{}:\n{:.1%} {}\n{:.1%} {}\n".format(
        overall_performance_text,
        accuracy_summary['overall'],
        accuracy_text,
        disparity,
        disparity_text)
    plt.title(right_title, fontsize=12, loc="right")
    ax.set_ylabel(ylabel_text)
    ax.set_xlabel(xlabel_text)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None))
    ax.axvline(linewidth=1, color=zero_vertical_line_color)
    plt.gca().set_xlim(-1, 1)
    ax.legend(loc='upper right', labelspacing=2, fontsize=10)

    return None
