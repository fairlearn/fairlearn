# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from fairlearn.metrics import selection_rate_group_summary


def plot_disparities_in_selection_rate(y_true, y_pred, sensitive_features):
    """Plot disparities in selection for a binary classifier with a single sensitive feature.

    This helps visualize differences in overall selection rate, and selection rates
    for groups classified by sensitive attributes.

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
    selection_rate_summary = selection_rate_group_summary(y_true,
                                                          y_pred,
                                                          sensitive_features=sensitive_features)
    sensitive_values = sensitive_features.unique()
    selection_rates = []
    for sensitive_value in sensitive_values:
        selection_rates.append(selection_rate_summary['by_group'][sensitive_value])
    disparity = abs(selection_rates[0] - selection_rates[1])

    # chart text for localization
    title_text = 'Disparity in selection rate'
    xlabel_text = 'Selection rate'
    ylabel_text = 'By group'
    overall_performance_text = 'Overall'
    selection_rate_text = 'selection rate'
    disparity_text = 'disparity'

    # chart styles
    figsize = (12, 4)
    plt.rc('font', size=12)
    height = 0.4
    selection_rate_color = '#666'  # grey
    overall_vertical_line_color = '#333'
    label_padding = 2

    # bars
    labels = []
    for sensitive_value in sensitive_values:
        label = "{}\n{:.1%}\n{}".format(sensitive_value,
                                        selection_rate_summary['by_group'][sensitive_value],
                                        selection_rate_text)
        labels.append(label)
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(labels,
            selection_rates,
            height,
            color=[selection_rate_color],
            left=0)

    # labels
    for index, selection_rate in enumerate(selection_rates):
        plt.annotate("{:.1%}".format(selection_rate),
                     (selection_rate, index),
                     textcoords="offset points",
                     xytext=(label_padding, 0),
                     ha="left")

    # axes, titles, legend, etc
    plt.title("\n{}\n".format(title_text), fontsize=24, loc="left")
    right_title = "{}:\n{:.1%} {}\n{:.1%} {}\n".format(overall_performance_text,
                                                       selection_rate_summary['overall'],
                                                       selection_rate_text,
                                                       disparity,
                                                       disparity_text)
    plt.title(right_title, fontsize=12, loc="right")
    ax.set_ylabel(ylabel_text)
    ax.set_xlabel(xlabel_text)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None))
    ax.axvline(linewidth=1, color=overall_vertical_line_color, x=selection_rate_summary['overall'])
    plt.gca().set_xlim(0, 1)

    return None
