# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Utilities for plotting curves."""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors

tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=7)
tab10_scalarMap = cm.ScalarMappable(norm=tab10_norm, cmap='Dark2')
debug_colors = [tab10_scalarMap.to_rgba(x) for x in range(10)]
debug_ncolors = len(debug_colors)
debug_colormap = {}

highlight_color = [0.95, 0.90, 0.40]


def _get_debug_color(key):
    if key not in debug_colormap:
        color = debug_colors[len(debug_colormap) % debug_ncolors]
        debug_colormap[key] = color
    return debug_colormap[key]


def plot_solution_and_show_plot(x_best, y_best, solution_label, xlabel, ylabel):
    """Plot the given solution with appropriate labels."""
    if y_best is None:
        plt.axvline(x=x_best, label=solution_label, ls='--')
    else:
        plt.plot(x_best, y_best, 'm*', ms=10, label=solution_label)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_overlap(x_grid, y_min):
    """Plot the overlap region."""
    line, = plt.plot(x_grid, y_min, color=highlight_color, lw=8, label='overlap')
    line.zorder -= 1


def plot_curve(sensitive_feature, x_col, y_col, points):
    """Plot the given curve with labels."""
    color = _get_debug_color(sensitive_feature)
    plt.plot(points[x_col], points[y_col], c=color, ls='-', lw=2.0,
             label='sensitive feature value ' + str(sensitive_feature))
