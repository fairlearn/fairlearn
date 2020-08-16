# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.


"""Module for visualization metrics in notebooks.

This module contains functions for visualizing metrics within
notebooks.
"""


from .plot_disparities_in_performance import plot_disparities_in_performance
from .plot_disparities_in_selection_rate import plot_disparities_in_selection_rate

__all__ = [
    "plot_disparities_in_performance",
    "plot_disparities_in_selection_rate"
]
