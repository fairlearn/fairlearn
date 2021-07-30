# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""This module contains methods which operate on a predictor, rather than an estimator.

The predictor's output is adjusted to fulfill specified parity constraints. The postprocessors
learn how to adjust the predictor's output from the training data.
"""

from ._threshold_optimizer import ThresholdOptimizer  # noqa: F401
from ._plotting import plot_threshold_optimizer  # noqa: F401

__all__ = [
    "ThresholdOptimizer",
    "plot_threshold_optimizer"
]
