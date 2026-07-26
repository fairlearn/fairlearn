# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Methods which operate on a predictor, rather than an estimator.

The predictor's output is adjusted to fulfill specified parity constraints. The postprocessors
learn how to adjust the predictor's output from the training data.
"""

from ._interpolated_thresholder import InterpolatedThresholder
from ._plotting import plot_threshold_optimizer
from ._threshold_operation import ThresholdOperation
from ._threshold_optimizer import ThresholdOptimizer

__all__ = [
    "InterpolatedThresholder",
    "ThresholdOperation",
    "ThresholdOptimizer",
    "plot_threshold_optimizer",
]
