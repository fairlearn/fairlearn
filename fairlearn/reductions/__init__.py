# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""This module contains algorithms implementing the reductions approach to disparity mitigation.

In this approach, disparity constraints are cast as Lagrange multipliers, which cause the
reweighting and relabelling of the input data. This *reduces* the problem back to standard machine
learning training.
"""

from ._exponentiated_gradient import ExponentiatedGradient  # noqa: F401
from ._grid_search import GridSearch  # noqa: F401
from ._moments import BoundedGroupLoss  # noqa: F401
from ._moments import EqualizedOdds  # noqa: F401
from ._moments import ErrorRate  # noqa: F401
from ._moments import TruePositiveRateParity  # noqa: F401
from ._moments import (
    AbsoluteLoss,
    ClassificationMoment,
    DemographicParity,
    ErrorRateParity,
    FalsePositiveRateParity,
    LossMoment,
    Moment,
    SquareLoss,
    UtilityParity,
    ZeroOneLoss,
)

__all__ = [
    "ExponentiatedGradient",
    "GridSearch",
    "AbsoluteLoss",
    "Moment",
    "ClassificationMoment",
    "UtilityParity",
    "DemographicParity",
    "EqualizedOdds",
    "TruePositiveRateParity",
    "FalsePositiveRateParity",
    "ErrorRateParity",
    "ErrorRate",
    "BoundedGroupLoss",
    "LossMoment",
    "SquareLoss",
    "ZeroOneLoss",
]
