# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""This module contains algorithms implementing the reductions approach to disparity mitigation.

In this approach, disparity constraints are cast as Lagrange multipliers, which cause the
reweighting and relabelling of the input data. This *reduces* the problem back to standard machine
learning training.
"""

from ._exponentiated_gradient import ExponentiatedGradient
from ._grid_search import GridSearch
from ._moments import BoundedGroupLoss
from ._moments import EqualizedOdds
from ._moments import ErrorRate
from ._moments import TruePositiveRateParity
from ._moments import (
    AbsoluteLoss,
    ClassificationMoment,
    DemographicParity,
    ErrorRateParity,
    FalsePositiveRateParity,
    LossMoment,
    MeanLoss,
    Moment,
    SquareLoss,
    UtilityParity,
    ZeroOneLoss,
)

__all__ = [
    "ExponentiatedGradient",
    "GridSearch",
    "AbsoluteLoss",
    "MeanLoss",
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
