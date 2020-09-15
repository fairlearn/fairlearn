# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""This module contains algorithms implementing the reductions approach to disparity mitigation.

In this approach, disparity constraints are cast as Lagrange multipliers, which cause the
reweighting and relabelling of the input data. This *reduces* the problem back to standard machine
learning training.
"""

from ._exponentiated_gradient import ExponentiatedGradient  # noqa: F401
from ._grid_search import GridSearch  # noqa: F401
from ._moments import AbsoluteLoss, Moment, UtilityParity  # noqa: F401
from ._moments import (DemographicParity, EqualizedOdds,  # noqa: F401
                       ErrorRateParity, ErrorRate,  # noqa: F401
                       TruePositiveRateParity,   # noqa: F401
                       FalsePositiveRateParity)   # noqa: F401
from ._moments import BoundedGroupLoss, SquareLoss, ZeroOneLoss  # noqa: F401
from ._moments import ClassificationMoment, LossMoment  # noqa: F401

_exponentiated_gradient = [
    "ExponentiatedGradient",
]

_grid_search = [
    "GridSearch",
]

_moments = [
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
    "ZeroOneLoss"
]

__all__ = [] + _exponentiated_gradient + _grid_search + _moments
