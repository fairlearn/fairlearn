# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""This module contains algorithms implementing the reductions approach to disparity mitigation.

In this approach, disparity constraints are cast as Lagrange multipliers, which cause the
reweighting and relabelling of the input data. This *reduces* the problem back to standard machine
learning training.
"""

from ._exponentiated_gradient import ExponentiatedGradient  # noqa: F401
from ._grid_search import GridSearch, GridSearchResult  # noqa: F401
from ._moments import AbsoluteLoss, Moment, ConditionalSelectionRate  # noqa: F401
from ._moments import DemographicParity, EqualizedOdds, ErrorRate   # noqa: F401
from ._moments import GroupLossMoment, SquareLoss, ZeroOneLoss  # noqa: F401
from ._moments import ClassificationMoment, LossMoment  # noqa: F401

_exponentiated_gradient = [
    "ExponentiatedGradient",
]

_grid_search = [
    "GridSearch",
    "GridSearchResult"
]

_moments = [
    "AbsoluteLoss",
    "Moment",
    "ClassificationMoment",
    "ConditionalSelectionRate",
    "DemographicParity",
    "EqualizedOdds",
    "ErrorRate",
    "GroupLossMoment",
    "LossMoment",
    "SquareLoss",
    "ZeroOneLoss"
]

__all__ = [] + _exponentiated_gradient + _grid_search + _moments
