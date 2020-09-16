# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Holds the various Moments."""

from .moment import Moment  # noqa: F401
from .moment import ClassificationMoment, LossMoment  # noq: F401

from .utility_parity import UtilityParity  # noqa: F401
from .utility_parity import DemographicParity, EqualizedOdds, \
    TruePositiveRateParity, FalsePositiveRateParity, ErrorRateParity  # noqa: F401
from .error_rate import ErrorRate   # noqa: F401
from .bounded_group_loss import BoundedGroupLoss   # noqa: F401
from .bounded_group_loss import SquareLoss, AbsoluteLoss, ZeroOneLoss  # noqa: F401

__all__ = [
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
