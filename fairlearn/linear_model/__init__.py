# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""This module contains existing linear models adapted to work with fairness constraints."""

from ._constrained_logistic import ConstrainedLogisticRegression

__all__ = [
    "ConstrainedLogisticRegression"
]
