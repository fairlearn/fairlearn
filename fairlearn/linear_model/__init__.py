# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""This module contains existing linear models that have
 been adapted to work with fairness constraints."""

from ._fair_logistic import ConstrainedLogisticRegression

__all__ = [
    "ConstrainedLogisticRegression"
]
