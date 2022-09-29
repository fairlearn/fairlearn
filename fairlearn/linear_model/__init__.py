# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""This module contains linear models adapted to adhere to fairness constraints."""

from ._constrained_logistic import ConstrainedLogisticRegression

__all__ = [
    "ConstrainedLogisticRegression"
]
