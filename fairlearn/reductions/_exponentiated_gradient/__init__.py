# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Package for Exponentiated Gradient."""

from .exponentiated_gradient import ExponentiatedGradient  # noqa: F401
from .exponentiated_gradient import ExponentiatedGradientResult  # noqa: F401

__all__ = [
    "ExponentiatedGradient",
    "ExponentiatedGradientResult"
]
