# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.


"""This module contains datasets that can be used for benchmarking and education."""


from ._fetch_adult import fetch_adult
from ._fetch_boston import fetch_boston

__all__ = [
    "fetch_adult",
    "fetch_boston",
]
