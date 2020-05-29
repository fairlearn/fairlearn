"""This module contains datasets that can be used for benchmarking and education."""


from ._fetch_adult import fetch_adult
from ._fetch_data import fetch_boston

__all__ = [
    "fetch_adult",
    "fetch_boston"
]
