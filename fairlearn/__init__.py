# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Tools for analyzing and mitigating disparity in Machine Learning models."""

from .show_versions import show_versions  # noqa: F401
from ._version import __version__

__all__ = ("__version__", )

__name__ = "fairlearn"
_base_version = __version__  # To enable the v0.4.6 docs
