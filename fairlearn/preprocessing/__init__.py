# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

"""This module contains preprocessing steps that help make pipelines more fair."""

from ._information_filter import InformationFilter

__all__ = ["InformationFilter"]
