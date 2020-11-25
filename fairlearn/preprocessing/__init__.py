# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Preprocessing tools to help deal with sensitive attributes."""

from ._linear_dep_remover import LinearDependenceRemover

__all__ = ["LinearDependenceRemover"]
