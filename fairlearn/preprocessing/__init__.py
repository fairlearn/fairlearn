# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

"""This module contains preprocessing algorithms for unfairness mitigation."""

from ._correlation_remover import CorrelationRemover

__all__ = ["CorrelationRemover"]
