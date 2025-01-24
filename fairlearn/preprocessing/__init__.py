# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Preprocessing tools to help deal with sensitive attributes."""

from ._correlation_remover import CorrelationRemover
from ._fair_representation_learner import FairRepresentationLearner

__all__ = ["CorrelationRemover", "FairRepresentationLearner"]
