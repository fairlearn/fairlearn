# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""Adversarial techniques to help mitigate fairness disparities."""

from ._adversarial_mitigation import AdversarialFairness, \
    AdversarialFairnessClassifier, AdversarialFairnessRegressor, Keyword

__all__ = [
    "AdversarialFairness",
    "AdversarialFairnessClassifier",
    "AdversarialFairnessRegressor",
    "Keyword"]
