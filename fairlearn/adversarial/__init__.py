# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""Adversarial techniques to help mitigate fairness disparities."""

from ._adversarial_mitigation import AdversarialFairness, \
    AdversarialFairnessClassifier, AdversarialFairnessRegressor
from ._preprocessor import FloatTransformer

__all__ = [
    "AdversarialFairness",
    "AdversarialFairnessClassifier",
    "AdversarialFairnessRegressor",
    "FloatTransformer"]
