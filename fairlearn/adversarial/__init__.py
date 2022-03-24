# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""Adversarial techniques to help mitigate unfairness."""

from ._adversarial_mitigation import (
    AdversarialFairness,
    AdversarialFairnessClassifier,
    AdversarialFairnessRegressor,
)


__all__ = [
    "AdversarialFairness",
    "AdversarialFairnessClassifier",
    "AdversarialFairnessRegressor",
]
