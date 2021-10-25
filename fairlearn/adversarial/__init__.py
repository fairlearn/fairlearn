# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""Adversarial techniques to help mitigate fairness disparities."""

from ._adversarial_mitigation import AdversarialMitigationBase, AdversarialClassifier, AdversarialRegressor

__all__ = ["AdversarialMitigationBase", "AdversarialClassifier", "AdversarialRegressor"]
