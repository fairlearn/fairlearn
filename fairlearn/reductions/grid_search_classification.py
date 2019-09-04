# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import numpy as np

from fairlearn.metrics import DemographicParity


class GridSearchClassification:
    def __init__(self,
                 learner,
                 fairness_metric=DemographicParity(),
                 number_of_lagrange_multipliers=7):
        self.learner = learner
        if not isinstance(fairness_metric, DemographicParity):
            raise RuntimeError("DemographicParity is only currently supported fairness metric")
        self.fairness_metric = copy.deepcopy(fairness_metric)
        self.number_of_lagrange_multipliers = number_of_lagrange_multipliers

    def fit(self, X, Y, protected_attribute, lagrange_multipliers=None):
        # Verify we have a binary classification problem
        unique_labels = np.unique(Y)
        if not set(unique_labels).issubset({0, 1}):
            raise RuntimeError("Supplied Y labels are not 0 or 1")

        self.fairness_metric.set_binary_protected_attribute(protected_attribute)

        # If not supplied, generate array of trial lagrange multipliers
        if lagrange_multipliers is None:
            lagrange_multipliers = self.fairness_metric.generate_binary_trade_offs(
                self.number_of_lagrange_multipliers)

        self.all_models = []
        for current_multiplier in lagrange_multipliers:
            # Generate weights and relabels
            weights, re_labels = self.fairness_metric.reweight_and_relabel_binary_attribute(
                Y, current_multiplier)

            # Run the learner
            current_learner = copy.deepcopy(self.learner)
            current_learner.fit(X, re_labels, sample_weight=np.absolute(weights))

            # Append the new model, along with its current_multiplier value
            # to the result
            # Note that we call it a model because it is a learner which has
            # had 'fit' called
            self.all_models.append({"model": current_learner,
                                    "lagrange_multiplier": current_multiplier})

        # Designate a 'best' model
        # Selection algorithm not yet fully implemented
        self.best_model = self.all_models[0]

    def predict(self, X):
        return self.best_model["model"].predict(X)

    def predict_proba(self, X):
        return self.best_model["model"].predict_proba(X)

    def posterior_predict(self, X):
        return [r["model"].predict(X) for r in self.all_models]

    def posterior_predict_proba(self, X):
        return [r["model"].predict_proba(X) for r in self.all_models]
