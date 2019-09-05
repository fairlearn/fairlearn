# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy

import numpy as np

from fairlearn.metrics import BoundedGroupLoss
from fairlearn.reductions.grid_search.simple_quality_metrics import SimpleRegressionQualityMetric
from . import utilities


class GridSearchRegression:
    def __init__(self,
                 learner,
                 fairness_metric=BoundedGroupLoss(),
                 quality_metric=SimpleRegressionQualityMetric(),
                 number_of_tradeoffs=7):
        self.learner = learner
        if not isinstance(fairness_metric, BoundedGroupLoss):
            raise RuntimeError("BoundedGroupLoss is the only currently supported fairness metric")
        self.fairness_metric = fairness_metric
        self.number_of_tradeoffs = number_of_tradeoffs

        self.quality_metric = quality_metric

    def fit(self, X, Y, protected_attribute, tradeoffs=None):
        # Extract required statistics from protected_attribute
        p0, p1, a0_val = utilities.generate_protected_attribute_info(protected_attribute)

        if tradeoffs is None:
            tradeoffs = np.linspace(0, 1, self.number_of_tradeoffs)

        self.all_models = []
        for tradeoff in tradeoffs:
            weight_func = np.vectorize(self._regression_weight_function)
            weights = weight_func(protected_attribute,
                                  tradeoff,
                                  p0, p1, a0_val)

            current_learner = copy.deepcopy(self.learner)
            current_learner.fit(X, Y, sample_weight=weights)

            self.all_models.append({"model": current_learner,
                                    "tradeoff": tradeoff})

        # Designate a 'best' model
        self.best_model = self._select_best_model(X, Y, protected_attribute, self.all_models)

    def predict(self, X):
        return self.best_model["model"].predict(X)

    def posterior_predict(self, X):
        return [r["model"].predict(X) for r in self.all_models]

    def _regression_weight_function(self, a_val, trade_off, p0, p1, a0_val):
        if a_val == a0_val:
            return trade_off / p0
        else:
            return (1 - trade_off) / p1

    def _select_best_model(self, X, Y, protected_attribute, model_list):
        self.quality_metric.set_data(X, Y, protected_attribute)

        for m in model_list:
            m["quality"] = self.quality_metric.get_quality(m["model"])

        best_model = max(model_list, key=lambda x: x["quality"])
        return best_model
