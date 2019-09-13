# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import numpy as np

from fairlearn.metrics import DemographicParity, BoundedGroupLoss
from fairlearn.reductions.reductions_learner import ReductionsLearner
from fairlearn.reductions.grid_search import QualityMetric, GridSearchResult


class GridSearch(ReductionsLearner):
    """Learner to perform a grid search given a blackbox algorithm.
    The supplied algorithm must implement a method
    fit(X, Y, sample_weight=[...])
    At the present time, the only disparity metrics supported
    are demographic parity (for classification) and bounded group
    loss (for regression)
    """

    _KW_PROTECTED_ATTRIBUTE = "protected_attribute"
    _KW_LAGRANGE_MULTIPLIERS = "lagrange_multipliers"
    _KW_NUMBER_LAGRANGE_MULTIPLIERS = "number_of_lagrange_multipliers"

    _MESSAGE_Y_NOT_BINARY = "Supplied Y labels are not 0 or 1"

    def __init__(self,
                 learner,
                 disparity_metric,
                 quality_metric):
        self.learner = learner
        if (not isinstance(disparity_metric, DemographicParity) and
                not isinstance(disparity_metric, BoundedGroupLoss)):
            raise RuntimeError("Unsupported disparity metric")
        self.disparity_metric = disparity_metric

        if not isinstance(quality_metric, QualityMetric):
            raise RuntimeError("quality_metric must derive from QualityMetric")
        self.quality_metric = quality_metric

    def fit(self, X, Y, **kwargs):
        if self._KW_PROTECTED_ATTRIBUTE in kwargs:
            protected_attribute = kwargs[self._KW_PROTECTED_ATTRIBUTE]
        else:
            raise RuntimeError("Must specify protected_attribute (for now)")

        lagrange_multipliers = None
        if self._KW_LAGRANGE_MULTIPLIERS in kwargs:
            lagrange_multipliers = kwargs[self._KW_LAGRANGE_MULTIPLIERS]

        number_of_lagrange_multipliers = None
        if self._KW_NUMBER_LAGRANGE_MULTIPLIERS in kwargs:
            number_of_lagrange_multipliers = kwargs[self._KW_NUMBER_LAGRANGE_MULTIPLIERS]

        # Prep the quality metric
        self.quality_metric.set_data(X, Y, protected_attribute)

        # We do not yet have disparity metrics fully implemented
        # For now, we assume that if we are passed a DemographicParity
        # object we have a binary classification problem whereas
        # BoundedGroupLoss indicates a regression
        if isinstance(self.disparity_metric, DemographicParity):
            self._fit_classification(X, Y, protected_attribute,
                                     lagrange_multipliers, number_of_lagrange_multipliers)
        elif isinstance(self.disparity_metric, BoundedGroupLoss):
            self._fit_regression(X, Y, protected_attribute,
                                 lagrange_multipliers, number_of_lagrange_multipliers)
        else:
            raise RuntimeError("Can't get here")

    def _fit_classification(self, X, Y, protected_attribute,
                            lagrange_multipliers, number_of_lagrange_multipliers):
        # Verify we have a binary classification problem
        unique_labels = np.unique(Y)
        if not set(unique_labels).issubset({0, 1}):
            raise RuntimeError(self._MESSAGE_Y_NOT_BINARY)

        # Extract required statistics from protected_attribute
        p0, p1, a0_val = self._generate_protected_attribute_info(protected_attribute)

        # If not supplied, generate array of trial lagrange multipliers
        if lagrange_multipliers is None:
            limit = 1
            if p1 > 0 and p0 / p1 > 1:
                limit = p0 / p1
            lagrange_multipliers = np.linspace(-2 * limit,
                                               2 * limit,
                                               number_of_lagrange_multipliers)

        self.all_models = []
        for current_multiplier in lagrange_multipliers:
            # Generate weights array
            sample_weights = self._generate_classification_weights(Y,
                                                                   protected_attribute,
                                                                   current_multiplier,
                                                                   p1 / p0,
                                                                   a0_val)

            # Generate Y'
            def f(x): return 1 if x > 0 else 0
            re_labels = np.vectorize(f)(sample_weights)

            # Run the learner
            current_learner = copy.deepcopy(self.learner)
            current_learner.fit(X, re_labels, sample_weight=np.absolute(sample_weights))

            # Evaluate the quality metric
            quality = self.quality_metric.get_quality(current_learner)

            # Append the new model, along with its current_multiplier value
            # to the result
            # Note that we call it a model because it is a learner which has
            # had 'fit' called
            nxt = GridSearchResult(current_learner, current_multiplier, quality)
            self.all_models.append(nxt)

        # Designate a 'best' model
        self.best_model = max(self.all_models, key=lambda x: x.quality_metric_value)

    def _fit_regression(self, X, Y, protected_attribute, tradeoffs, number_of_tradeoffs):
        # Extract required statistics from protected_attribute
        p0, p1, a0_val = self._generate_protected_attribute_info(protected_attribute)

        if tradeoffs is None:
            tradeoffs = np.linspace(0, 1, number_of_tradeoffs)

        self.all_models = []
        for tradeoff in tradeoffs:
            weight_func = np.vectorize(self._regression_weight_function)
            weights = weight_func(protected_attribute,
                                  tradeoff,
                                  p0, p1, a0_val)

            current_learner = copy.deepcopy(self.learner)
            current_learner.fit(X, Y, sample_weight=weights)

            # Evaluate the quality metric
            quality = self.quality_metric.get_quality(current_learner)

            nxt = GridSearchResult(current_learner, tradeoff, quality)
            self.all_models.append(nxt)

        # Designate a 'best' model
        self.best_model = max(self.all_models, key=lambda x: x.quality_metric_value)

    def predict(self, X):
        return self.best_model.model.predict(X)

    def predict_proba(self, X):
        return self.best_model.model.predict_proba(X)

    def posterior_predict(self, X):
        return [r.model.predict(X) for r in self.all_models]

    def posterior_predict_proba(self, X):
        return [r.model.predict_proba(X) for r in self.all_models]

    def _classification_weight_function(self, y_val, a_val, L, p_ratio, a0_val):
        # Used by the classification side of GridSearch to generate a sample
        # set of weights, following demographic parity
        # Weights vary with the current value of the Lagrange multiplier
        if a_val == a0_val:
            return 2 * y_val - 1 - L * p_ratio
        else:
            return 2 * y_val - 1 + L

    def _generate_protected_attribute_info(self, protected_attribute):
        unique_labels, counts = np.unique(
            protected_attribute, return_counts=True)
        if len(unique_labels) > 2:
            raise RuntimeError("Protected Attribute contains "
                               "more than two unique values")

        p0 = counts[0] / len(protected_attribute)
        p1 = 1 - p0

        return p0, p1, unique_labels[0]

    def _generate_classification_weights(self, y, protected_attribute, L, p_ratio, a0_val):
        weight_func = np.vectorize(self._classification_weight_function)
        return weight_func(y, protected_attribute, L, p_ratio, a0_val)

    def _regression_weight_function(self, a_val, trade_off, p0, p1, a0_val):
        # Reweighting function for Bounded Group Loss for regression
        # Note that it uses a trade_off parameter which varies between 0 and 1
        if a_val == a0_val:
            return trade_off / p0
        else:
            return (1 - trade_off) / p1
