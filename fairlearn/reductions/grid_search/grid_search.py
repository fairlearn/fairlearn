# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import numpy as np
import pandas as pd

from fairlearn.reductions.moments import DemographicParity, BoundedGroupLoss
from fairlearn.reductions.reductions_estimator import ReductionsEstimator
from fairlearn.reductions.grid_search import QualityMetric, GridSearchResult


class GridSearch(ReductionsEstimator):
    """Learner to perform a grid search given a blackbox algorithm.
    The supplied algorithm must implement a method
    fit(X, y, sample_weight=[...])
    At the present time, the only disparity metrics supported
    are demographic parity (for classification) and bounded group
    loss (for regression)
    """

    _KW_LAGRANGE_MULTIPLIERS = "lagrange_multipliers"
    _KW_NUMBER_LAGRANGE_MULTIPLIERS = "number_of_lagrange_multipliers"

    _MESSAGE_Y_NOT_BINARY = "Supplied y labels are not 0 or 1"
    _MESSAGE_X_NONE = "Must supply X"
    _MESSAGE_Y_NONE = "Must supply y"
    _MESSAGE_X_Y_ROWS = "X and y must have same number of rows"
    _MESSAGE_X_A_ROWS = "X and the target attribute must have same number of rows"

    def __init__(self,
                 learner,
                 constraint,
                 quality_metric):
        self.learner = learner
        if (not isinstance(constraint, DemographicParity) and
                not isinstance(constraint, BoundedGroupLoss)):
            raise RuntimeError("Unsupported constraint")
        self.constraint = constraint

        if not isinstance(quality_metric, QualityMetric):
            raise RuntimeError("quality_metric must derive from QualityMetric")
        self.quality_metric = quality_metric

    def fit(self, X, y, aux_data=None, **kwargs):
        if X is None:
            raise ValueError(self._MESSAGE_X_NONE)

        if y is None:
            raise ValueError(self._MESSAGE_Y_NONE)

        lagrange_multipliers = None
        if self._KW_LAGRANGE_MULTIPLIERS in kwargs:
            lagrange_multipliers = kwargs[self._KW_LAGRANGE_MULTIPLIERS]

        number_of_lagrange_multipliers = None
        if self._KW_NUMBER_LAGRANGE_MULTIPLIERS in kwargs:
            number_of_lagrange_multipliers = kwargs[self._KW_NUMBER_LAGRANGE_MULTIPLIERS]

        # Extract the target attribute
        A = self._make_vector(self.constraint.extract_A(X, aux_data), "A")

        # Extract the Y values
        y_vector = self._make_vector(y, "y")

        X_rows, _ = self._get_matrix_shape(X, "X")
        if X_rows != y_vector.shape[0]:
            raise RuntimeError(self._MESSAGE_X_Y_ROWS)
        if X_rows != A.shape[0]:
            raise RuntimeError(self._MESSAGE_X_A_ROWS)

        # Prep the quality metric
        self.quality_metric.set_data(X, y_vector, A)

        # We do not yet have disparity metrics fully implemented
        # For now, we assume that if we are passed a DemographicParity
        # object we have a binary classification problem whereas
        # BoundedGroupLoss indicates a regression
        if isinstance(self.constraint, DemographicParity):
            self._fit_classification(X, y_vector, A,
                                     lagrange_multipliers, number_of_lagrange_multipliers)
        elif isinstance(self.constraint, BoundedGroupLoss):
            self._fit_regression(X, y_vector, A,
                                 lagrange_multipliers, number_of_lagrange_multipliers)
        else:
            raise RuntimeError("Can't get here")

    def _fit_classification(self, X, y, target_attribute,
                            lagrange_multipliers, number_of_lagrange_multipliers):
        # Verify we have a binary classification problem
        unique_labels = np.unique(y)
        if not set(unique_labels).issubset({0, 1}):
            raise RuntimeError(self._MESSAGE_Y_NOT_BINARY)

        # Extract required statistics from target_attribute
        p0, p1, a0_val = self._generate_target_attribute_info(target_attribute)

        # If not supplied, generate array of trial lagrange multipliers
        if lagrange_multipliers is None:
            limit = 1
            if p1 > 0 and p0 / p1 > 1:
                limit = p0 / p1
            lagrange_multipliers = np.linspace(-2 * limit,
                                               2 * limit,
                                               number_of_lagrange_multipliers)

        self.all_results = []
        for current_multiplier in lagrange_multipliers:
            # Generate weights array
            sample_weights = self._generate_classification_weights(y,
                                                                   target_attribute,
                                                                   current_multiplier,
                                                                   p1 / p0,
                                                                   a0_val)

            # Generate y'
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
            self.all_results.append(nxt)

        # Designate a 'best' model
        self.best_result = max(self.all_results, key=lambda x: x.quality_metric_value)

    def _fit_regression(self, X, y, target_attribute, tradeoffs, number_of_tradeoffs):
        # Extract required statistics from target_attribute
        p0, p1, a0_val = self._generate_target_attribute_info(target_attribute)

        if tradeoffs is None:
            tradeoffs = np.linspace(0, 1, number_of_tradeoffs)

        self.all_results = []
        for tradeoff in tradeoffs:
            weight_func = np.vectorize(self._regression_weight_function)
            weights = weight_func(target_attribute,
                                  tradeoff,
                                  p0, p1, a0_val)

            current_learner = copy.deepcopy(self.learner)
            current_learner.fit(X, y, sample_weight=weights)

            # Evaluate the quality metric
            quality = self.quality_metric.get_quality(current_learner)

            nxt = GridSearchResult(current_learner, tradeoff, quality)
            self.all_results.append(nxt)

        # Designate a 'best' model
        self.best_result = max(self.all_results, key=lambda x: x.quality_metric_value)

    def predict(self, X):
        return self.best_result.model.predict(X)

    def predict_proba(self, X):
        return self.best_result.model.predict_proba(X)

    def posterior_predict(self, X):
        return [r.model.predict(X) for r in self.all_results]

    def posterior_predict_proba(self, X):
        return [r.model.predict_proba(X) for r in self.all_results]

    def _classification_weight_function(self, y_val, a_val, L, p_ratio, a0_val):
        # Used by the classification side of GridSearch to generate a sample
        # set of weights, following demographic parity
        # Weights vary with the current value of the Lagrange multiplier
        if a_val == a0_val:
            return 2 * y_val - 1 - L * p_ratio
        else:
            return 2 * y_val - 1 + L

    def _generate_target_attribute_info(self, target_attribute):
        unique_labels, counts = np.unique(
            target_attribute, return_counts=True)
        if len(unique_labels) > 2:
            raise RuntimeError("Target Attribute contains "
                               "more than two unique values")

        p0 = counts[0] / len(target_attribute)
        p1 = 1 - p0

        return p0, p1, unique_labels[0]

    def _generate_classification_weights(self, y, target_attribute, L, p_ratio, a0_val):
        weight_func = np.vectorize(self._classification_weight_function)
        return weight_func(y, target_attribute, L, p_ratio, a0_val)

    def _regression_weight_function(self, a_val, trade_off, p0, p1, a0_val):
        # Reweighting function for Bounded Group Loss for regression
        # Note that it uses a trade_off parameter which varies between 0 and 1
        if a_val == a0_val:
            return trade_off / p0
        else:
            return (1 - trade_off) / p1

    def _make_vector(self, formless, formless_name):
        formed_vector = None
        if isinstance(formless, list):
            formed_vector = np.array(formless)
        elif isinstance(formless, pd.DataFrame):
            if len(formless.columns) == 1:
                formed_vector = formless[0].to_numpy()
            else:
                msgfmt = "{0} is a DataFrame with more than one column"
                raise RuntimeError(msgfmt.format(formless_name))
        elif isinstance(formless, pd.Series):
            formed_vector = formless.to_numpy()
        elif isinstance(formless, np.ndarray):
            if len(formless.shape) == 1:
                formed_vector = formless
            elif len(formless.shape) == 2 and formless.shape[1] == 1:
                formed_vector = formless[:, 0]
            else:
                msgfmt = "{0} is an ndarray with more than one column"
                raise RuntimeError(msgfmt.format(formless_name))
        else:
            msgfmt = "{0} not an ndarray or DataFrame"
            raise RuntimeError(msgfmt.format(formless_name))

        return formed_vector

    def _get_matrix_shape(self, formless, formless_name):
        num_rows = -1
        num_cols = -1

        if isinstance(formless, pd.DataFrame):
            num_cols = len(formless.columns)
            num_rows = len(formless.index)
        elif isinstance(formless, np.ndarray):
            if len(formless.shape) == 2:
                num_rows = formless.shape[0]
                num_cols = formless.shape[1]
            else:
                msgfmt = "{0} is an ndarray which is not 2D"
                raise RuntimeError(msgfmt.format(formless_name))
        else:
            msgfmt = "{0} not an ndarray or DataFrame"
            raise RuntimeError(msgfmt.format(formless_name))
        return num_rows, num_cols
