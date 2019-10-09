# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import numpy as np
import pandas as pd

from fairlearn.metrics import DemographicParity, BoundedGroupLoss
from fairlearn.reductions.reductions_estimator import ReductionsEstimator
from fairlearn.reductions.grid_search import QualityMetric, GridSearchResult
from fairlearn.reductions.moments.moment import Moment, _REDUCTION_TYPE_CLASSIFICATION


class _GridGenerator:
    """A generator of a grid of points with a bounded L1 norm.
    """

    def __init__(self, grid_size, grid_limit, pos_basis, neg_basis, neg_allowed, force_L1_norm):
        # grid parameters
        self.dim = len(pos_basis.columns)
        self.neg_allowed = neg_allowed
        self.force_L1_norm = force_L1_norm

        # true dimensionality of the grid
        if self.force_L1_norm:
            true_dim = self.dim - 1
        else:
            true_dim = self.dim

        # a conservative lower bound on the scaling parameter of the grid
        n_units = (float(grid_size) / (2.0**neg_allowed.sum())) ** (1.0/true_dim) - 1
        n_units = int(np.floor(n_units))
        if n_units < 0:
            n_units = 0

        # find the grid of size at least "size" and save the first "size" entries
        while True:
            int_grid = self.build_integer_grid(n_units)
            if len(int_grid) >= grid_size:
                # re-scale the integer grid, separate into positive and negative parts
                pos_coefs = pd.DataFrame(self.accumulator[:grid_size]).T * (float(grid_limit) / n_units)   # noqa: E501
                neg_coefs = -pos_coefs.copy()
                pos_coefs[pos_coefs < 0] = 0.0
                neg_coefs[neg_coefs < 0] = 0.0
                # convert the grid of basis coefficients into a grid of lambda vectors
                self.grid = pos_basis.dot(pos_coefs) + neg_basis.dot(neg_coefs)
                break
            n_units = n_units + 1

    def build_integer_grid(self, n_units):
        # initialize working variables for the grid accumulation
        self.entry = np.zeros(self.dim)
        self.accumulator = []
        # recursively create the integer grid
        self.accumulate_integer_grid(0, n_units)
        return self.accumulator

    def accumulate_integer_grid(self, index, max_val):
        if index == self.dim:
            self.accumulator.append(self.entry.copy())
        else:
            if (index == self.dim-1) and (self.force_L1_norm):
                if self.neg_allowed[index] and max_val > 0:
                    values = [-max_val, max_val]
                else:
                    values = [max_val]
            else:
                min_val = -max_val if self.neg_allowed[index] else 0
                values = range(min_val, max_val+1)
            for current_value in values:
                self.entry[index] = current_value
                self.accumulate_integer_grid(index+1, max_val-abs(current_value))


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

    _FLIP_ATTRIBUTE_VALS = False

    def __init__(self,
                 learner,
                 disparity_metric,
                 quality_metric,
                 grid_size=10,
                 grid_limit=2.0,
                 grid=None):
        self.learner = learner
        if (not isinstance(disparity_metric, DemographicParity) and
                not isinstance(disparity_metric, BoundedGroupLoss) and
                not isinstance(disparity_metric, Moment)):
            raise RuntimeError("Unsupported disparity metric")
        self.disparity_metric = disparity_metric

        if not isinstance(quality_metric, QualityMetric):
            raise RuntimeError("quality_metric must derive from QualityMetric")
        self.quality_metric = quality_metric

        self.grid_size = grid_size
        self.grid_limit = float(grid_limit)
        self.grid = grid

    def fit(self, X, y, aux_data=None, **kwargs):
        if X is None:
            raise ValueError(self._MESSAGE_X_NONE)

        if y is None:
            raise ValueError(self._MESSAGE_Y_NONE)

        if aux_data is None:
            raise RuntimeError("Must specify aux_data (for now)")

        lagrange_multipliers = None
        if self._KW_LAGRANGE_MULTIPLIERS in kwargs:
            lagrange_multipliers = kwargs[self._KW_LAGRANGE_MULTIPLIERS]

        number_of_lagrange_multipliers = None
        if self._KW_NUMBER_LAGRANGE_MULTIPLIERS in kwargs:
            number_of_lagrange_multipliers = kwargs[self._KW_NUMBER_LAGRANGE_MULTIPLIERS]

        # Extract the target attribute
        A = self._make_vector(aux_data, "aux_data")

        # Extract the Y values
        y_vector = self._make_vector(y, "y")

        X_rows, _ = self._get_matrix_shape(X, "X")
        if X_rows != y_vector.shape[0]:
            raise RuntimeError(self._MESSAGE_X_Y_ROWS)
        if X_rows != A.shape[0]:
            raise RuntimeError(self._MESSAGE_X_A_ROWS)

        # Prep the quality metric
        self.quality_metric.set_data(X, y_vector, A)

        if isinstance(self.disparity_metric, Moment):
            # Prep the disparity metric and objective
            self.disparity_metric.init(X, A, y_vector)
            objective = self.disparity_metric.default_objective()
            objective.init(X, A, y_vector)
            is_classification_reduction = (self.disparity_metric.reduction_type == _REDUCTION_TYPE_CLASSIFICATION)  # noqa: E501

            # Basis information
            pos_basis = self.disparity_metric.pos_basis
            neg_basis = self.disparity_metric.neg_basis
            neg_allowed = self.disparity_metric.neg_basis_present
            objective_in_the_span = (self.disparity_metric.default_objective_lambda_vec is not None)   # noqa: E501

            if self.grid is None:
                grid = _GridGenerator(self.grid_size,
                                      self.grid_limit,
                                      pos_basis,
                                      neg_basis,
                                      neg_allowed,
                                      objective_in_the_span).grid
            else:
                grid = self.grid

            # Fit the estimates
            self.all_results = []
            for i in grid.columns:
                lambda_vec = grid[i]
                weights = self.disparity_metric.signed_weights(lambda_vec)
                if not objective_in_the_span:
                    weights = weights + objective.signed_weights()
                if is_classification_reduction:
                    y_reduction = 1 * (weights > 0)
                    weights = weights.abs()
                else:
                    y_reduction = y

                current_learner = copy.deepcopy(self.learner)
                current_learner.fit(X, y_reduction, sample_weight=weights)

                # Evaluate the quality metric
                quality = self.quality_metric.get_quality(current_learner)

                nxt = GridSearchResult(current_learner, lambda_vec, quality)
                self.all_results.append(nxt)

            # Designate a 'best' model
            self.best_result = max(self.all_results, key=lambda x: x.quality_metric_value)
            return

        # We do not yet have disparity metrics fully implemented
        # For now, we assume that if we are passed a DemographicParity
        # object we have a binary classification problem whereas
        # BoundedGroupLoss indicates a regression
        if isinstance(self.disparity_metric, DemographicParity):
            self._fit_classification(X, y_vector, A,
                                     lagrange_multipliers, number_of_lagrange_multipliers)
        elif isinstance(self.disparity_metric, BoundedGroupLoss):
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

        if self._FLIP_ATTRIBUTE_VALS:
            return p1, p0, unique_labels[1]
        else:
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
