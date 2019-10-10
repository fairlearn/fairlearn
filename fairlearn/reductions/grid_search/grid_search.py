# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import numpy as np
import pandas as pd

from fairlearn.reductions.reductions_estimator import ReductionsEstimator
from fairlearn.reductions.grid_search import QualityMetric, GridSearchResult
from fairlearn.reductions.moments.moment import Moment, ClassificationMoment, _KW_GROUP_MEMBERSHIP


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

    _MESSAGE_Y_NOT_BINARY = "Supplied y labels are not 0 or 1"
    _MESSAGE_X_NONE = "Must supply X"
    _MESSAGE_Y_NONE = "Must supply y"
    _MESSAGE_X_Y_ROWS = "X and y must have same number of rows"
    _MESSAGE_X_A_ROWS = "X and the target attribute must have same number of rows"

    def __init__(self,
                 learner,
                 disparity_metric,
                 quality_metric,
                 grid_size=10,
                 grid_limit=2.0,
                 grid=None):
        self.learner = learner
        if not isinstance(disparity_metric, Moment):
            raise RuntimeError("Unsupported disparity metric")
        self.disparity_metric = disparity_metric

        if not isinstance(quality_metric, QualityMetric):
            raise RuntimeError("quality_metric must derive from QualityMetric")
        self.quality_metric = quality_metric

        self.grid_size = grid_size
        self.grid_limit = float(grid_limit)
        self.grid = grid

    def fit(self, X, y, **kwargs):
        if X is None:
            raise ValueError(self._MESSAGE_X_NONE)

        if y is None:
            raise ValueError(self._MESSAGE_Y_NONE)

        if _KW_GROUP_MEMBERSHIP not in kwargs:
            raise RuntimeError("Must specify {0} (for now)".format(_KW_GROUP_MEMBERSHIP))

        # Extract the target attribute
        A = self._make_vector(kwargs[_KW_GROUP_MEMBERSHIP], _KW_GROUP_MEMBERSHIP)

        unique_labels = np.unique(A)
        if len(unique_labels) > 2:
            raise RuntimeError("Target Attribute contains "
                               "more than two unique values")

        # Extract the Y values
        y_vector = self._make_vector(y, "y")

        X_rows, _ = self._get_matrix_shape(X, "X")
        if X_rows != y_vector.shape[0]:
            raise RuntimeError(self._MESSAGE_X_Y_ROWS)
        if X_rows != A.shape[0]:
            raise RuntimeError(self._MESSAGE_X_A_ROWS)

        # Prep the quality metric
        self.quality_metric.set_data(X, y_vector, A)

        if isinstance(self.disparity_metric, ClassificationMoment):
            # We have a classification problem
            # Need to make sure that y is binary (for now)
            unique_labels = np.unique(y_vector)
            if not set(unique_labels).issubset({0, 1}):
                raise RuntimeError(self._MESSAGE_Y_NOT_BINARY)

        # Prep the disparity metric and objective
        self.disparity_metric.load_data(X, y_vector, **kwargs)    # sensitive_features=A)
        objective = self.disparity_metric.default_objective()
        objective.load_data(X, y_vector, **kwargs)    # sensitive_features=A)
        is_classification_reduction = isinstance(self.disparity_metric, ClassificationMoment)

        # Basis information
        pos_basis = self.disparity_metric.pos_basis
        neg_basis = self.disparity_metric.neg_basis
        neg_allowed = self.disparity_metric.neg_basis_present
        objective_in_the_span = (self.disparity_metric.default_objective_lambda_vec is not None)

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
                y_reduction = y_vector

            current_learner = copy.deepcopy(self.learner)
            current_learner.fit(X, y_reduction, sample_weight=weights)

            # Evaluate the quality metric
            quality = self.quality_metric.get_quality(current_learner)

            nxt = GridSearchResult(current_learner, lambda_vec, quality)
            self.all_results.append(nxt)

        # Designate a 'best' model
        self.best_result = max(self.all_results, key=lambda x: x.quality_metric_value)
        return

    def predict(self, X):
        return self.best_result.model.predict(X)

    def predict_proba(self, X):
        return self.best_result.model.predict_proba(X)

    def posterior_predict(self, X):
        return [r.model.predict(X) for r in self.all_results]

    def posterior_predict_proba(self, X):
        return [r.model.predict_proba(X) for r in self.all_results]

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
