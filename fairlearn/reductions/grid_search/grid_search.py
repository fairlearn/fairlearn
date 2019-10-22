# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import numpy as np
import pandas as pd

from fairlearn.reductions import Reduction
from fairlearn.reductions.grid_search import GridSearchResult
from fairlearn.reductions.moments.moment import Moment, ClassificationMoment
from fairlearn import _KW_SENSITIVE_FEATURES

TRADEOFF_OPTIMIZATION = "tradeoff_optimization"


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
        n_units = (float(grid_size) / (2.0**neg_allowed.sum())
                   ) ** (1.0 / true_dim) - 1
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
            if (index == self.dim - 1) and (self.force_L1_norm):
                if self.neg_allowed[index] and max_val > 0:
                    values = [-max_val, max_val]
                else:
                    values = [max_val]
            else:
                min_val = -max_val if self.neg_allowed[index] else 0
                values = range(min_val, max_val + 1)
            for current_value in values:
                self.entry[index] = current_value
                self.accumulate_integer_grid(
                    index + 1, max_val - abs(current_value))


class GridSearch(Reduction):
    """Learner to perform a grid search given a blackbox estimator algorithm.

    Parameters
    ----------
    estimator : Object
        An object corresponding to the scikit-learn Estimator concept with sample
        weights. That is it implements predict(X, y, sample_weight)

    constraints : Moment
        An object of type Moment which describes the constraint (demographic parity etc.)
        to be applied
    """

    _MESSAGE_Y_NOT_BINARY = "Supplied y labels are not 0 or 1"
    _MESSAGE_X_NONE = "Must supply X"
    _MESSAGE_Y_NONE = "Must supply y"
    _MESSAGE_X_Y_ROWS = "X and y must have same number of rows"
    _MESSAGE_X_SENSITIVE_ROWS = "X and the sensitive features must have same number of rows"

    def __init__(self,
                 estimator,
                 constraints,
                 selection_rule=TRADEOFF_OPTIMIZATION,
                 constraint_weight=0.5,
                 grid_size=10,
                 grid_limit=2.0,
                 grid=None):
        self.estimator = estimator
        if not isinstance(constraints, Moment):
            raise RuntimeError("Unsupported disparity metric")
        self.constraints = constraints

        if (selection_rule == TRADEOFF_OPTIMIZATION):
            if not (0.0 <= constraint_weight <= 1.0):
                raise RuntimeError(
                    "Must specify constraint_weight between 0.0 and 1.0")
        else:
            raise RuntimeError("Unsupported selection rule")
        self.selection_rule = selection_rule
        self.constraint_weight = float(constraint_weight)
        self.objective_weight = 1.0 - constraint_weight

        self.grid_size = grid_size
        self.grid_limit = float(grid_limit)
        self.grid = grid

    def fit(self, X, y, **kwargs):
        """Runs the grid search. This will result in multiple copies of the
        estimator being made, and the `fit` method of each one called.

        Parameters
        ----------
        X : array_like
            The feature data for the machine learning problem

        y : array_like
            The ground truth labels for the machine learning problem

        sensitive_features : array_like
            A (currently) required keyword argument listing the
            feature used by the constraints object
        """
        if X is None:
            raise ValueError(self._MESSAGE_X_NONE)

        if y is None:
            raise ValueError(self._MESSAGE_Y_NONE)

        if _KW_SENSITIVE_FEATURES not in kwargs:
            raise RuntimeError(
                "Must specify {0} (for now)".format(_KW_SENSITIVE_FEATURES))

        # Extract the target attribute
        sensitive = self._make_vector(
            kwargs[_KW_SENSITIVE_FEATURES], _KW_SENSITIVE_FEATURES)

        unique_labels = np.unique(sensitive)
        if len(unique_labels) > 2:
            raise RuntimeError("Sensitive features contain "
                               "more than two unique values")

        # Extract the Y values
        y_vector = self._make_vector(y, "y")

        X_rows, _ = self._get_matrix_shape(X, "X")
        if X_rows != y_vector.shape[0]:
            raise RuntimeError(self._MESSAGE_X_Y_ROWS)
        if X_rows != sensitive.shape[0]:
            raise RuntimeError(self._MESSAGE_X_SENSITIVE_ROWS)

        if isinstance(self.constraints, ClassificationMoment):
            # We have a classification problem
            # Need to make sure that y is binary (for now)
            unique_labels = np.unique(y_vector)
            if not set(unique_labels).issubset({0, 1}):
                raise RuntimeError(self._MESSAGE_Y_NOT_BINARY)

        # Prep the disparity metric and objective
        self.constraints.load_data(X, y_vector, **kwargs)
        objective = self.constraints.default_objective()
        objective.load_data(X, y_vector, **kwargs)
        is_classification_reduction = isinstance(
            self.constraints, ClassificationMoment)

        # Basis information
        pos_basis = self.constraints.pos_basis
        neg_basis = self.constraints.neg_basis
        neg_allowed = self.constraints.neg_basis_present
        objective_in_the_span = (
            self.constraints.default_objective_lambda_vec is not None)

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
            weights = self.constraints.signed_weights(lambda_vec)
            if not objective_in_the_span:
                weights = weights + objective.signed_weights()
            if is_classification_reduction:
                y_reduction = 1 * (weights > 0)
                weights = weights.abs()
            else:
                y_reduction = y_vector

            current_estimator = copy.deepcopy(self.estimator)
            current_estimator.fit(X, y_reduction, sample_weight=weights)
            def predict_fct(X): return current_estimator.predict(X)

            nxt = GridSearchResult(current_estimator,
                                   lambda_vec,
                                   objective.gamma(predict_fct)[0],
                                   self.constraints.gamma(predict_fct))
            self.all_results.append(nxt)

        if self.selection_rule == TRADEOFF_OPTIMIZATION:
            def loss_fct(x):
                return self.objective_weight * x.objective + self.constraint_weight * x.gamma.max()
            self.best_result = min(self.all_results, key=loss_fct)
        else:
            raise RuntimeError("Unsupported selection rule")

        return

    def predict(self, X):
        """Provides a prediction for the given input data based
        on the best model found by the grid search.

        Parameters
        ----------
        X : array_like
            The data for which predictions are required

        Returns
        -------
        out : scalar or array_like
            The prediction. If X represents the data for a single example
            the result will be a scalar. Otherwise the result will be an
            array
        """
        return self.best_result.predictor.predict(X)

    def predict_proba(self, X):
        return self.best_result.predictor.predict_proba(X)

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
