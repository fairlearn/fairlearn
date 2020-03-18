# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import logging
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from time import time

from fairlearn._input_validation import _validate_and_reformat_input, _KW_SENSITIVE_FEATURES
from fairlearn import _NO_PREDICT_BEFORE_FIT
from fairlearn.reductions._moments import Moment, ClassificationMoment
from .grid_search_result import GridSearchResult

logger = logging.getLogger(__name__)

TRADEOFF_OPTIMIZATION = "tradeoff_optimization"


class _GridGenerator:
    """A generator of a grid of points with a bounded L1 norm."""

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
        n_units = (float(grid_size) / (2.0**neg_allowed.sum())) ** (1.0 / true_dim) - 1   # noqa: E501
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
                self.accumulate_integer_grid(index + 1, max_val - abs(current_value))   # noqa: E501


class GridSearch(BaseEstimator, MetaEstimatorMixin):
    """Estimator to perform a grid search given a blackbox estimator algorithm.

    The approach used is taken from section 3.4 of
    `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`_.

    :param estimator: An estimator implementing methods :code:`fit(X, y, sample_weight)` and
        :code:`predict(X)`, where `X` is the matrix of features, `y` is the vector of labels, and
        `sample_weight` is a vector of weights; labels `y` and predictions returned by
        :code:`predict(X)` are either 0 or 1.
    :type estimator: estimator

    :param constraints: The disparity constraints expressed as moments.
    :type constraints: fairlearn.reductions.Moment

    :param selection_rule: Specifies the procedure for selecting the best model found by the
        grid search. At the present time, the only valid value is "tradeoff_optimization" which
        minimises a weighted sum of the error rate and constraint violation.
    :type selection_rule: str

    :param constraint_weight: When the `selection_rule` is "tradeoff_optimization" this specifies
        the relative weight put on the constraint violation when selecting the best model.
        The weight placed on the error rate will be :code:`1-constraint_weight`
    :type constraint_weight: float

    :param grid_size: The number of Lagrange multipliers to generate in the grid
    :type grid_size: int

    :param grid_limit: The largest Lagrange multiplier to generate. The grid will contain values
        distributed between :code:`-grid_limit` and :code:`grid_limit` by default
    :type grid_limit: float

    :param grid: Instead of supplying a size and limit for the grid, users may specify the exact
        set of Lagrange multipliers they desire using this argument.
    """

    def __init__(self,
                 estimator,
                 constraints,
                 selection_rule=TRADEOFF_OPTIMIZATION,
                 constraint_weight=0.5,
                 grid_size=10,
                 grid_limit=2.0,
                 grid=None):
        """Construct a GridSearch object."""
        self.estimator = estimator
        if not isinstance(constraints, Moment):
            raise RuntimeError("Unsupported disparity metric")
        self.constraints = constraints

        if (selection_rule == TRADEOFF_OPTIMIZATION):
            if not (0.0 <= constraint_weight <= 1.0):
                raise RuntimeError("Must specify constraint_weight between 0.0 and 1.0")   # noqa: E501
        else:
            raise RuntimeError("Unsupported selection rule")
        self.selection_rule = selection_rule
        self.constraint_weight = float(constraint_weight)
        self.objective_weight = 1.0 - constraint_weight

        self.grid_size = grid_size
        self.grid_limit = float(grid_limit)
        self.grid = grid

        self._all_results = []
        self._best_result = None

    @property
    def all_results(self):
        """Return a list of :class:`GridSearchResult` from each point in the grid."""
        return self._all_results

    @property
    def best_result(self):
        """Return the best result found from the grid search.

        The predictor contained in this instance of
        :class:`GridSearchResult` is used in calls to
        :code:`predict` and :code:`predict_proba`.
        """
        return self._best_result

    def fit(self, X, y, **kwargs):
        """Run the grid search.

        This will result in multiple copies of the
        estimator being made, and the :code:`fit(X)` method
        of each one called.

        :param X: The feature matrix
        :type X: numpy.ndarray or pandas.DataFrame

        :param y: The label vector
        :type y: numpy.ndarray, pandas.DataFrame, pandas.Series, or list

        :param sensitive_features: A (currently) required keyword argument listing the
            feature used by the constraints object
        :type sensitive_features: numpy.ndarray, pandas.DataFrame, pandas.Series, or list (for now)
        """
        if isinstance(self.constraints, ClassificationMoment):
            logger.debug("Classification problem detected")
            is_classification_reduction = True
        else:
            logger.debug("Regression problem detected")
            is_classification_reduction = False

        _, y_train, sensitive_features_train = _validate_and_reformat_input(
            X, y, enforce_binary_sensitive_feature=True,
            enforce_binary_labels=is_classification_reduction, **kwargs)

        kwargs[_KW_SENSITIVE_FEATURES] = sensitive_features_train

        # Prep the parity constraints and objective
        logger.debug("Preparing constraints and objective")
        self.constraints.load_data(X, y_train, **kwargs)
        objective = self.constraints.default_objective()
        objective.load_data(X, y_train, **kwargs)

        # Basis information
        pos_basis = self.constraints.pos_basis
        neg_basis = self.constraints.neg_basis
        neg_allowed = self.constraints.neg_basis_present
        objective_in_the_span = (self.constraints.default_objective_lambda_vec is not None)

        if self.grid is None:
            logger.debug("Creating grid of size %i", self.grid_size)
            grid = _GridGenerator(self.grid_size,
                                  self.grid_limit,
                                  pos_basis,
                                  neg_basis,
                                  neg_allowed,
                                  objective_in_the_span).grid
        else:
            logger.debug("Using supplied grid")
            grid = self.grid

        # Fit the estimates
        logger.debug("Setup complete. Starting grid search")
        self._all_results = []
        for i in grid.columns:
            lambda_vec = grid[i]
            logger.debug("Obtaining weights")
            weights = self.constraints.signed_weights(lambda_vec)
            if not objective_in_the_span:
                weights = weights + objective.signed_weights()

            if is_classification_reduction:
                logger.debug("Applying relabelling for classification problem")
                y_reduction = 1 * (weights > 0)
                weights = weights.abs()
            else:
                y_reduction = y_train

            current_estimator = copy.deepcopy(self.estimator)
            logger.debug("Calling underlying estimator")
            oracle_call_start_time = time()
            current_estimator.fit(X, y_reduction, sample_weight=weights)
            oracle_call_execution_time = time() - oracle_call_start_time
            logger.debug("Call to underlying estimator complete")

            def predict_fct(X): return current_estimator.predict(X)
            nxt = GridSearchResult(current_estimator,
                                   lambda_vec,
                                   objective.gamma(predict_fct)[0],
                                   self.constraints.gamma(predict_fct),
                                   oracle_call_execution_time)
            self._all_results.append(nxt)

        logger.debug("Selecting best_result")
        if self.selection_rule == TRADEOFF_OPTIMIZATION:
            def loss_fct(x):
                return self.objective_weight * x.objective + self.constraint_weight * x.gamma.max()
            self._best_result = min(self._all_results, key=loss_fct)
        else:
            raise RuntimeError("Unsupported selection rule")

        return

    def predict(self, X):
        """Provide a prediction using the best model found by the grid search.

        This dispatches `X` to the :code:`predict(X)` method of the
        selected estimator, and hence the return type is dependent on that method.

        :param X: Feature data
        :type X: numpy.ndarray or pandas.DataFrame
        """
        if self.best_result is None:
            raise NotFittedError(_NO_PREDICT_BEFORE_FIT)
        return self.best_result.predictor.predict(X)

    def predict_proba(self, X):
        """Provide the result of :code:`predict_proba` from the best model found by the grid search.

        The underlying estimator must support :code:`predict_proba(X)` for this
        to work. The return type is determined by this method.

        :param X: Feature data
        :type X: numpy.ndarray or pandas.DataFrame
        """
        if self.best_result is None:
            raise NotFittedError(_NO_PREDICT_BEFORE_FIT)
        return self.best_result.predictor.predict_proba(X)
