# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import logging
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from time import time

from fairlearn._input_validation import _validate_and_reformat_input, _KW_SENSITIVE_FEATURES
from fairlearn import _NO_PREDICT_BEFORE_FIT
from fairlearn.reductions._moments import Moment, ClassificationMoment
from ._grid_generator import _GridGenerator


logger = logging.getLogger(__name__)

TRADEOFF_OPTIMIZATION = "tradeoff_optimization"


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

        self._best_grid_index = None
        self._predictors = []
        self._lambda_vecs = pd.DataFrame()
        self._objectives = []
        self._gammas = pd.DataFrame()
        self._oracle_execution_times = []

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
            self._predictors.append(current_estimator)
            self._lambda_vecs[i] = lambda_vec
            self._objectives.append(objective.gamma(predict_fct)[0])
            self._gammas[i] = self.constraints.gamma(predict_fct)
            self._oracle_execution_times.append(oracle_call_execution_time)

        logger.debug("Selecting best_result")
        if self.selection_rule == TRADEOFF_OPTIMIZATION:
            def loss_fct(i):
                return self.objective_weight * self._objectives[i] + \
                    self.constraint_weight * self._gammas[i].max()
            losses = [loss_fct(i) for i in range(len(self._objectives))]
            self._best_grid_index = losses.index(min(losses))
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
        if self._best_grid_index is None:
            raise NotFittedError(_NO_PREDICT_BEFORE_FIT)
        return self._predictors[self._best_grid_index].predict(X)

    def predict_proba(self, X):
        """Provide the result of :code:`predict_proba` from the best model found by the grid search.

        The underlying estimator must support :code:`predict_proba(X)` for this
        to work. The return type is determined by this method.

        :param X: Feature data
        :type X: numpy.ndarray or pandas.DataFrame
        """
        if self._best_grid_index is None:
            raise NotFittedError(_NO_PREDICT_BEFORE_FIT)
        return self._predictors[self._best_grid_index].predict_proba(X)
