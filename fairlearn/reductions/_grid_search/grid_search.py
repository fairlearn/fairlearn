# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import copy
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted
from time import time

from fairlearn.reductions._moments import Moment, ClassificationMoment
from ._grid_generator import _GridGenerator


logger = logging.getLogger(__name__)

TRADEOFF_OPTIMIZATION = "tradeoff_optimization"


class GridSearch(BaseEstimator, MetaEstimatorMixin):
    """Estimator to perform a grid search given a blackbox estimator algorithm.

    The approach used is taken from section 3.4 of
    `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`_.

    Parameters
    ----------
    estimator : estimator
        An estimator implementing methods :code:`fit(X, y, sample_weight)` and
        :code:`predict(X)`, where `X` is the matrix of features, `y` is the
        vector of labels (binary classification) or continuous values
        (regression), and `sample_weight` is a vector of weights.
        In binary classification labels `y` and predictions returned by
        :code:`predict(X)` are either 0 or 1.
        In regression values `y` and predictions are continuous.
    constraints : fairlearn.reductions.Moment
        The disparity constraints expressed as moments
    selection_rule : str
        Specifies the procedure for selecting the best model found by the grid
        search. At the present time, the only valid value is
        "tradeoff_optimization" which minimizes a weighted sum of the error
        rate and constraint violation.
    constraint_weight : float
        When the `selection_rule` is "tradeoff_optimization" this specifies
        the relative weight put on the constraint violation when selecting the
        best model. The weight placed on the error rate will be
        :code:`1-constraint_weight`
    grid_size : int
        The number of Lagrange multipliers to generate in the grid
    grid_limit : float
        The largest Lagrange multiplier to generate. The grid will contain
        values distributed between :code:`-grid_limit` and :code:`grid_limit`
        by default
    grid_offset : :class:`pandas:pandas.DataFrame`
        Shifts the grid of Lagrangian multiplier by that value.
        It is '0' by default
    grid :
        Instead of supplying a size and limit for the grid, users may specify
        the exact set of Lagrange multipliers they desire using this argument.
    sample_weight_name : str
        Name of the argument to `estimator.fit()` which supplies the sample weights
        (defaults to `sample_weight`)
    """

    def __init__(self,
                 estimator,
                 constraints,
                 selection_rule=TRADEOFF_OPTIMIZATION,
                 constraint_weight=0.5,
                 grid_size=10,
                 grid_limit=2.0,
                 grid_offset=None,
                 grid=None,
                 sample_weight_name="sample_weight"):
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
        self.grid_offset = grid_offset
        self.grid = grid
        self.sample_weight_name = sample_weight_name

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
        self.predictors_ = []
        self.lambda_vecs_ = pd.DataFrame(dtype=np.float64)
        self.objectives_ = []
        self.gammas_ = pd.DataFrame(dtype=np.float64)
        self.oracle_execution_times_ = []

        if isinstance(self.constraints, ClassificationMoment):
            logger.debug("Classification problem detected")
            is_classification_reduction = True
        else:
            logger.debug("Regression problem detected")
            is_classification_reduction = False

        # Prep the parity constraints and objective
        logger.debug("Preparing constraints and objective")
        self.constraints.load_data(X, y, **kwargs)
        objective = self.constraints.default_objective()
        objective.load_data(X, y, **kwargs)

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
                                  objective_in_the_span,
                                  self.grid_offset).grid
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
                y_reduction = self.constraints._y_as_series

            y_reduction_unique = np.unique(y_reduction)
            if len(y_reduction_unique) == 1:
                logger.debug("y_reduction had single value. Using DummyClassifier")
                current_estimator = DummyClassifier(strategy='constant',
                                                    constant=y_reduction_unique[0])
            else:
                logger.debug("Using underlying estimator")
                current_estimator = copy.deepcopy(self.estimator)

            oracle_call_start_time = time()
            current_estimator.fit(X, y_reduction, **{self.sample_weight_name: weights})
            oracle_call_execution_time = time() - oracle_call_start_time
            logger.debug("Call to estimator complete")

            def predict_fct(X): return current_estimator.predict(X)
            self.predictors_.append(current_estimator)
            self.lambda_vecs_[i] = lambda_vec
            self.objectives_.append(objective.gamma(predict_fct)[0])
            self.gammas_[i] = self.constraints.gamma(predict_fct)
            self.oracle_execution_times_.append(oracle_call_execution_time)

        logger.debug("Selecting best_result")
        if self.selection_rule == TRADEOFF_OPTIMIZATION:
            def loss_fct(i):
                return self.objective_weight * self.objectives_[i] + \
                    self.constraint_weight * self.gammas_[i].max()
            losses = [loss_fct(i) for i in range(len(self.objectives_))]
            self.best_idx_ = losses.index(min(losses))
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
        check_is_fitted(self)
        return self.predictors_[self.best_idx_].predict(X)

    def predict_proba(self, X):
        """Provide the result of :code:`predict_proba` from the best model found by the grid search.

        The underlying estimator must support :code:`predict_proba(X)` for this
        to work. The return type is determined by this method.

        :param X: Feature data
        :type X: numpy.ndarray or pandas.DataFrame
        """
        check_is_fitted(self)
        return self.predictors_[self.best_idx_].predict_proba(X)
