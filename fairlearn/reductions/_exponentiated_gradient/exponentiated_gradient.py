# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils.validation import check_is_fitted
from ._constants import _ACCURACY_MUL, _REGRET_CHECK_START_T, _REGRET_CHECK_INCREASE_T, \
    _SHRINK_REGRET, _SHRINK_ETA, _MIN_ITER, _PRECISION, _INDENTATION
from ._lagrangian import _Lagrangian

from fairlearn.reductions._moments import ClassificationMoment
from fairlearn._input_validation import _validate_and_reformat_input

logger = logging.getLogger(__name__)


class ExponentiatedGradient(BaseEstimator, MetaEstimatorMixin):
    """An Estimator which implements the exponentiated gradient approach to reductions.

    The exponentiated gradient algorithm is described in detail by
    `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`_.

    :param estimator: An estimator implementing methods :code:`fit(X, y, sample_weight)` and
        :code:`predict(X)`, where `X` is the matrix of features, `y` is the vector of labels, and
        `sample_weight` is a vector of weights; labels `y` and predictions returned by
        :code:`predict(X)` are either 0 or 1.
    :type estimator: estimator

    :param constraints: The disparity constraints expressed as moments
    :type constraints: fairlearn.reductions.Moment

    :param eps: Allowed fairness constraint violation; the solution is guaranteed to have the
        error within :code:`2*best_gap` of the best error under constraint eps; the constraint
        violation is at most :code:`2*(eps+best_gap)`
    :type eps: float

    :param max_iter: Maximum number of iterations
    :type max_iter: int

    :param nu: Convergence threshold for the duality gap, corresponding to a
        conservative automatic setting based on the statistical uncertainty in measuring
        classification error
    :type nu: float

    :param eta0: Initial setting of the learning rate
    :type eta0: float

    :param run_linprog_step: if True each step of exponentiated gradient is followed by the saddle
        point optimization over the convex hull of classifiers returned so far; default True
    :type run_linprog_step: bool
    """

    def __init__(self, estimator, constraints, eps=0.01, max_iter=50, nu=None,
                 eta0=2.0, run_linprog_step=True):  # noqa: D103
        self.estimator = estimator
        self.constraints = constraints
        self.eps = eps
        self.max_iter = max_iter
        self.nu = nu
        self.eta0 = eta0
        self.run_linprog_step = run_linprog_step

    def fit(self, X, y, **kwargs):
        """Return a fair classifier under specified fairness constraints.

        :param X: The feature matrix
        :type X: numpy.ndarray or pandas.DataFrame

        :param y: The label vector
        :type y: numpy.ndarray, pandas.DataFrame, pandas.Series, or list
        """
        self.lambda_vecs_EG_ = pd.DataFrame()
        self.lambda_vecs_LP_ = pd.DataFrame()
        self.lambda_vecs_ = pd.DataFrame()

        if isinstance(self.constraints, ClassificationMoment):
            logger.debug("Classification problem detected")
            is_classification_reduction = True
        else:
            logger.debug("Regression problem detected")
            is_classification_reduction = False

        _, y_train, sensitive_features = _validate_and_reformat_input(
            X, y, enforce_binary_labels=is_classification_reduction, **kwargs)

        n = y_train.shape[0]

        logger.debug("...Exponentiated Gradient STARTING")

        B = 1 / self.eps
        lagrangian = _Lagrangian(X, sensitive_features, y_train, self.estimator,
                                 self.constraints, self.eps, B)

        theta = pd.Series(0, lagrangian.constraints.index)
        Qsum = pd.Series(dtype="float64")
        gaps_EG = []
        gaps = []
        Qs = []

        last_regret_checked = _REGRET_CHECK_START_T
        last_gap = np.PINF
        for t in range(0, self.max_iter):
            logger.debug("...iter=%03d", t)

            # set lambdas for every constraint
            lambda_vec = B * np.exp(theta) / (1 + np.exp(theta).sum())
            self.lambda_vecs_EG_[t] = lambda_vec
            lambda_EG = self.lambda_vecs_EG_.mean(axis=1)

            # select classifier according to best_h method
            h, h_idx = lagrangian.best_h(lambda_vec)

            if t == 0:
                if self.nu is None:
                    self.nu = _ACCURACY_MUL * (h(X) - y_train).abs().std() / np.sqrt(n)
                eta_min = self.nu / (2 * B)
                eta = self.eta0 / B
                logger.debug("...eps=%.3f, B=%.1f, nu=%.6f, max_iter=%d, eta_min=%.6f",
                             self.eps, B, self.nu, self.max_iter, eta_min)

            if h_idx not in Qsum.index:
                Qsum.at[h_idx] = 0.0
            Qsum[h_idx] += 1.0
            gamma = lagrangian.gammas[h_idx]
            Q_EG = Qsum / Qsum.sum()
            result_EG = lagrangian.eval_gap(Q_EG, lambda_EG, self.nu)
            gap_EG = result_EG.gap()
            gaps_EG.append(gap_EG)

            if t == 0 or not self.run_linprog_step:
                gap_LP = np.PINF
            else:
                # saddle point optimization over the convex hull of
                # classifiers returned so far
                Q_LP, self.lambda_vecs_LP_[t], result_LP = lagrangian.solve_linprog(self.nu)
                gap_LP = result_LP.gap()

            # keep values from exponentiated gradient or linear programming
            if gap_EG < gap_LP:
                Qs.append(Q_EG)
                gaps.append(gap_EG)
            else:
                Qs.append(Q_LP)
                gaps.append(gap_LP)

            logger.debug("%seta=%.6f, L_low=%.3f, L=%.3f, L_high=%.3f, gap=%.6f, disp=%.3f, "
                         "err=%.3f, gap_LP=%.6f",
                         _INDENTATION, eta, result_EG.L_low, result_EG.L, result_EG.L_high,
                         gap_EG, result_EG.gamma.max(), result_EG.error, gap_LP)

            if (gaps[t] < self.nu) and (t >= _MIN_ITER):
                # solution found
                break

            # update regret
            if t >= last_regret_checked * _REGRET_CHECK_INCREASE_T:
                best_gap = min(gaps_EG)

                if best_gap > last_gap * _SHRINK_REGRET:
                    eta *= _SHRINK_ETA
                last_regret_checked = t
                last_gap = best_gap

            # update theta based on learning rate
            theta += eta * (gamma - self.eps)

        # retain relevant result data
        gaps_series = pd.Series(gaps)
        gaps_best = gaps_series[gaps_series <= gaps_series.min() + _PRECISION]
        self.best_iter_ = gaps_best.index[-1]
        self.best_gap_ = gaps[self.best_iter_]
        self.weights_ = Qs[self.best_iter_]
        self._hs = lagrangian.hs
        for h_idx in self._hs.index:
            if h_idx not in self.weights_.index:
                self.weights_.at[h_idx] = 0.0

        self.last_iter_ = len(Qs) - 1
        self.predictors_ = lagrangian.predictors
        self.n_oracle_calls_ = lagrangian.n_oracle_calls
        self.n_oracle_calls_dummy_returned_ = lagrangian.n_oracle_calls_dummy_returned
        self.oracle_execution_times_ = lagrangian.oracle_execution_times
        self.lambda_vecs_ = lagrangian.lambdas

        logger.debug("...eps=%.3f, B=%.1f, nu=%.6f, max_iter=%d, eta_min=%.6f",
                     self.eps, B, self.nu, self.max_iter, eta_min)
        logger.debug("...last_iter=%d, best_iter=%d, best_gap=%.6f, n_oracle_calls=%d, n_hs=%d",
                     self.last_iter_, self.best_iter_, self.best_gap_, lagrangian.n_oracle_calls,
                     len(lagrangian.predictors))

    def predict(self, X):
        """Provide a prediction for the given input data.

        Note that this is non-deterministic, due to the nature of the
        exponentiated gradient algorithm.

        :param X: Feature data
        :type X: numpy.ndarray or pandas.DataFrame

        :return: The prediction. If `X` represents the data for a single example
            the result will be a scalar. Otherwise the result will be a vector
        :rtype: Scalar or vector
        """
        check_is_fitted(self)

        positive_probs = self._pmf_predict(X)[:, 1]
        return (positive_probs >= np.random.rand(len(positive_probs))) * 1

    def _pmf_predict(self, X):
        """Probability mass function for the given input data.

        :param X: Feature data
        :type X: numpy.ndarray or pandas.DataFrame
        :return: Array of tuples with the probabilities of predicting 0 and 1.
        :rtype: pandas.DataFrame
        """
        check_is_fitted(self)

        pred = pd.DataFrame()
        for t in range(len(self._hs)):
            pred[t] = self._hs[t](X)
        positive_probs = pred[self.weights_.index].dot(self.weights_).to_frame()
        return np.concatenate((1-positive_probs, positive_probs), axis=1)
