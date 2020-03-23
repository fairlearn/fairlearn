# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from ._constants import _ACCURACY_MUL, _REGRET_CHECK_START_T, _REGRET_CHECK_INCREASE_T, \
    _SHRINK_REGRET, _SHRINK_ETA, _MIN_T, _RUN_LP_STEP, _PRECISION, _INDENTATION
from ._lagrangian import _Lagrangian
from fairlearn._input_validation import _validate_and_reformat_input

logger = logging.getLogger(__name__)


def _mean_pred(X, hs, weights):
    """Return a weighted average of predictions produced by classifiers in `hs`."""
    pred = pd.DataFrame()
    for t in range(len(hs)):
        pred[t] = hs[t](X)
    return pred[weights.index].dot(weights)


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

    :param eps: Allowed fairness constraint violation; the solution best_classifier is
        guaranteed to have the classification error within :code:`2*best_gap` of the best error
        under constraint eps; the constraint violation is at most :code:`2*(eps+best_gap)`
    :type eps: float

    :param T: Maximum number of iterations
    :type T: int

    :param nu: Convergence threshold for the duality gap, corresponding to a
        conservative automatic setting based on the statistical uncertainty in measuring
        classification error
    :type nu: float

    :param eta_mul: Initial setting of the learning rate
    :type eta_mul: float
    """

    def __init__(self, estimator, constraints, eps=0.01, T=50, nu=None, eta_mul=2.0):  # noqa: D103
        self._estimator = estimator
        self._constraints = constraints
        self._eps = eps
        self._T = T
        self._nu = nu
        self._eta_mul = eta_mul

        self._best_classifier = None
        self._best_gap = None
        self._classifiers = None
        self._weights = None
        self._last_t = None
        self._best_t = None
        self._n_oracle_calls = 0
        self._oracle_calls_execution_time = None

    def fit(self, X, y, **kwargs):
        """Return a fair classifier under specified fairness constraints.

        :param X: The feature matrix
        :type X: numpy.ndarray or pandas.DataFrame

        :param y: The label vector
        :type y: numpy.ndarray, pandas.DataFrame, pandas.Series, or list
        """
        _, y_train, A = _validate_and_reformat_input(X, y, **kwargs)

        n = y_train.shape[0]

        logger.debug("...Exponentiated Gradient STARTING")

        B = 1 / self._eps
        lagrangian = _Lagrangian(X, A, y_train, self._estimator, self._constraints,
                                 self._eps, B)

        theta = pd.Series(0, lagrangian.constraints.index)
        Qsum = pd.Series()
        lambdas = pd.DataFrame()
        gaps_EG = []
        gaps = []
        Qs = []

        last_regret_checked = _REGRET_CHECK_START_T
        last_gap = np.PINF
        for t in range(0, self._T):
            logger.debug("...iter=%03d", t)

            # set lambdas for every constraint
            lambda_vec = B * np.exp(theta) / (1 + np.exp(theta).sum())
            lambdas[t] = lambda_vec
            lambda_EG = lambdas.mean(axis=1)

            # select classifier according to best_h method
            h, h_idx = lagrangian.best_h(lambda_vec)
            pred_h = h(X)

            if t == 0:
                if self._nu is None:
                    self._nu = _ACCURACY_MUL * (pred_h - y_train).abs().std() / np.sqrt(n)
                eta_min = self._nu / (2 * B)
                eta = self._eta_mul / B
                logger.debug("...eps=%.3f, B=%.1f, nu=%.6f, T=%d, eta_min=%.6f",
                             self._eps, B, self._nu, self._T, eta_min)

            if h_idx not in Qsum.index:
                Qsum.at[h_idx] = 0.0
            Qsum[h_idx] += 1.0
            gamma = lagrangian.gammas[h_idx]
            Q_EG = Qsum / Qsum.sum()
            result_EG = lagrangian.eval_gap(Q_EG, lambda_EG, self._nu)
            gap_EG = result_EG.gap()
            gaps_EG.append(gap_EG)

            if t == 0 or not _RUN_LP_STEP:
                gap_LP = np.PINF
            else:
                # saddle point optimization over the convex hull of
                # classifiers returned so far
                Q_LP, _, result_LP = lagrangian.solve_linprog(self._nu)
                gap_LP = result_LP.gap()

            # keep values from exponentiated gradient or linear programming
            if gap_EG < gap_LP:
                Qs.append(Q_EG)
                gaps.append(gap_EG)
            else:
                Qs.append(Q_LP)
                gaps.append(gap_LP)

            logger.debug("%seta=%.6f, L_low=%.3f, L=%.3f, L_high=%.3f"
                         ", gap=%.6f, disp=%.3f, err=%.3f, gap_LP=%.6f",
                         _INDENTATION, eta, result_EG.L_low,
                         result_EG.L, result_EG.L_high,
                         gap_EG, result_EG.gamma.max(),
                         result_EG.error, gap_LP)

            if (gaps[t] < self._nu) and (t >= _MIN_T):
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
            theta += eta * (gamma - self._eps)

        # retain relevant result data
        gaps_series = pd.Series(gaps)
        gaps_best = gaps_series[gaps_series <= gaps_series.min() + _PRECISION]
        self._best_t = gaps_best.index[-1]
        self._best_gap = gaps[self._best_t]
        self._weights = Qs[self._best_t]
        hs = lagrangian.hs
        for h_idx in hs.index:
            if h_idx not in self._weights.index:
                self._weights.at[h_idx] = 0.0

        self._last_t = len(Qs) - 1
        self._best_classifier = lambda X: _mean_pred(X, hs, self._weights)
        self._classifiers = lagrangian.classifiers
        self._n_oracle_calls = lagrangian.n_oracle_calls
        self._oracle_calls_execution_time = lagrangian.oracle_calls_execution_time

        logger.debug("...eps=%.3f, B=%.1f, nu=%.6f, T=%d, eta_min=%.6f",
                     self._eps, B, self._nu, self._T, eta_min)
        logger.debug("...last_t=%d, best_t=%d, best_gap=%.6f, n_oracle_calls=%d, n_hs=%d",
                     self._last_t, self._best_t, self._best_gap, lagrangian.n_oracle_calls,
                     len(lagrangian.classifiers))

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
        positive_probs = self._best_classifier(X)
        return (positive_probs >= np.random.rand(len(positive_probs))) * 1

    def _pmf_predict(self, X):
        """Probability mass function for the given input data.

        :param X: The data for which predictions are required
        :type X: Array
        :return: Array of tuples with the probabilities of predicting 0 and 1.
        :rtype: Array
        """
        positive_probs = self._best_classifier(X)
        return np.concatenate((1-positive_probs, positive_probs), axis=1)
