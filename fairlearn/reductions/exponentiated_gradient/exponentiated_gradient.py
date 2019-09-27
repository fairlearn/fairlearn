# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This module implements the Lagrangian reduction of fair binary
classification to standard binary classification.

FUNCTIONS
exponentiated_gradient -- optimize accuracy subject to fairness constraints
"""

import logging
import numpy as np
import pandas as pd
from fairlearn.reductions import ReductionsEstimator
from ._constants import _ACCURACY_MUL, _REGRET_CHECK_START_T, _REGRET_CHECK_INCREASE_T, \
    _SHRINK_REGRET, _SHRINK_ETA, _MIN_T, _RUN_LP_STEP, _PRECISION, _INDENTATION
from ._lagrangian import _Lagrangian

logger = logging.getLogger(__name__)


def _mean_pred(X, hs, weights):
    """Return a weighted average of predictions produced by classifiers in hs"""
    pred = pd.DataFrame()
    for t in range(len(hs)):
        pred[t] = hs[t](X)
    return pred[weights.index].dot(weights)


class ExponentiatedGradientResult:
    def __init__(self, best_classifier, best_gap, classifiers, weights, last_t, best_t,
                 n_oracle_calls):
        self._best_classifier = best_classifier
        self._best_gap = best_gap
        self._classifiers = classifiers
        self._weights = weights
        self._last_t = last_t
        self._best_t = best_t
        self._n_oracle_calls = n_oracle_calls

    def _as_dict(self):
        return {
            "best_classifier": self._best_classifier,
            "best_gap": self._best_gap,
            "classifiers": self._classifiers,
            "weights": self._weights,
            "last_t": self._last_t,
            "best_t": self._best_t,
            "n_oracle_calls": self._n_oracle_calls
        }


class ExponentiatedGradient(ReductionsEstimator):
    def __init__(self, estimator, constraints, eps=0.01, T=50, nu=None, eta_mul=2.0):
        self._estimator = estimator
        self._constraints = constraints
        self._eps = eps
        self._T = T
        self._nu = nu
        self._eta_mul = eta_mul
        self._best_classifier = None
        self._classifiers = None

    def fit(self, X, y, group_data=None, **kwargs):
        # TODO: validate input data; unify between grid search and expgrad?
        self._expgrad_result = exponentiated_gradient(X, group_data, y, self._estimator,
                                                      constraints=self._constraints,
                                                      eps=self._eps, T=self._T, nu=self._nu,
                                                      eta_mul=self._eta_mul)
        self._best_classifier = self._expgrad_result._best_classifier
        self._classifiers = self._expgrad_result._classifiers
        # TODO: figure out whether we should keep the remaining data of the result object

    def predict(self, X):
        return self._best_classifier(X)

    def predict_proba(self, X):
        raise NotImplementedError()

    def posterior_predict(self, X):
        raise NotImplementedError()

    def posterior_predict_proba(self, X):
        raise NotImplementedError()


def exponentiated_gradient(X, A, y, estimator,
                           constraints,
                           eps=0.01,
                           T=50,
                           nu=None,
                           eta_mul=2.0):
    """
    Return a fair classifier under specified fairness constraints via exponentiated-gradient
    reduction.

    :param X: a DataFrame containing covariates
    :param A: a Series containing the protected attribute
    :param y: a Series containing labels in {0,1}
    :param estimator: an estimator implementing methods fit(X,Y,W) and predict(X), where X is the
        DataFrame of covariates, and Y and W are the Series containing the labels and weights,
        respectively; labels Y and predictions returned by predict(X) are in {0,1}
    :param constraints: the disparity constraints expressed as moments
    :param eps: allowed fairness constraint violation (default 0.01)
    :param T: max number of iterations (default 50)
    :param nu: convergence threshold for the duality gap (default None), corresponding to a
        conservative automatic setting based on the statistical uncertainty in measuring
        classification error)
    :param eta_mul: initial setting of the learning rate (default 2.0)
    :param debug: if True, then debugging output is produced (default False)

    :return: Returns named tuple with fields:
      best_classifier -- a function that maps a DataFrame X containing
                         covariates to a Series containing the corresponding
                         probabilistic decisions in [0,1]
      best_gap -- the quality of best_classifier; if the algorithm has
                  converged then best_gap <= nu; the solution best_classifier
                  is guaranteed to have the classification error within
                  2*best_gap of the best error under constraint eps; the
                  constraint violation is at most 2*(eps+best_gap)
      classifiers -- the base classifiers generated (instances of estimator)
      weights -- the weights of those classifiers within best_classifier
      last_t -- the last executed iteration; always last_t < T
      best_t -- the iteration in which best_classifier was obtained
      n_oracle_calls -- how many times the estimator was called
    """
    n = X.shape[0]

    logger.debug("...Exponentiated Gradient STARTING")

    B = 1 / eps
    lagrangian = _Lagrangian(X, A, y, estimator, constraints, eps, B)

    theta = pd.Series(0, lagrangian.constraints.index)
    Qsum = pd.Series()
    lambdas = pd.DataFrame()
    gaps_EG = []
    gaps = []
    Qs = []

    last_regret_checked = _REGRET_CHECK_START_T
    last_gap = np.PINF
    for t in range(0, T):
        logger.debug("...iter=%03d" % t)

        # set lambdas for every constraint
        lambda_vec = B * np.exp(theta) / (1 + np.exp(theta).sum())
        lambdas[t] = lambda_vec
        lambda_EG = lambdas.mean(axis=1)

        # select classifier according to best_h method
        h, h_idx = lagrangian.best_h(lambda_vec)
        pred_h = h(X)

        if t == 0:
            if nu is None:
                nu = _ACCURACY_MUL * (pred_h - y).abs().std() / np.sqrt(n)
            eta_min = nu / (2 * B)
            eta = eta_mul / B
            logger.debug("...eps=%.3f, B=%.1f, nu=%.6f, T=%d, eta_min=%.6f"
                         % (eps, B, nu, T, eta_min))

        if h_idx not in Qsum.index:
            Qsum.at[h_idx] = 0.0
        Qsum[h_idx] += 1.0
        gamma = lagrangian.gammas[h_idx]
        Q_EG = Qsum / Qsum.sum()
        result_EG = lagrangian.eval_gap(Q_EG, lambda_EG, nu)
        gap_EG = result_EG.gap()
        gaps_EG.append(gap_EG)

        if t == 0 or not _RUN_LP_STEP:
            gap_LP = np.PINF
        else:
            # saddle point optimization over the convex hull of
            # classifiers returned so far
            Q_LP, _, result_LP = lagrangian.solve_linprog(nu)
            gap_LP = result_LP.gap()

        # keep values from exponentiated gradient or linear programming
        if gap_EG < gap_LP:
            Qs.append(Q_EG)
            gaps.append(gap_EG)
        else:
            Qs.append(Q_LP)
            gaps.append(gap_LP)

        logger.debug("%seta=%.6f, L_low=%.3f, L=%.3f, L_high=%.3f"
                     ", gap=%.6f, disp=%.3f, err=%.3f, gap_LP=%.6f"
                     % (_INDENTATION, eta, result_EG.L_low,
                        result_EG.L, result_EG.L_high,
                        gap_EG, result_EG.gamma.max(),
                        result_EG.error, gap_LP))

        if (gaps[t] < nu) and (t >= _MIN_T):
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
        theta += eta * (gamma - eps)

    return _format_results(gaps, Qs, lagrangian, eps, B, nu, T, eta_min)


def _format_results(gaps, Qs, lagrangian, eps, B, nu, T, eta_min):
    gaps_series = pd.Series(gaps)
    gaps_best = gaps_series[gaps_series <= gaps_series.min() + _PRECISION]
    best_t = gaps_best.index[-1]
    weights = Qs[best_t]
    hs = lagrangian.hs
    for h_idx in hs.index:
        if h_idx not in weights.index:
            weights.at[h_idx] = 0.0

    def best_classifier(X): return _mean_pred(X, hs, weights)
    best_gap = gaps[best_t]

    last_t = len(Qs) - 1

    result = ExponentiatedGradientResult(
        best_classifier,
        best_gap,
        lagrangian.classifiers,
        weights,
        last_t,
        best_t,
        lagrangian.n_oracle_calls)

    logger.debug("...eps=%.3f, B=%.1f, nu=%.6f, T=%d, eta_min=%.6f"
                 % (eps, B, nu, T, eta_min))
    logger.debug("...last_t=%d, best_t=%d, best_gap=%.6f, n_oracle_calls=%d, n_hs=%d"
                 % (last_t, best_t, best_gap, lagrangian.n_oracle_calls,
                    len(lagrangian.classifiers)))

    return result
