# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import numpy as np
import pandas as pd
import pickle
import scipy.optimize as opt

from fairlearn.reductions import moments
from ._constants import _PRECISION, _INDENTATION, _LINE

logger = logging.getLogger(__name__)


class _Lagrangian:
    """ Operations related to the Lagrangian"""

    def __init__(self, dataX, dataA, dataY, learner, constraints, eps, B, opt_lambda=True):
        self.X = dataX
        self.obj = moments.MisclassificationError()
        self.obj.init(dataX, dataA, dataY)
        self.constraints = constraints
        self.constraints.init(dataX, dataA, dataY)
        self.pickled_learner = pickle.dumps(learner)
        self.eps = eps
        self.B = B
        self.opt_lambda = opt_lambda
        self.hs = pd.Series()
        self.classifiers = pd.Series()
        self.errors = pd.Series()
        self.gammas = pd.DataFrame()
        self.n = self.X.shape[0]
        self.n_oracle_calls = 0
        self.last_linprog_n_hs = 0
        self.last_linprog_result = None

    def eval_from_error_gamma(self, error, gamma, lambda_vec):
        """ Return the value of the Lagrangian.
        Returned values:
        L -- value of the Lagrangian
        L_high -- value of the Lagrangian under the best
        response of the lambda player
        """
        lambda_signed = self.constraints.lambda_signed(lambda_vec)
        if self.opt_lambda:
            L = error + np.sum(lambda_vec * gamma) \
                - self.eps * np.sum(lambda_signed.abs())
        else:
            L = error + np.sum(lambda_vec * gamma) \
                - self.eps * np.sum(lambda_vec)
        max_gamma = gamma.max()
        if max_gamma < self.eps:
            L_high = error
        else:
            L_high = error + self.B * (max_gamma - self.eps)
        return L, L_high

    def eval(self, h, lambda_vec):
        """ Return the value of the Lagrangian.
        Returned values:
        L -- value of the Lagrangian
        L_high -- value of the Lagrangian under the best
        response of the lambda player
        gamma -- vector of constraint violations
        error -- the empirical error
        """
        if callable(h):
            error = self.obj.gamma(h)[0]
            gamma = self.constraints.gamma(h)
        else:
            error = self.errors[h.index].dot(h)
            gamma = self.gammas[h.index].dot(h)
        L, L_high = self.eval_from_error_gamma(error, gamma, lambda_vec)
        return L, L_high, gamma, error

    def eval_gap(self, h, lambda_hat, nu):
        """Return the duality gap object for the given h and lambda_hat"""
        L, L_high, gamma, error = self.eval(h, lambda_hat)
        result = _GapResult(L, L, L_high, gamma, error)
        for mul in [1.0, 2.0, 5.0, 10.0]:
            h_hat, h_hat_idx = self.best_h(mul * lambda_hat)
            logger.debug("%smul=%.0f" % (_INDENTATION, mul))
            L_low_mul, _, _, _ = self.eval(
                pd.Series({h_hat_idx: 1.0}), lambda_hat)
            if L_low_mul < result.L_low:
                result.L_low = L_low_mul
            if result.gap() > nu + _PRECISION:
                break
        return result

    def solve_linprog(self, nu):
        n_hs = len(self.hs)
        n_constraints = len(self.constraints.index)
        if self.last_linprog_n_hs == n_hs:
            return self.last_linprog_result
        c = np.concatenate((self.errors, [self.B]))
        A_ub = np.concatenate(
            (self.gammas - self.eps, -np.ones((n_constraints, 1))), axis=1)
        b_ub = np.zeros(n_constraints)
        A_eq = np.concatenate(
            (np.ones((1, n_hs)), np.zeros((1, 1))), axis=1)
        b_eq = np.ones(1)
        result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub,
                             A_eq=A_eq, b_eq=b_eq, method='simplex')
        h = pd.Series(result.x[:-1], self.hs.index)
        dual_c = np.concatenate((b_ub, -b_eq))
        dual_A_ub = np.concatenate(
            (-A_ub.transpose(), A_eq.transpose()), axis=1)
        dual_b_ub = c
        dual_bounds = [
            (None, None) if i == n_constraints else (0, None)
            for i in range(n_constraints + 1)]
        result_dual = opt.linprog(
            dual_c, A_ub=dual_A_ub, b_ub=dual_b_ub, bounds=dual_bounds)
        lambda_vec = pd.Series(result_dual.x[:-1], self.constraints.index)
        self.last_linprog_n_hs = n_hs
        self.last_linprog_result = (
            h, lambda_vec, self.eval_gap(h, lambda_vec, nu))
        return self.last_linprog_result

    def best_h(self, lambda_vec):
        """Return the classifier that solves the best-response problem
        for the vector of Lagrange multipliers lambda_vec."""

        signed_weights = self.obj.signed_weights() \
            + self.constraints.signed_weights(lambda_vec)
        redY = 1 * (signed_weights > 0)
        redW = signed_weights.abs()
        redW = self.n * redW / redW.sum()

        classifier = pickle.loads(self.pickled_learner)
        classifier.fit(self.X, redY, redW)
        self.n_oracle_calls += 1

        def h(X): return classifier.predict(X)
        h_error = self.obj.gamma(h)[0]
        h_gamma = self.constraints.gamma(h)
        h_value = h_error + h_gamma.dot(lambda_vec)

        if not self.hs.empty:
            values = self.errors + self.gammas.transpose().dot(lambda_vec)
            best_idx = values.idxmin()
            best_value = values[best_idx]
        else:
            best_idx = -1
            best_value = np.PINF

        if h_value < best_value - _PRECISION:
            logger.debug("%sbest_h: val improvement %f" % (_LINE, best_value - h_value))
            h_idx = len(self.hs)
            self.hs.at[h_idx] = h
            self.classifiers.at[h_idx] = classifier
            self.errors.at[h_idx] = h_error
            self.gammas[h_idx] = h_gamma
            best_idx = h_idx

        return self.hs[best_idx], best_idx


class _GapResult:
    """ The result of a duality gap computation"""

    def __init__(self, L, L_low, L_high, gamma, error):
        self.L = L
        self.L_low = L_low
        self.L_high = L_high
        self.gamma = gamma
        self.error = error

    def gap(self):
        return max(self.L - self.L_low, self.L_high - self.L)
