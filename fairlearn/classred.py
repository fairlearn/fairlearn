# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This module implements the Lagrangian reduction of fair binary
classification to standard binary classification.

FUNCTIONS
expgrad -- optimize accuracy subject to fairness constraints
"""

from __future__ import print_function

__all__ = ["expgrad"]

import numpy as np
import scipy.optimize as opt
import pandas as pd
import pickle
import functools
import fairlearn.moments as moments
from collections import namedtuple

print = functools.partial(print, flush=True)

_PRECISION = 1e-12


class _GapResult:
    # The result of a duality gap computation
    def __init__(self, L, L_low, L_high, gamma, error):
        self.L = L
        self.L_low = L_low
        self.L_high = L_high
        self.gamma = gamma
        self.error = error

    def gap(self):
        return max(self.L-self.L_low, self.L_high-self.L)


class _Lagrangian:
    # Operations related to the Lagrangian
    def __init__(self, dataX, dataA, dataY, learner, cons, eps, B,
                 opt_lambda=True, debug=False):
        self.X = dataX
        self.obj = moments.MisclassError()
        self.obj.init(dataX, dataA, dataY)
        self.cons = cons
        self.cons.init(dataX, dataA, dataY)
        self.pickled_learner = pickle.dumps(learner)
        self.eps = eps
        self.B = B
        self.opt_lambda = opt_lambda
        self.debug = debug
        self.hs = pd.Series()
        self.classifiers = pd.Series()
        self.errors = pd.Series()
        self.gammas = pd.DataFrame()
        self.n = self.X.shape[0]
        self.n_oracle_calls = 0
        self.last_linprog_n_hs = 0
        self.last_linprog_result = None
        
    def eval_from_error_gamma(self, error, gamma, lambda_vec):
        # Return the value of the Lagrangian.
        #
        # Returned values:
        #   L -- value of the Lagrangian
        #   L_high -- value of the Lagrangian under the best response of the
        #             lambda player
        
        lambda_signed = lambda_vec["+"] - lambda_vec["-"]
        if self.opt_lambda:
            L = error + np.sum(lambda_vec*gamma) \
                - self.eps*np.sum(lambda_signed.abs())
        else:
            L = error + np.sum(lambda_vec*gamma) \
                - self.eps*np.sum(lambda_vec)
        max_gamma = gamma.max()
        if max_gamma < self.eps:
            L_high = error
        else:
            L_high = error + self.B*(max_gamma-self.eps)
        return L, L_high
    
    def eval(self, h, lambda_vec):
        # Return the value of the Lagrangian.
        #
        # Returned values:
        #   L -- value of the Lagrangian
        #   L_high -- value of the Lagrangian under the best response of the
        #             lambda player
        #   gamma -- vector of constraint violations
        #   error -- the empirical error
        
        if callable(h):
            error = self.obj.gamma(h)[0]
            gamma = self.cons.gamma(h)
        else:
            error = self.errors[h.index].dot(h)
            gamma = self.gammas[h.index].dot(h)
        L, L_high = self.eval_from_error_gamma(error, gamma, lambda_vec)
        return L, L_high, gamma, error

    def eval_gap(self, h, lambda_hat, nu):
        # Return the duality gap object for the given h and lambda_hat
        
        L, L_high, gamma, error \
            = self.eval(h, lambda_hat)
        res = _GapResult(L, L, L_high, gamma, error)
        for mul in [1.0, 2.0, 5.0, 10.0]:
            h_hat, h_hat_idx = self.best_h(mul*lambda_hat)
            if self.debug:
                print("%smul=%.0f" % (" "*9, mul))
            L_low_mul, tmp, tmp, tmp \
                = self.eval(pd.Series({h_hat_idx: 1.0}), lambda_hat)
            if (L_low_mul < res.L_low):
                res.L_low = L_low_mul
            if res.gap() > nu+_PRECISION:
                break
        return res
    
    def solve_linprog(self, nu):
        n_hs = len(self.hs)
        n_cons = len(self.cons.index)
        if self.last_linprog_n_hs == n_hs:
            return self.last_linprog_res
        c = np.concatenate((self.errors, [self.B]))
        A_ub = np.concatenate(
            (self.gammas-self.eps, -np.ones((n_cons, 1))), axis=1)
        b_ub = np.zeros(n_cons)
        A_eq = np.concatenate(
            (np.ones((1, n_hs)), np.zeros((1, 1))), axis=1)
        b_eq = np.ones(1)
        res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        h = pd.Series(res.x[:-1], self.hs.index)
        dual_c = np.concatenate((b_ub, -b_eq))
        dual_A_ub = np.concatenate(
            (-A_ub.transpose(), A_eq.transpose()), axis=1)
        dual_b_ub = c
        dual_bounds = [
            (None, None) if i==n_cons else (0, None) for i in range(n_cons+1)]
        res_dual = opt.linprog(dual_c, A_ub=dual_A_ub, b_ub=dual_b_ub,
                               bounds=dual_bounds)
        lambda_vec = pd.Series(res_dual.x[:-1], self.cons.index)
        self.last_linprog_n_hs = n_hs
        self.last_linprog_res = (h, lambda_vec,
                                 self.eval_gap(h, lambda_vec, nu))
        return self.last_linprog_res

    def best_h(self, lambda_vec):
        # Return the classifier that solves the best-response problem
        # for the vector of Lagrange multipliers lambda_vec.
    
        signed_weights = self.obj.signed_weights() \
                         + self.cons.signed_weights(lambda_vec)
        redY = 1*(signed_weights > 0)
        redW = signed_weights.abs()
        redW = self.n*redW/redW.sum()

        classifier = pickle.loads(self.pickled_learner)
        classifier.fit(self.X, redY, redW)
        self.n_oracle_calls += 1
        
        h = lambda X: classifier.predict(X)
        h_error = self.obj.gamma(h)[0]
        h_gamma = self.cons.gamma(h)
        h_val = h_error + h_gamma.dot(lambda_vec)

        if not self.hs.empty:
            vals =  self.errors + self.gammas.transpose().dot(lambda_vec)
            best_idx = vals.idxmin()
            best_val = vals[best_idx]
        else:
            best_idx = -1
            best_val = np.PINF

        if h_val < best_val-_PRECISION:
            if self.debug:
                print("%sbest_h: val improvement %f" % ("_"*9, best_val-h_val))
            h_idx = len(self.hs)
            self.hs.at[h_idx] = h
            self.classifiers.at[h_idx] = classifier
            self.errors.at[h_idx] = h_error
            self.gammas[h_idx] = h_gamma
            best_idx = h_idx

        return self.hs[best_idx], best_idx


def _mean_pred(dataX, hs, weights):
    # Return a weighted average of predictions produced by classifiers in hs
    
    pred = pd.DataFrame()
    for t in range(len(hs)):
        pred[t] = hs[t](dataX)
    return pred[weights.index].dot(weights)


### Explicit optimization parameters of expgrad

# A multiplier controlling the automatic setting of nu.
_ACCURACY_MUL = 0.5

# Parameters controlling adaptive shrinking of the learning rate.
_REGR_CHECK_START_T = 5
_REGR_CHECK_INCREASE_T = 1.6
_SHRINK_REGRET = 0.8
_SHRINK_ETA = 0.8

# The smallest number of iterations after which expgrad terminates.
_MIN_T = 5

# If _RUN_LP_STEP is set to True, then each step of exponentiated gradient is
# followed by the saddle point optimization over the convex hull of
# classifiers returned so far.
_RUN_LP_STEP = True


def expgrad(dataX, dataA, dataY, learner, cons=moments.DP(), eps=0.01,
            T=50, nu=None, eta_mul=2.0, debug=False):
    """
    Return a fair classifier under specified fairness constraints
    via exponentiated-gradient reduction.
    
    Required input arguments:
      dataX -- a DataFrame containing covariates
      dataA -- a Series containing the protected attribute
      dataY -- a Series containing labels in {0,1}
      learner -- a learner implementing methods fit(X,Y,W) and predict(X),
                 where X is the DataFrame of covariates, and Y and W
                 are the Series containing the labels and weights,
                 respectively; labels Y and predictions returned by
                 predict(X) are in {0,1}

    Optional keyword arguments:
      cons -- the fairness measure (default moments.DP())
      eps -- allowed fairness constraint violation (default 0.01)
      T -- max number of iterations (default 50)
      nu -- convergence threshold for the duality gap (default None,
            corresponding to a conservative automatic setting based on the
            statistical uncertainty in measuring classification error)
      eta_mul -- initial setting of the learning rate (default 2.0)
      debug -- if True, then debugging output is produced (default False)

    Returned named tuple with fields:
      best_classifier -- a function that maps a DataFrame X containing
                         covariates to a Series containing the corresponding
                         probabilistic decisions in [0,1]
      best_gap -- the quality of best_classifier; if the algorithm has
                  converged then best_gap<= nu; the solution best_classifier
                  is guaranteed to have the classification error within
                  2*best_gap of the best error under constraint eps; the
                  constraint violation is at most 2*(eps+best_gap)
      classifiers -- the base classifiers generated (instances of learner)
      weights -- the weights of those classifiers within best_classifier
      last_t -- the last executed iteration; always last_t < T
      best_t -- the iteration in which best_classifier was obtained
      n_oracle_calls -- how many times the learner was called
    """

    ExpgradResult = namedtuple("ExgradResult",
                               "best_classifier best_gap classifiers weights"
                               " last_t best_t n_oracle_calls")

    n = dataX.shape[0]
    assert len(dataX.shape)==2 and len(dataA.shape)==1 and len(dataY.shape)==1, \
        "dataX must be a DataFrame and dataY and dataA must be Series"
    assert dataA.shape[0]==n and dataY.shape[0]==n, \
        "the number of rows in all data fields must match"

    if debug:
        print("...EG STARTING")

    B = 1/eps
    lagr = _Lagrangian(dataX, dataA, dataY, learner, cons, eps, B,
                       debug=debug)

    theta  = pd.Series(0, lagr.cons.index)
    Qsum = pd.Series()
    lambdas  = pd.DataFrame()
    gaps_EG = []
    gaps = []
    Qs = []

    last_regr_checked = _REGR_CHECK_START_T
    last_gap = np.PINF
    for t in range(0, T):
        if debug:
            print("...iter=%03d" % t)

        lambda_vec = B*np.exp(theta) / (1+np.exp(theta).sum())
        lambdas[t] = lambda_vec
        lambda_EG = lambdas.mean(axis=1)

        h, h_idx = lagr.best_h(lambda_vec)
        pred_h = h(dataX)

        if t == 0:
            if nu is None:
                nu = _ACCURACY_MUL * (pred_h-dataY).abs().std() / np.sqrt(n)
            eta_min = nu / (2*B)
            eta = eta_mul / B
            if debug:
                print("...eps=%.3f, B=%.1f, nu=%.6f, T=%d, eta_min=%.6f"
                      % (eps, B, nu, T, eta_min))

        if not Qsum.index.contains(h_idx):
            Qsum.at[h_idx] = 0.0
        Qsum[h_idx] += 1.0
        gamma = lagr.gammas[h_idx]

        Q_EG = Qsum / Qsum.sum()
        res_EG = lagr.eval_gap(Q_EG, lambda_EG, nu)
        gap_EG = res_EG.gap()
        gaps_EG.append(gap_EG)
        
        if (t == 0) or not _RUN_LP_STEP:
            gap_LP = np.PINF
        else:
            Q_LP, lambda_LP, res_LP = lagr.solve_linprog(nu)
            gap_LP = res_LP.gap()
            
        if gap_EG < gap_LP:
            Qs.append(Q_EG)
            gaps.append(gap_EG)
        else:
            Qs.append(Q_LP)
            gaps.append(gap_LP)

        if debug:
            print("%seta=%.6f, L_low=%.3f, L=%.3f, L_high=%.3f"
                  ", gap=%.6f, disp=%.3f, err=%.3f, gap_LP=%.6f"
                  % (" "*9, eta, res_EG.L_low, res_EG.L, res_EG.L_high,
                     gap_EG, res_EG.gamma.max(), res_EG.error, gap_LP))

        if (gaps[t] < nu) and (t >= _MIN_T):
            break

        if t >= last_regr_checked*_REGR_CHECK_INCREASE_T:
            best_gap = min(gaps_EG)

            if best_gap > last_gap*_SHRINK_REGRET:
                eta *= _SHRINK_ETA
            last_regr_checked = t
            last_gap = best_gap
            
        theta += eta*(gamma-eps)
        
    last_t = len(Qs)-1
    gaps_series = pd.Series(gaps)
    gaps_best = gaps_series[gaps_series<=gaps_series.min()+_PRECISION]
    best_t = gaps_best.index[-1]
    weights = Qs[best_t]
    hs = lagr.hs
    for h_idx in hs.index:
        if not weights.index.contains(h_idx):
            weights.at[h_idx] = 0.0
    best_classifier = lambda X : _mean_pred(X, hs, weights)
    best_gap = gaps[best_t]

    res = ExpgradResult(best_classifier=best_classifier,
                        best_gap=best_gap,
                        classifiers=lagr.classifiers,
                        weights=weights,
                        last_t=last_t,
                        best_t=best_t,
                        n_oracle_calls=lagr.n_oracle_calls)

    if debug:
        print("...eps=%.3f, B=%.1f, nu=%.6f, T=%d, eta_min=%.6f"
              % (eps, B, nu, T, eta_min))
        print("...last_t=%d, best_t=%d, best_gap=%.6f"
              ", n_oracle_calls=%d, n_hs=%d"
              % (res.last_t, res.best_t, res.best_gap,
                 res.n_oracle_calls, len(res.classifiers)))

    return res
