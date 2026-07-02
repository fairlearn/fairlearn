# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Tests for the small in-test estimators in ``simple_learners``.

The :class:`LeastSquaresRegressor` here exposes an ``alpha`` knob used by
``test_bgl_unfair`` to make the inner regression's optimum mathematically
unique (and hence platform-deterministic). See
https://github.com/fairlearn/fairlearn/issues/265.
"""

import numpy as np
import pandas as pd

from .simple_learners import LeastSquaresRegressor


def _well_conditioned_problem(seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((20, 3)), columns=["a", "b", "c"])
    y = 2.0 * X["a"].to_numpy() - 1.5 * X["b"].to_numpy() + 0.3
    y = y + 0.01 * rng.standard_normal(20)
    sample_weight = np.ones(20)
    return X, y, sample_weight


def _rank_deficient_problem(seed=0):
    """Design matrix with an exact linear dependency: column c == a + b."""
    X, y, sample_weight = _well_conditioned_problem(seed=seed)
    X = X.copy()
    X["c"] = X["a"] + X["b"]
    return X, y, sample_weight


def test_least_squares_regressor_alpha_default_matches_unregularized():
    X, y, sample_weight = _well_conditioned_problem()
    est = LeastSquaresRegressor()
    est.fit(X, y, sample_weight)

    # Compare to the explicit unregularized weighted LS via np.linalg.lstsq.
    sqrtW = np.sqrt(sample_weight)
    matX = np.asarray(X) * sqrtW[:, np.newaxis]
    vecY = np.asarray(y) * sqrtW
    expected, *_ = np.linalg.lstsq(matX, vecY, rcond=-1)

    np.testing.assert_allclose(est.weights.values, expected, atol=1e-12)


def test_least_squares_regressor_small_alpha_negligibly_perturbs_well_conditioned_fit():
    """For a well-conditioned design, ``alpha=1e-6`` must barely move the fit."""
    X, y, sample_weight = _well_conditioned_problem()
    est_unreg = LeastSquaresRegressor(alpha=0.0)
    est_unreg.fit(X, y, sample_weight)
    est_reg = LeastSquaresRegressor(alpha=1e-6)
    est_reg.fit(X, y, sample_weight)

    np.testing.assert_allclose(est_reg.weights.values, est_unreg.weights.values, atol=1e-5)


def test_least_squares_regressor_alpha_matches_closed_form_ridge():
    """``alpha>0`` must match the analytic ridge solution to machine precision.

    The closed form is ``beta = (X^T W X + alpha I)^{-1} X^T W y``. Verifying
    this is the strongest correctness check for the augmented-``lstsq``
    formulation: it guarantees the implementation computes a *specific*,
    mathematically uniquely-defined point — not merely a consistent one.
    """
    X, y, sample_weight = _well_conditioned_problem()
    alpha = 1e-6

    est = LeastSquaresRegressor(alpha=alpha)
    est.fit(X, y, sample_weight)

    X_arr = np.asarray(X)
    W = np.diag(sample_weight)
    beta_closed = np.linalg.solve(
        X_arr.T @ W @ X_arr + alpha * np.eye(X_arr.shape[1]),
        X_arr.T @ W @ np.asarray(y),
    )

    np.testing.assert_allclose(est.weights.values, beta_closed, atol=1e-12)


def test_least_squares_regressor_alpha_resolves_rank_deficient_optimum():
    """Ridge with ``alpha>0`` gives a single mathematically-defined answer on
    a rank-deficient design — the exact failure mode in
    https://github.com/fairlearn/fairlearn/issues/265.

    On a rank-deficient design the unregularized least-squares loss is
    minimized by an entire affine subspace; *which* element of that subspace
    a solver picks depends on its LAPACK driver, and that's the published
    platform-flakiness. Ridge regularization makes ``X^T W X + alpha I``
    strictly positive-definite for any ``alpha > 0``, so the minimizer is
    uniquely defined and equal to the closed form below regardless of which
    solver computes it.
    """
    X, y, sample_weight = _rank_deficient_problem()

    # Sanity-check that the design really is rank-deficient — otherwise the
    # test would say nothing about the failure mode we're guarding against.
    assert np.linalg.matrix_rank(np.asarray(X)) < X.shape[1]

    alpha = 1e-6
    est = LeastSquaresRegressor(alpha=alpha)
    est.fit(X, y, sample_weight)

    X_arr = np.asarray(X)
    W = np.diag(sample_weight)
    beta_closed = np.linalg.solve(
        X_arr.T @ W @ X_arr + alpha * np.eye(X_arr.shape[1]),
        X_arr.T @ W @ np.asarray(y),
    )

    np.testing.assert_allclose(est.weights.values, beta_closed, atol=1e-10)


def test_least_squares_regressor_predict_uses_fitted_weights():
    X, y, sample_weight = _well_conditioned_problem()
    est = LeastSquaresRegressor(alpha=1e-6)
    est.fit(X, y, sample_weight)
    expected = np.asarray(X).dot(est.weights.values)
    np.testing.assert_allclose(np.asarray(est.predict(X)), expected, atol=1e-12)
