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


def test_least_squares_regressor_alpha_makes_rank_deficient_fit_unique():
    """Ridge regularization must give a unique, reproducible answer when the
    unregularized problem is rank-deficient (the failure mode in #265)."""
    X, y, sample_weight = _rank_deficient_problem()

    weights_runs = []
    for _ in range(5):
        est = LeastSquaresRegressor(alpha=1e-6)
        est.fit(X, y, sample_weight)
        weights_runs.append(est.weights.values.copy())

    reference = weights_runs[0]
    for w in weights_runs[1:]:
        np.testing.assert_array_equal(w, reference)


def test_least_squares_regressor_predict_uses_fitted_weights():
    X, y, sample_weight = _well_conditioned_problem()
    est = LeastSquaresRegressor(alpha=1e-6)
    est.fit(X, y, sample_weight)
    expected = np.asarray(X).dot(est.weights.values)
    np.testing.assert_allclose(np.asarray(est.predict(X)), expected, atol=1e-12)
