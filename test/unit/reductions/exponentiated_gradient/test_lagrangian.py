# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pickle
import pytest

from fairlearn.reductions._exponentiated_gradient._lagrangian import _Lagrangian
from fairlearn.reductions import DemographicParity, EqualizedOdds

from .test_utilities import _get_data
from .simple_learners import LeastSquaresBinaryClassifierLearner


@pytest.mark.parametrize("eps", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("Constraints", [DemographicParity, EqualizedOdds])
@pytest.mark.parametrize("h_vec", [pd.Series([1.0])])
@pytest.mark.parametrize("opt_lambda", [True, False])
def test_lagrangian_eval(eps, Constraints, h_vec, opt_lambda):
    X, y, A = _get_data(A_two_dim=False)
    estimator = LeastSquaresBinaryClassifierLearner()
    constraints = Constraints()

    # epsilon (and thereby also B) only affects L_high and L
    B = 1 / eps

    lagrangian = _Lagrangian(X, A, y, estimator, constraints, eps, B, opt_lambda=opt_lambda)

    # set up initial lambda vector based on a 0-initialized theta
    theta = pd.Series(0, lagrangian.constraints.index)
    lambda_vec = np.exp(theta) / (1 + np.exp(theta).sum())

    # call oracle to determine error and gamma and calculate exp
    fitted_estimator = lagrangian._call_oracle(lambda_vec)
    def h(X): return fitted_estimator.predict(X)
    best_h_error = lagrangian.obj.gamma(h)[0]
    best_h_gamma = lagrangian.constraints.gamma(h)

    # opt_lambda affects only the calculation of L
    if opt_lambda:
        projected_lambda = lagrangian.constraints.project_lambda(lambda_vec)
        L_expected = best_h_error + np.sum(projected_lambda * best_h_gamma) - \
            eps * np.sum(projected_lambda)
    else:
        L_expected = best_h_error + np.sum(lambda_vec * best_h_gamma) - eps * np.sum(lambda_vec)

    L_high_expected = best_h_error + B * (best_h_gamma.max() - eps)

    # manually set errors and gammas which would otherwise be done in the best_h step
    lagrangian.errors_ = pd.Series([best_h_error])
    lagrangian.gammas_ = pd.Series([best_h_gamma])

    # call _eval to get the desired results L, L_high, gamma, error
    L, L_high, gamma, error = lagrangian._eval(h_vec, lambda_vec)

    assert L == L_expected
    assert L_high == L_high_expected
    assert error == 0.25
    assert (gamma == best_h_gamma).all()

    # in this particular example the estimator is always the same
    expected_estimator_weights = pd.Series({
        'X1': 0.538136,
        'X2': 0.457627,
        'X3': 0.021186})
    assert (np.isclose(fitted_estimator.weights, expected_estimator_weights, atol=1.e-6)).all()


@pytest.mark.parametrize("Constraints", [DemographicParity, EqualizedOdds])
@pytest.mark.parametrize("eps", [0.001, 0.01, 0.1])
def test_call_oracle(Constraints, eps, mocker):
    X, y, A = _get_data(A_two_dim=False)
    # Using a mocked estimator here since we don't actually want to fit one, but rather care about
    # having that object's fit method called exactly once.
    estimator = mocker.MagicMock()
    constraints = Constraints()

    # ExponentiatedGradient pickles and unpickles the estimator, which isn't possible for the mock
    # object, so we mock that process as well. It sets the result from pickle.loads as the
    # estimator, so we can simply overwrite the return value to be our mocked estimator object.
    mocker.patch('pickle.dumps')
    pickle.loads = mocker.MagicMock(return_value=estimator)

    lagrangian = _Lagrangian(X, A, y, estimator, constraints, eps, 1/eps)

    # set up initial lambda vector based on a 0-initialized theta
    theta = pd.Series(0, lagrangian.constraints.index)
    lambda_vec = np.exp(theta) / (1 + np.exp(theta).sum())

    signed_weights = lagrangian.obj.signed_weights() + \
        lagrangian.constraints.signed_weights(lambda_vec)
    redY = 1 * (signed_weights > 0)
    redW = signed_weights.abs()
    redW = y.shape[0] * redW / redW.sum()

    _ = lagrangian._call_oracle(lambda_vec)

    # Ideally we'd prefer calling assert_called_once_with(args) but that is not compatible with
    # pandas data structures.
    assert len(estimator.method_calls) == 1
    name, args, kwargs = estimator.method_calls[0]
    assert name == 'fit'
    assert len(args) == 2
    assert len(kwargs) == 1
    assert (args[0] == X).all().all()
    assert (args[1] == redY).all()
    assert (kwargs['sample_weight'] == redW).all()
    assert lagrangian.n_oracle_calls_ == 1
    assert len(lagrangian.oracle_execution_times_) == 1
