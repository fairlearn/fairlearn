# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier

from fairlearn.reductions import (
    BoundedGroupLoss,
    DemographicParity,
    EqualizedOdds,
    ErrorRate,
    ErrorRateParity,
    FalsePositiveRateParity,
    LossMoment,
    TruePositiveRateParity,
    ZeroOneLoss,
)
from fairlearn.reductions._exponentiated_gradient._lagrangian import (
    _MESSAGE_BAD_OBJECTIVE,
    _Lagrangian,
)
from fairlearn.reductions._moments.bounded_group_loss import MeanLoss

from .simple_learners import LeastSquaresBinaryClassifierLearner, MockEstimator
from .test_utilities import _get_data

_PRECISION = 1e-6

CLASSIFICATION_CONSTRAINTS = [
    DemographicParity,
    EqualizedOdds,
    TruePositiveRateParity,
    FalsePositiveRateParity,
    ErrorRateParity,
]

REGRESSION_CONSTRAINTS = [BoundedGroupLoss]

ALL_CONSTRAINTS = CLASSIFICATION_CONSTRAINTS + REGRESSION_CONSTRAINTS

ALL_OBJECTIVES = [ErrorRate, MeanLoss]


@pytest.mark.parametrize("eps", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("Constraints", ALL_CONSTRAINTS)
@pytest.mark.parametrize("use_Q_callable", [True, False])
@pytest.mark.parametrize("opt_lambda", [True, False])
def test_lagrangian_eval(eps, Constraints, use_Q_callable, opt_lambda):
    X, y, A = _get_data(A_two_dim=False)
    estimator = LeastSquaresBinaryClassifierLearner()

    if issubclass(Constraints, LossMoment):
        task_type = "regression"
        constraints = Constraints(ZeroOneLoss(), upper_bound=eps)
    else:
        task_type = "classification"
        constraints = Constraints(difference_bound=eps)

    # epsilon (and thereby also B) only affects L_high and L
    B = 1 / eps

    lagrangian = _Lagrangian(
        X=X,
        y=y,
        estimator=estimator,
        constraints=deepcopy(constraints),
        B=B,
        opt_lambda=opt_lambda,
        sensitive_features=A,
    )

    lambda_vec = get_lambda_vec(constraints, X, y, A)

    # call oracle to determine error and gamma and calculate exp
    fitted_estimator = lagrangian._call_oracle(lambda_vec)

    def h(X):
        return fitted_estimator.predict(X)

    best_h_error = lagrangian.obj.gamma(h).iloc[0]
    best_h_gamma = lagrangian.constraints.gamma(h)

    # opt_lambda affects only the calculation of L
    if opt_lambda:
        projected_lambda = constraints.project_lambda(lambda_vec)
        L_expected = (
            best_h_error
            + np.sum(projected_lambda * best_h_gamma)
            - eps * np.sum(projected_lambda)
        )
    else:
        L_expected = (
            best_h_error + np.sum(lambda_vec * best_h_gamma) - eps * np.sum(lambda_vec)
        )

    L_high_expected = best_h_error + B * (best_h_gamma.max() - eps)

    # manually set errors and gammas which would otherwise be done in the best_h step
    lagrangian.errors = pd.Series([best_h_error])
    lagrangian.gammas = pd.Series([best_h_gamma])

    # call _eval to get the desired results L, L_high, gamma, error;
    # _eval is compatible with a callable h or a vector Q
    Q_vec = pd.Series([1.0])
    L, L_high, gamma, error = lagrangian._eval(
        h if use_Q_callable else Q_vec, lambda_vec
    )

    # in this particular example the estimator is always the same
    expected_estimator_weights = {
        "regression": pd.Series({"X1": 0.541252, "X2": 0.454293, "X3": 0.019203}),
        "classification": pd.Series({"X1": 0.538136, "X2": 0.457627, "X3": 0.021186}),
    }
    assert (
        np.isclose(
            fitted_estimator.weights,
            expected_estimator_weights[task_type],
            atol=1.0e-6,
        )
    ).all()

    assert L == pytest.approx(L_expected, abs=_PRECISION)
    assert L_high == pytest.approx(L_high_expected, abs=_PRECISION)
    assert error == 0.25
    assert (gamma == best_h_gamma).all()


@pytest.mark.parametrize("Constraints", ALL_CONSTRAINTS)
@pytest.mark.parametrize("eps", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("estimator", [MockEstimator, BaseEstimator])
def test_call_oracle(Constraints, eps, estimator, mocker):
    X, y, A = _get_data(A_two_dim=False)
    # Using a real estimator here with a mocked `fit` method since we don't actually
    # want to fit one, but rather care about having that object's fit method called exactly once.
    estimator.fit = mocker.MagicMock(name="fit")
    if issubclass(Constraints, LossMoment):
        constraints = Constraints(ZeroOneLoss())
    else:
        constraints = Constraints()

    lagrangian = _Lagrangian(
        X=X,
        y=y,
        estimator=estimator,
        constraints=deepcopy(constraints),
        B=1 / eps,
        sensitive_features=A,
    )

    # Set up initial lambda vector based on a 0-initialized theta and use separate constraints
    # object for it to avoid the dependence on the lagrangian object.
    lambda_vec, new_weights, new_labels = get_lambda_new_weights_and_labels(
        constraints, X, y, A
    )

    _ = lagrangian._call_oracle(lambda_vec)

    # Ideally we'd prefer calling assert_called_once_with(args) but that is not compatible with
    # pandas data structures.
    assert len(estimator.fit.mock_calls) == 1
    _, args, kwargs = estimator.fit.mock_calls[0]
    assert (args[0] == X).all().all()
    assert (args[1] == new_labels).all()
    assert (kwargs["sample_weight"] == new_weights).all()
    assert lagrangian.n_oracle_calls == 1
    assert len(lagrangian.oracle_execution_times) == 1


@pytest.mark.parametrize(
    "Constraints",
    [DemographicParity, EqualizedOdds, ErrorRateParity, BoundedGroupLoss],
)
@pytest.mark.parametrize("eps", [0.001, 0.01, 0.1])
@pytest.mark.parametrize("y_value", [0, 1])
def test_call_oracle_single_y_value(Constraints, eps, y_value, mocker):
    X_dict = {
        "c": [0, 1, 4, 1, 5, 1, 6, 0, 2, 4],
        "d": [1, 5, 1, 6, 2, 3, 5, 1, 5, 2],
    }
    X = pd.DataFrame(X_dict)

    # Try with both possible y values for binary classification to ensure that
    # constraints that focus only on positives or negatives can handle the
    # case where none of the rows apply to them.
    y = pd.Series([y_value] * 10)
    A = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    # We mock the estimator, but we only patch it for pickling
    estimator = mocker.MagicMock()
    if issubclass(Constraints, LossMoment):
        constraints = Constraints(ZeroOneLoss(), upper_bound=eps)
    else:
        constraints = Constraints(difference_bound=eps)

    lagrangian = _Lagrangian(
        X=X,
        y=y,
        estimator=estimator,
        constraints=deepcopy(constraints),
        B=1 / eps,
        sensitive_features=A,
    )

    # Set up initial lambda vector based on a 0-initialized theta and use separate constraints
    # object for it to avoid the dependence on the lagrangian object.
    lambda_vec = get_lambda_vec(constraints, X, y, A)

    test_X_dict = {"c": [10000], "d": [2000000]}
    test_X = pd.DataFrame(test_X_dict)

    result_estimator = lagrangian._call_oracle(lambda_vec)
    assert isinstance(result_estimator, DummyClassifier)
    assert result_estimator.predict(test_X) == y_value
    assert lagrangian.n_oracle_calls_dummy_returned == 1

    # Make sure the mocked estimator wasn't called
    assert len(estimator.method_calls) == 0


@pytest.mark.parametrize("Constraints", ALL_CONSTRAINTS)
@pytest.mark.parametrize("Objective", ALL_OBJECTIVES)
def test_objective_constraints_compatibility(Constraints, Objective):
    X, y, A = _get_data(A_two_dim=False)
    estimator = LeastSquaresBinaryClassifierLearner()

    if issubclass(Constraints, LossMoment):
        constraints = Constraints(ZeroOneLoss())
    else:
        constraints = Constraints()

    if issubclass(Objective, LossMoment):
        objective = Objective(ZeroOneLoss())
    else:
        objective = Objective()

    if objective._moment_type() != constraints._moment_type():
        with pytest.raises(ValueError) as execInfo:
            _ = _Lagrangian(
                X=X,
                y=y,
                estimator=estimator,
                constraints=deepcopy(constraints),
                objective=objective,
                B=1.0,
                sensitive_features=A,
            )
        assert (
            _MESSAGE_BAD_OBJECTIVE.format(
                objective._moment_type(), constraints._moment_type()
            )
            in execInfo.value.args[0]
        )
    else:
        # No exception raised
        _ = _Lagrangian(
            X=X,
            y=y,
            estimator=estimator,
            constraints=deepcopy(constraints),
            objective=objective,
            B=1.0,
            sensitive_features=A,
        )


def get_lambda_vec(constraints, X, y, A):
    # set up initial lambda vector based on a 0-initialized theta
    constraints.load_data(X, y, sensitive_features=A)
    objective = constraints.default_objective()
    objective.load_data(X, y, sensitive_features=A)
    theta = pd.Series(0, constraints.index)
    lambda_vec = np.exp(theta) / (1 + np.exp(theta).sum())
    return lambda_vec


def get_lambda_new_weights_and_labels(constraints, X, y, A):
    lambda_vec = get_lambda_vec(constraints, X, y, A)
    objective = constraints.default_objective()
    objective.load_data(X, y, sensitive_features=A)
    signed_weights = objective.signed_weights() + constraints.signed_weights(lambda_vec)

    if isinstance(constraints, LossMoment):
        redY = y
    else:  # classification
        redY = 1 * (signed_weights > 0)
    redW = signed_weights.abs()
    redW = y.shape[0] * redW / redW.sum()
    return lambda_vec, redW, redY
