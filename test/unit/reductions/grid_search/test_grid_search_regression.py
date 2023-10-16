# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import copy
from test.unit.reductions.grid_search.utilities import assert_n_grid_search_results
from test.unit.utility_functions import logging_all_close

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from fairlearn.reductions import BoundedGroupLoss, GridSearch, SquareLoss, ZeroOneLoss


def _simple_regression_data(
    number_a0, number_a1, a0_factor, a1_factor, a0_label, a1_label, A_two_dim=False
):
    a0s = np.full(number_a0, a0_label)
    a1s = np.full(number_a1, a1_label)

    a0_scores = np.linspace(0, 1, number_a0)
    a1_scores = np.linspace(0, 1, number_a1)
    score_feature = np.concatenate((a0_scores, a1_scores), axis=None)

    A = np.concatenate((a0s, a1s), axis=None)

    Y_a0 = a0_factor * a0_scores
    Y_a1 = a1_factor * a1_scores

    Y = np.concatenate((Y_a0, Y_a1), axis=None)

    X = pd.DataFrame(
        {
            "actual_feature": score_feature,
            "sensitive_features": A,
            "constant_ones_feature": np.ones(len(Y)),
        }
    )

    if A_two_dim:
        A = np.stack((A, A), -1)

    return X, Y, A


@pytest.mark.parametrize("A_two_dim", [False, True])
def test_bgl_unfair(A_two_dim):
    a0_count = 5
    a1_count = 7

    a0_label = 2
    a1_label = 3

    a0_factor = 1
    a1_factor = 16

    grid_size = 7

    X, Y, A = _simple_regression_data(
        a0_count, a1_count, a0_factor, a1_factor, a0_label, a1_label, A_two_dim
    )

    bgl_square_loss = BoundedGroupLoss(SquareLoss(-np.inf, np.inf))
    grid_search = GridSearch(
        LinearRegression(), constraints=bgl_square_loss, grid_size=grid_size
    )

    grid_search.fit(X, Y, sensitive_features=A)

    assert_n_grid_search_results(grid_size, grid_search)

    test_X = pd.DataFrame(
        {
            "actual_feature": [0.2, 0.7],
            "sensitive_features": [a0_label, a1_label],
            "constant_ones_feature": [1, 1],
        }
    )

    best_predict = grid_search.predict(test_X)
    assert np.allclose([-1.91764706, 9.61176471], best_predict)

    all_predict = [predictor.predict(test_X) for predictor in grid_search.predictors_]

    # TODO: investigate where the different outcomes for the first grid point are from, likely
    # due to some ignored data points at the edge resulting in another solution with the same
    # least squares loss (i.e. both solutions acceptable).
    # Reflects https://github.com/fairlearn/fairlearn/issues/265
    assert logging_all_close([[3.2, 11.2]], [all_predict[0]]) or logging_all_close(
        [[3.03010885, 11.2]], [all_predict[0]]
    )

    assert logging_all_close(
        [
            [-3.47346939, 10.64897959],
            [-2.68, 10.12],
            [-1.91764706, 9.61176471],
            [-1.18461538, 9.12307692],
            [-0.47924528, 8.65283019],
            [0.2, 0.7],
        ],
        all_predict[1:],
    )


# TODO: enable two-dimensional A scenarios by investigating issues
# with LinearRegression in this use case
@pytest.mark.parametrize("A_two_dim", [False])
def test_bgl_unmitigated_same(A_two_dim):
    a0_count = 4
    a1_count = 4

    a0_label = 2
    a1_label = 3

    a0_factor = 1
    a1_factor = 16

    X, y, A = _simple_regression_data(
        a0_count, a1_count, a0_factor, a1_factor, a0_label, a1_label, A_two_dim
    )

    estimator = LinearRegression()

    unmitigated_estimator = copy.deepcopy(estimator)
    unmitigated_estimator.fit(X, y)

    # Do the grid search with a zero Lagrange multiplier
    idx = pd.Index(sorted([a0_label, a1_label]))
    lagrange_balanced_series = pd.Series([1.0, 1.0], index=idx)
    grid_df = pd.DataFrame(lagrange_balanced_series)

    grid_search = GridSearch(
        estimator, constraints=BoundedGroupLoss(ZeroOneLoss()), grid=grid_df
    )
    grid_search.fit(X, y, sensitive_features=A)

    raw_coef = unmitigated_estimator.coef_
    gs_coef = grid_search.predictors_[grid_search.best_idx_].coef_
    # Can't quite get exact match, but this should be very close
    assert np.allclose(raw_coef, gs_coef, rtol=1e-10, atol=1e-7)


# TODO: enable two-dimensional A scenarios by investigating issues
# with LinearRegression in this use case
@pytest.mark.parametrize("A_two_dim", [False])
def test_bgl_lagrange_specifications(A_two_dim):
    a0_count = 13
    a1_count = 4

    a0_label = 5
    a1_label = 3

    a0_factor = 1
    a1_factor = 16

    X, y, A = _simple_regression_data(
        a0_count, a1_count, a0_factor, a1_factor, a0_label, a1_label, A_two_dim
    )

    estimator = LinearRegression()

    # Do the grid search with a zero Lagrange multiplier
    idx = pd.Index(sorted([a0_label, a1_label]))
    l0_series = pd.Series([2.0, 0.0], index=idx)
    l1_series = pd.Series([1.5, 0.5], index=idx)
    l2_series = pd.Series([1.0, 1.0], index=idx)
    l3_series = pd.Series([0.5, 1.5], index=idx)
    l4_series = pd.Series([0.0, 2.0], index=idx)
    grid_df = pd.concat([l0_series, l1_series, l2_series, l3_series, l4_series], axis=1)

    grid_search1 = GridSearch(
        copy.deepcopy(estimator),
        constraints=BoundedGroupLoss(ZeroOneLoss()),
        grid_size=5,
    )

    grid_search2 = GridSearch(
        copy.deepcopy(estimator),
        constraints=BoundedGroupLoss(ZeroOneLoss()),
        grid=grid_df,
    )

    tradeoffs = [0, 0.25, 0.5, 0.75, 1]

    grid_search1.fit(X, y, sensitive_features=A)
    grid_search2.fit(X, y, sensitive_features=A)

    assert_n_grid_search_results(len(tradeoffs), grid_search1)
    assert_n_grid_search_results(len(tradeoffs), grid_search2)

    # Check we generated the same multipliers
    for i in range(len(tradeoffs)):
        lm1 = grid_search1.lambda_vecs_[i]
        lm2 = grid_search2.lambda_vecs_[i]
        assert lm1.equals(lm2)

    # Check the models are the same
    for i in range(len(tradeoffs)):
        coef1 = grid_search1.predictors_[i].coef_
        coef2 = grid_search2.predictors_[i].coef_
        assert np.array_equal(coef1, coef2)
