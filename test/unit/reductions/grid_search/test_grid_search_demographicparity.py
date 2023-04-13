# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import copy
from test.unit.reductions.grid_search.utilities import assert_n_grid_search_results

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from fairlearn.reductions import DemographicParity, GridSearch


def _simple_threshold_data(
    number_a0,
    number_a1,
    a0_threshold,
    a1_threshold,
    a0_label,
    a1_label,
    A_two_dim=False,
):
    a0s = np.full(number_a0, a0_label)
    a1s = np.full(number_a1, a1_label)

    a0_scores = np.linspace(0, 1, number_a0)
    a1_scores = np.linspace(0, 1, number_a1)
    score_feature = np.concatenate((a0_scores, a1_scores), axis=None)

    A = np.concatenate((a0s, a1s), axis=None)

    Y_a0 = [x > a0_threshold for x in a0_scores]
    Y_a1 = [x > a1_threshold for x in a1_scores]

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
def test_demographicparity_fair_uneven_populations(A_two_dim):
    # Variant of test_demographicparity_already_fair, which has unequal
    # populations in the two classes
    # Also allow the threshold to be adjustable

    score_threshold = 0.625

    number_a0 = 4
    number_a1 = 4

    a0_label = 17
    a1_label = 37

    grid_size = 11

    X, Y, A = _simple_threshold_data(
        number_a0,
        number_a1,
        score_threshold,
        score_threshold,
        a0_label,
        a1_label,
        A_two_dim,
    )

    grid_search = GridSearch(
        LogisticRegression(solver="liblinear", fit_intercept=True),
        constraints=DemographicParity(),
        grid_size=grid_size,
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

    sample_results = grid_search.predict(test_X)
    assert np.array_equal(sample_results, [0, 1])

    sample_proba = grid_search.predict_proba(test_X)
    assert np.allclose(
        sample_proba, [[0.53748641, 0.46251359], [0.46688736, 0.53311264]]
    )

    sample_results = grid_search.predictors_[0].predict(test_X)
    assert np.array_equal(sample_results, [1, 0])


@pytest.mark.parametrize("A_two_dim", [False])
@pytest.mark.parametrize("offset", [[0, 0.2, 0, 0]])
def test_demographicparity_fair_uneven_populations_with_grid_offset(A_two_dim, offset):
    # Grid of Lagrangian multipliers has some initial offset

    score_threshold = 0.625

    number_a0 = 4
    number_a1 = 4

    a0_label = 17
    a1_label = 37

    grid_size = 11
    iterables = [["+", "-"], ["all"], [a0_label, a1_label]]
    midx = pd.MultiIndex.from_product(iterables, names=["sign", "event", "group_id"])
    grid_offset = pd.Series(offset, index=midx)

    X, Y, A = _simple_threshold_data(
        number_a0,
        number_a1,
        score_threshold,
        score_threshold,
        a0_label,
        a1_label,
        A_two_dim,
    )

    grid_search = GridSearch(
        LogisticRegression(solver="liblinear", fit_intercept=True),
        constraints=DemographicParity(),
        grid_size=grid_size,
        grid_offset=grid_offset,
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

    sample_results = grid_search.predict(test_X)
    assert np.array_equal(sample_results, [0, 1])

    sample_proba = grid_search.predict_proba(test_X)
    assert np.allclose(
        sample_proba, [[0.55069845, 0.44930155], [0.41546008, 0.58453992]]
    )

    sample_results = grid_search.predictors_[0].predict(test_X)
    assert np.array_equal(sample_results, [1, 0])


# TODO: try with two-dimensional A, but liblinear solver has issues.
@pytest.mark.parametrize("A_two_dim", [False])
def test_lambda_vec_zero_unchanged_model(A_two_dim):
    score_threshold = 0.6

    number_a0 = 64
    number_a1 = 24

    a0_label = 7
    a1_label = 22

    X, y, A = _simple_threshold_data(
        number_a0,
        number_a1,
        score_threshold,
        score_threshold,
        a0_label,
        a1_label,
        A_two_dim,
    )

    estimator = LogisticRegression(
        solver="liblinear", fit_intercept=True, random_state=97
    )

    # Train an unmitigated estimator
    unmitigated_estimator = copy.deepcopy(estimator)
    unmitigated_estimator.fit(X, y)

    # Do the grid search with a zero Lagrange multiplier
    iterables = [["+", "-"], ["all"], [a0_label, a1_label]]
    midx = pd.MultiIndex.from_product(iterables, names=["sign", "event", "group_id"])
    lagrange_zero_series = pd.Series(np.zeros(4), index=midx)
    grid_df = pd.DataFrame(lagrange_zero_series)

    grid_search = GridSearch(estimator, constraints=DemographicParity(), grid=grid_df)
    grid_search.fit(X, y, sensitive_features=A)
    assert_n_grid_search_results(1, grid_search)

    # Check coefficients
    gs_coeff = grid_search.predictors_[grid_search.best_idx_].coef_
    um_coeff = unmitigated_estimator.coef_
    assert np.array_equal(gs_coeff, um_coeff)


# TODO: try with two-dimensional A, but liblinear solver has issues.
@pytest.mark.parametrize("A_two_dim", [False])
def test_can_specify_and_generate_lambda_vecs(A_two_dim):
    score_threshold = 0.4

    number_a0 = 32
    number_a1 = 24

    a0_label = 11
    a1_label = 3

    X, y, A = _simple_threshold_data(
        number_a0, number_a1, score_threshold, score_threshold, a0_label, a1_label
    )

    estimator = LogisticRegression(
        solver="liblinear", fit_intercept=True, random_state=97
    )

    iterables = [["+", "-"], ["all"], sorted([a0_label, a1_label])]
    midx = pd.MultiIndex.from_product(iterables, names=["sign", "event", "group_id"])
    lagrange_negative_series = pd.Series([0.0, 0.0, 0.0, 2.0], index=midx)
    lagrange_zero_series = pd.Series(np.zeros(4), index=midx)
    lagrange_positive_series = pd.Series([0.0, 2.0, 0.0, 0.0], index=midx)
    grid_df = pd.concat(
        [lagrange_negative_series, lagrange_zero_series, lagrange_positive_series],
        axis=1,
    )

    grid_search1 = GridSearch(
        copy.deepcopy(estimator), constraints=DemographicParity(), grid_size=3
    )

    grid_search2 = GridSearch(
        copy.deepcopy(estimator), constraints=DemographicParity(), grid=grid_df
    )

    # Try both ways of specifying the Lagrange multipliers
    grid_search2.fit(X, y, sensitive_features=A)
    grid_search1.fit(X, y, sensitive_features=A)

    assert_n_grid_search_results(3, grid_search1)
    assert_n_grid_search_results(3, grid_search2)

    # Check we generated the same multipliers
    for i in range(3):
        lm1 = grid_search1.lambda_vecs_[i]
        lm2 = grid_search2.lambda_vecs_[i]
        assert lm1.equals(lm2)

    # Check the models are the same
    for i in range(3):
        coef1 = grid_search1.predictors_[i].coef_
        coef2 = grid_search2.predictors_[i].coef_
        assert np.array_equal(coef1, coef2)


def test_single_y_class():
    # Setup with data designed to result in "all single class"
    # at some point in the grid
    X_dict = {"c": [0, 1, 4, 1, 5, 1, 6, 0, 2, 4], "d": [1, 5, 1, 6, 2, 3, 5, 1, 5, 2]}
    X = pd.DataFrame(X_dict)

    # Set y to a constant
    y_val = 1
    y = np.full(10, y_val)
    A = ["a", "a", "a", "a", "a", "a", "a", "a", "a", "b"]

    estimator = LogisticRegression(
        solver="liblinear", fit_intercept=True, random_state=97
    )

    grid_search = GridSearch(
        copy.deepcopy(estimator),
        constraints=DemographicParity(),
        grid_size=3,
        grid_limit=0.1,
        grid_offset=5,
    )

    # We want to avoid an exception on the following line
    grid_search.fit(X, y, sensitive_features=A)

    # Check all predictors are DummyClassifiers
    test_X_dict = {"c": [134534, 27381], "d": [1923, 14123]}
    test_X = pd.DataFrame(test_X_dict)
    for p in grid_search.predictors_:
        assert isinstance(p, DummyClassifier)
        assert np.array_equal(p.predict(test_X), [y_val, y_val])
