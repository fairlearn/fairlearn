# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from fairlearn.reductions import GridSearch
from fairlearn.reductions import GroupLossMoment
from fairlearn.reductions import SquareLoss, ZeroOneLoss

import copy
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _simple_regression_data(number_a0, number_a1,
                            a0_factor, a1_factor,
                            a0_label, a1_label):

    a0s = np.full(number_a0, a0_label)
    a1s = np.full(number_a1, a1_label)

    a0_scores = np.linspace(0, 1, number_a0)
    a1_scores = np.linspace(0, 1, number_a1)
    score_feature = np.concatenate((a0_scores, a1_scores), axis=None)

    A = np.concatenate((a0s, a1s), axis=None)

    Y_a0 = a0_factor * a0_scores
    Y_a1 = a1_factor * a1_scores

    Y = np.concatenate((Y_a0, Y_a1), axis=None)

    X = pd.DataFrame({"actual_feature": score_feature,
                      "sensitive_features": A,
                      "constant_ones_feature": np.ones(len(Y))})
    return X, Y, A


def test_bgl_unfair():
    a0_count = 5
    a1_count = 7

    a0_label = 2
    a1_label = 3

    a0_factor = 1
    a1_factor = 16

    X, Y, A = _simple_regression_data(a0_count, a1_count,
                                      a0_factor, a1_factor,
                                      a0_label, a1_label)

    bgl_square_loss = GroupLossMoment(SquareLoss(-np.inf, np.inf))
    target = GridSearch(LinearRegression(),
                        constraints=bgl_square_loss,
                        grid_size=7)

    target.fit(X, Y, sensitive_features=A)

    assert len(target.all_results) == 7

    test_X = pd.DataFrame({"actual_feature": [0.2, 0.7],
                           "sensitive_features": [a0_label, a1_label],
                           "constant_ones_feature": [1, 1]})

    best_predict = target.predict(test_X)
    assert np.allclose([-1.91764706,  9.61176471], best_predict)

    all_predict = [r.predictor.predict(test_X) for r in target.all_results]
    assert np.allclose([[3.2, 11.2],
                        [-3.47346939, 10.64897959],
                        [-2.68, 10.12],
                        [-1.91764706, 9.61176471],
                        [-1.18461538,  9.12307692],
                        [-0.47924528,  8.65283019],
                        [0.2, 0.7]],
                       all_predict)


def test_bgl_unmitigated_same():
    a0_count = 4
    a1_count = 4

    a0_label = 2
    a1_label = 3

    a0_factor = 1
    a1_factor = 16

    X, y, A = _simple_regression_data(a0_count, a1_count,
                                      a0_factor, a1_factor,
                                      a0_label, a1_label)

    estimator = LinearRegression()

    unmitigated_estimator = copy.deepcopy(estimator)
    unmitigated_estimator.fit(X, y)

    # Do the grid search with a zero Lagrange multiplier
    idx = pd.Int64Index(sorted([a0_label, a1_label]))
    lagrange_balanced_series = pd.Series([1.0, 1.0], index=idx)
    grid_df = pd.DataFrame(lagrange_balanced_series)

    target = GridSearch(estimator,
                        constraints=GroupLossMoment(ZeroOneLoss()),
                        grid=grid_df)
    target.fit(X, y, sensitive_features=A)

    raw_coef = unmitigated_estimator.coef_
    gs_coef = target.best_result.predictor.coef_
    # Can't quite get exact match, but this should be very close
    assert np.allclose(raw_coef, gs_coef, rtol=1e-10, atol=1e-7)


def test_bgl_lagrange_specifications():
    a0_count = 13
    a1_count = 4

    a0_label = 5
    a1_label = 3

    a0_factor = 1
    a1_factor = 16

    X, y, A = _simple_regression_data(a0_count, a1_count,
                                      a0_factor, a1_factor,
                                      a0_label, a1_label)

    estimator = LinearRegression()

    # Do the grid search with a zero Lagrange multiplier
    idx = pd.Int64Index(sorted([a0_label, a1_label]))
    l0_series = pd.Series([2.0, 0.0], index=idx)
    l1_series = pd.Series([1.5, 0.5], index=idx)
    l2_series = pd.Series([1.0, 1.0], index=idx)
    l3_series = pd.Series([0.5, 1.5], index=idx)
    l4_series = pd.Series([0.0, 2.0], index=idx)
    grid_df = pd.concat([l0_series,
                         l1_series,
                         l2_series,
                         l3_series,
                         l4_series],
                        axis=1)

    target1 = GridSearch(copy.deepcopy(estimator),
                         constraints=GroupLossMoment(ZeroOneLoss()),
                         grid_size=5)

    target2 = GridSearch(copy.deepcopy(estimator),
                         constraints=GroupLossMoment(ZeroOneLoss()),
                         grid=grid_df)

    tradeoffs = [0, 0.25, 0.5, 0.75, 1]

    target1.fit(X, y, sensitive_features=A)
    target2.fit(X, y, sensitive_features=A)

    assert len(target1.all_results) == len(tradeoffs)
    assert len(target2.all_results) == len(tradeoffs)

    # Check we generated the same multipliers
    for i in range(len(tradeoffs)):
        lm1 = target1.all_results[i].lambda_vec
        lm2 = target2.all_results[i].lambda_vec
        assert lm1.equals(lm2)

    # Check the models are the same
    for i in range(len(tradeoffs)):
        coef1 = target1.all_results[i].predictor.coef_
        coef2 = target2.all_results[i].predictor.coef_
        assert np.array_equal(coef1, coef2)
