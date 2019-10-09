# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from fairlearn.metrics import BoundedGroupLoss
from fairlearn.reductions import GridSearch
from fairlearn.reductions.grid_search.simple_quality_metrics import SimpleRegressionQualityMetric
import fairlearn.reductions.moments as moments

import copy
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge


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
                      "aux_data_feature": A,
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

    target = GridSearch(LinearRegression(),
                        disparity_metric=BoundedGroupLoss(),
                        quality_metric=SimpleRegressionQualityMetric())

    target.fit(X, Y, aux_data=A, number_of_lagrange_multipliers=7)

    assert len(target.all_results) == 7

    test_X = pd.DataFrame({"actual_feature": [0.2, 0.7],
                           "aux_data_feature": [a0_label, a1_label],
                           "constant_ones_feature": [1, 1]})

    best_predict = target.predict(test_X)
    assert np.allclose([-1.91764706,  9.61176471], best_predict)

    all_predict = target.posterior_predict(test_X)
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

    target = GridSearch(estimator,
                        disparity_metric=BoundedGroupLoss(),
                        quality_metric=SimpleRegressionQualityMetric())
    # The value 0.5 comes from the counts of a0 and a1
    target.fit(X, y, aux_data=A, lagrange_multipliers=[0.5])

    assert np.array_equal(unmitigated_estimator.coef_, target.best_result.model.coef_)


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
    target1 = GridSearch(copy.deepcopy(estimator),
                         disparity_metric=BoundedGroupLoss(),
                         quality_metric=SimpleRegressionQualityMetric())

    target2 = GridSearch(copy.deepcopy(estimator),
                         disparity_metric=BoundedGroupLoss(),
                         quality_metric=SimpleRegressionQualityMetric())

    tradeoffs = [0, 0.25, 0.5, 0.75, 1]

    target1.fit(X, y, aux_data=A, number_of_lagrange_multipliers=len(tradeoffs))
    target2.fit(X, y, aux_data=A, lagrange_multipliers=tradeoffs)

    assert len(target1.all_results) == len(tradeoffs)
    assert len(target2.all_results) == len(tradeoffs)

    # Check we generated the same multipliers
    for i in range(len(tradeoffs)):
        lm1 = target1.all_results[i].lagrange_multiplier
        lm2 = target2.all_results[i].lagrange_multiplier
        assert lm1 == lm2

    # Check the models are the same
    for i in range(len(tradeoffs)):
        coef1 = target1.all_results[i].model.coef_
        coef2 = target2.all_results[i].model.coef_
        assert np.array_equal(coef1, coef2)


def test_compare_custom_vs_moments():
    a0_count = 13
    a1_count = 37

    a0_label = 2
    a1_label = 3

    a0_factor = 1
    a1_factor = 16

    X, y, A = _simple_regression_data(a0_count, a1_count,
                                      a0_factor, a1_factor,
                                      a0_label, a1_label)

    target1 = GridSearch(Ridge(alpha=1.0),
                         disparity_metric=BoundedGroupLoss(),
                         quality_metric=SimpleRegressionQualityMetric())
    target2 = GridSearch(Ridge(alpha=2.0),
                         disparity_metric=moments.GroupLossMoment(moments.ZeroOneLoss()),
                         quality_metric=SimpleRegressionQualityMetric(),
                         grid_size=11)

    target2.fit(X, y, aux_data=A)
    lm = [r.lagrange_multiplier.iat[0] / 2 for r in target2.all_results]
    target1.fit(X, y, aux_data=A, lagrange_multipliers=lm)

    assert len(target1.all_results) == len(target2.all_results)

    # Check the models are the same
    for i in range(len(target1.all_results)):
        coef1 = target1.all_results[i].model.coef_
        coef2 = target2.all_results[i].model.coef_
        assert np.allclose(coef1, coef2, rtol=1e-2, atol=1e-3)
