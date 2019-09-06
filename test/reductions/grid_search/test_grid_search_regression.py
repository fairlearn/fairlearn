# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from fairlearn.metrics import BoundedGroupLoss
from fairlearn.reductions import GridSearch
from fairlearn.reductions.grid_search.simple_quality_metrics import SimpleRegressionQualityMetric

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
                      "protected_attribute_feature": A,
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
                        fairness_metric=BoundedGroupLoss(),
                        quality_metric=SimpleRegressionQualityMetric())

    target.fit(X, Y, protected_attribute=A, number_of_lagrange_multipliers=7)

    assert len(target.all_models) == 7

    test_X = pd.DataFrame({"actual_feature": [0.2, 0.7],
                           "protected_attribute_feature": [a0_label, a1_label],
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
