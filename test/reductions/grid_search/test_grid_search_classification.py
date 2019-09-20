# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from fairlearn.metrics import DemographicParity
from fairlearn.reductions import GridSearch
from fairlearn.reductions.grid_search.simple_quality_metrics import SimpleClassificationQualityMetric  # noqa: E501

import copy
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _simple_threshold_data(number_a0, number_a1,
                           a0_threshold, a1_threshold,
                           a0_label, a1_label):

    a0s = np.full(number_a0, a0_label)
    a1s = np.full(number_a1, a1_label)

    a0_scores = np.linspace(0, 1, number_a0)
    a1_scores = np.linspace(0, 1, number_a1)
    score_feature = np.concatenate((a0_scores, a1_scores), axis=None)

    A = np.concatenate((a0s, a1s), axis=None)

    Y_a0 = [x > a0_threshold for x in a0_scores]
    Y_a1 = [x > a1_threshold for x in a1_scores]

    Y = np.concatenate((Y_a0, Y_a1), axis=None)

    X = pd.DataFrame({"actual_feature": score_feature,
                      "aux_data_feature": A,
                      "constant_ones_feature": np.ones(len(Y))})
    return X, Y, A


def test_demographicparity_fair_uneven_populations():
    # Variant of test_demographicparity_already_fair, which has unequal
    # populations in the two classes
    # Also allow the threshold to be adjustable

    score_threshold = 0.625

    number_a0 = 4
    number_a1 = 4

    a0_label = 17
    a1_label = 37

    X, Y, A = _simple_threshold_data(number_a0, number_a1,
                                     score_threshold, score_threshold,
                                     a0_label, a1_label)

    target = GridSearch(LogisticRegression(solver='liblinear', fit_intercept=True),
                        disparity_metric=DemographicParity(),
                        quality_metric=SimpleClassificationQualityMetric())

    target.fit(X, Y, aux_data=A,
               number_of_lagrange_multipliers=11)
    assert len(target.all_results) == 11

    test_X = pd.DataFrame({"actual_feature": [0.2, 0.7],
                           "aux_data_feature": [a0_label, a1_label],
                           "constant_ones_feature": [1, 1]})

    sample_results = target.predict(test_X)
    sample_proba = target.predict_proba(test_X)
    assert np.allclose(sample_proba, [[0.53748641, 0.46251359], [0.46688736, 0.53311264]])

    sample_results = target.all_results[0].model.predict(test_X)
    assert np.array_equal(sample_results, [1, 0])

    all_results = target.posterior_predict(test_X)
    assert len(all_results) == 11

    all_proba = target.posterior_predict_proba(test_X)
    assert len(all_proba) == 11


def test_lagrange_multiplier_zero_unchanged_model():
    score_threshold = 0.6

    number_a0 = 64
    number_a1 = 24

    a0_label = 7
    a1_label = 22

    X, y, A = _simple_threshold_data(number_a0, number_a1,
                                     score_threshold, score_threshold,
                                     a0_label, a1_label)

    learner = LogisticRegression(solver='liblinear',
                                 fit_intercept=True,
                                 random_state=97)

    # Train an unmitigated learner
    unmitigated_learner = copy.deepcopy(learner)
    unmitigated_learner.fit(X, y)

    # Do the grid search with a zero Lagrange multiplier
    target = GridSearch(learner,
                        disparity_metric=DemographicParity(),
                        quality_metric=SimpleClassificationQualityMetric())
    target.fit(X, y, aux_data=A, lagrange_multipliers=[0])

    # Check coefficients
    gs_coeff = target.best_result.model.coef_
    um_coeff = unmitigated_learner.coef_
    assert np.array_equal(gs_coeff, um_coeff)


def test_can_specify_and_generate_lagrange_multipliers():
    score_threshold = 0.4

    number_a0 = 32
    number_a1 = 24

    a0_label = 11
    a1_label = 3

    X, y, A = _simple_threshold_data(number_a0, number_a1,
                                     score_threshold, score_threshold,
                                     a0_label, a1_label)

    learner = LogisticRegression(solver='liblinear',
                                 fit_intercept=True,
                                 random_state=97)

    target1 = GridSearch(copy.deepcopy(learner),
                         disparity_metric=DemographicParity(),
                         quality_metric=SimpleClassificationQualityMetric())

    target2 = GridSearch(copy.deepcopy(learner),
                         disparity_metric=DemographicParity(),
                         quality_metric=SimpleClassificationQualityMetric())

    # Note that using integers for my_lagrange causes the test to fail
    my_lagrange = [-2.0, 0, 2.0]

    # Try both ways of specifying the Lagrange multipliers
    target2.fit(X, y, aux_data=A, lagrange_multipliers=my_lagrange)
    target1.fit(X, y, aux_data=A, number_of_lagrange_multipliers=len(my_lagrange))

    assert len(target1.all_results) == len(my_lagrange)
    assert len(target2.all_results) == len(my_lagrange)

    # Check we generated the same multipliers
    for i in range(len(my_lagrange)):
        lm1 = target1.all_results[i].lagrange_multiplier
        lm2 = target2.all_results[i].lagrange_multiplier
        assert lm1 == lm2

    # Check the models are the same
    for i in range(len(my_lagrange)):
        coef1 = target1.all_results[i].model.coef_
        coef2 = target2.all_results[i].model.coef_
        assert np.array_equal(coef1, coef2)
