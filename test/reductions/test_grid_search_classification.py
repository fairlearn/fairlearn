# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from fairlearn.metrics import DemographicParity
from fairlearn.reductions import GridSearch
from fairlearn.reductions.grid_search.simple_quality_metrics import SimpleClassificationQualityMetric  # noqa: E501

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
                      "protected_attribute_feature": A,
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

    target = GridSearch(LogisticRegression(),
                        fairness_metric=DemographicParity(),
                        quality_metric=SimpleClassificationQualityMetric(),
                        number_of_lagrange_multipliers=11)

    target.fit(X, Y, A)
    assert len(target.all_models) == 11

    test_X = pd.DataFrame({"actual_feature": [0.2, 0.7],
                           "protected_attribute_feature": [a0_label, a1_label],
                           "constant_ones_feature": [1, 1]})

    sample_results = target.predict(test_X)
    sample_proba = target.predict_proba(test_X)
    assert np.allclose(sample_proba, [[0.53748641, 0.46251359], [0.46688736, 0.53311264]])

    sample_results = target.all_models[0]["model"].predict(test_X)
    assert np.array_equal(sample_results, [1, 0])

    all_results = target.posterior_predict(test_X)
    assert len(all_results) == 11

    all_proba = target.posterior_predict_proba(test_X)
    assert len(all_proba) == 11
