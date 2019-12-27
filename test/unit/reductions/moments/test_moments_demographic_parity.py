# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

from fairlearn.reductions import DemographicParity


def simple_threshold_data(number_a0, number_a1,
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

    X = pd.DataFrame({"credit_score_feature": score_feature,
                      "example_sensitive_feature": A})
    return X, Y, A


def test_construct_and_load():
    dp = DemographicParity()

    assert dp.short_name == "DemographicParity"

    num_samples_a0 = 5
    num_samples_a1 = 7

    a0_threshold = 0.2
    a1_threshold = 0.7

    a0_label = 2
    a1_label = 3

    X, Y, A = simple_threshold_data(num_samples_a0, num_samples_a1,
                                    a0_threshold, a1_threshold,
                                    a0_label, a1_label)

    dp.load_data(X, Y, sensitive_features=A)
    assert dp.data_loaded
    assert dp.n == num_samples_a0 + num_samples_a1
    print("==========================")
    print(dp.tags)
    assert dp.tags['label'].equals(pd.Series(Y))
    assert dp.tags['group_id'].equals(pd.Series(A))
