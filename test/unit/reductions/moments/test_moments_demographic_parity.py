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

    num_samples_a0 = 10
    num_samples_a1 = 30
    num_samples = num_samples_a0 + num_samples_a1

    a0_threshold = 0.2
    a1_threshold = 0.7

    a0_label = 2
    a1_label = 3

    X, Y, A = simple_threshold_data(num_samples_a0, num_samples_a1,
                                    a0_threshold, a1_threshold,
                                    a0_label, a1_label)

    # Load up the (rigged) data
    dp.load_data(X, Y, sensitive_features=A)
    assert dp.data_loaded
    assert dp.n == num_samples_a0 + num_samples_a1

    # Examine the tags DF
    assert dp.tags['label'].equals(pd.Series(Y))
    assert dp.tags['group_id'].equals(pd.Series(A))
    assert dp.tags['event'].map(lambda x: x == 'all').all()

    # Examine the prob_event DF
    # There's only the 'all' event and everything belongs to it
    assert len(dp.prob_event.index) == 1
    assert dp.prob_event.loc['all'] == 1

    # Examine the prob_group_event DF
    # There's only an 'all' event but this records the fractions
    # of each label in the population
    assert len(dp.prob_group_event.index) == 2
    assert dp.prob_group_event.loc[('all', a0_label)] == num_samples_a0 / num_samples
    assert dp.prob_group_event.loc[('all', a1_label)] == num_samples_a1 / num_samples

    # Examine the neg_basis DF
    # This is obviously looking at the \lambda_{-} values in some way
    assert len(dp.neg_basis.index) == 4
    assert dp.neg_basis[0]['+', 'all', a0_label] == 0
    assert dp.neg_basis[0]['+', 'all', a1_label] == 0
    assert dp.neg_basis[0]['-', 'all', a0_label] == 1
    assert dp.neg_basis[0]['-', 'all', a1_label] == 0

    # Examine the pos_basis DF
    # This is looking at the \lambda_{+} values and picking out the
    # one associated with the first label
    assert len(dp.pos_basis.index) == 4
    assert dp.pos_basis[0]['+', 'all', a0_label] == 1
    assert dp.pos_basis[0]['+', 'all', a1_label] == 0
    assert dp.pos_basis[0]['-', 'all', a0_label] == 0
    assert dp.pos_basis[0]['-', 'all', a1_label] == 0

    print(dp.neg_basis_present)
    assert False
