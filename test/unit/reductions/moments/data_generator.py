# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd


def simple_binary_threshold_data(number_a0, number_a1,
                                 a0_threshold, a1_threshold,
                                 a0_label, a1_label):
    """Generate some simple (biased) thresholded data.

    This is based on a rigged credit scoring scenario, with two
    classes in the sensitive feature, denoted by a0 and a1. The
    specified number of samples for each are uniformly generated in
    the range zero to one, but the threshold for each class to get
    a positive result in the Y array can be set differently for each.

    The uniformly distributed score is set in the 'example_feature'
    of the resultant X array; the sensitive feature is included in
    this array under the label 'example_sensitive_feature'

    :param number_a0: Number of samples to generate with the label a0
    :type number_a0: int

    :param number_a1: Number of samples to generate with the label a1
    :type number_a1: int

    :param a0_threshold: Threshold value for samples with label a0 to
        get a result of '1' in the Y array
    :type a0_threshold: float

    :param a1_threshold: Threshold value for samples with label a1 to
        get a result of '1' in the Y array
    :type a1_threshold: float

    :param a0_label: The label value for the a0 class
    :type a0_label: int, string

    :param a1_label: The label value for the a1 class
    :type a1_label: int, string

    Returns X, Y, A:
        X is the feature array (containing the uniformly distributed feature
        and the sensitive feature labels)
        Y is the classification array (either 0 or 1)
        A is the array of sensitive features (which will duplicate a column
        in the X array)
    """
    a0s = np.full(number_a0, a0_label)
    a1s = np.full(number_a1, a1_label)

    a0_scores = np.linspace(0, 1, number_a0)
    a1_scores = np.linspace(0, 1, number_a1)
    score_feature = np.concatenate((a0_scores, a1_scores), axis=None)

    A = np.concatenate((a0s, a1s), axis=None)

    Y_a0 = [x > a0_threshold for x in a0_scores]
    Y_a1 = [x > a1_threshold for x in a1_scores]

    Y = np.concatenate((Y_a0, Y_a1), axis=None)

    X = pd.DataFrame({"example_feature": score_feature,
                      "example_sensitive_feature": A})
    return X, Y, A
