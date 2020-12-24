# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import logging
import math
# import numpy as np
# import pandas as pd
# import scipy.optimize as opt
# from sklearn import clone
# from sklearn.dummy import DummyClassifier
# from time import time
# import inspect
# import resampling


logger = logging.getLogger(__name__)


def resample_fixed_size_deterministic(X, y, weights):
    """ Resampling methods: Deterministic and keep the sample size.

        Parameters
        ----------
        X : {numpy.ndarray, pandas.DataFrame}
            the training features
        sensitive_features : {numpy.ndarray, pandas.Series, pandas.DataFrame, list}
            the sensitive features to use for constraints
        y : {numpy.ndarray, pandas.Series, pandas.DataFrame, list}
            the training labels
        weight : {numpy.ndarray, pandas.Series, pandas.DataFrame, list}
            the resampling weights
        """

    factor = len(X) / sum(weights)
    resampled_X, resampled_y = [], []

    scaled_weight = [weight * factor for weight in weights]
    for i in range(len(X)):
        if scaled_weight[i] >= 1 and len(resampled_X) < len(X):
            add_num = math.floor(scaled_weight[i])
        resampled_X += [X[i]] * add_num
        resampled_y += [y[i]] * add_num
    scaled_weight[i] -= add_num

    remain_slots = len(X) - len(resampled_X)

    if remain_slots > 0:
        scale_tuple = list(enumerate(scaled_weight))
        scale_tuple.sort(key=lambda x: x[1], reverse=True)
        scale_tuple = scale_tuple[:remain_slots]
        for index, scale in scale_tuple:
            resampled_X += [X[index]]
            resampled_y += [y[index]]

        return resampled_X, resampled_y