# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.validation import check_is_fitted, column_or_1d


"""
Implements the "reweighing" algorithm from
Data preprocessing techniques for classification
without discrimination by Kamiran and Calders
https://link.springer.com/content/pdf/10.1007%2Fs10115-011-0463-8.pdf
"""


def calculate_weights(Y, A):
    """
    Calculate the weights, according to Alg 3 in the paper.

    Parameters
    ----------
    Y : array
        One-dimensional array-like containing prediction/data to reweight to.
    A : array
        One-dimensional array-like sensitive features.

    Returns
    -------
    W : array
        One-dimensional array-like containing weights
    """
    Y = column_or_1d(check_array(Y, ensure_2d=False))
    A = column_or_1d(check_array(A, ensure_2d=False))
    check_consistent_length(Y, A)
    Ys = np.unique(Y)
    As = np.unique(A)
    # Per y, store whether an item in Y is y.
    Y_is_y = [(Y == y).astype(int) for y in Ys]
    A_is_a = [(A == a).astype(int) for a in As]
    # Per y, store the sum, because we use it multiple times.
    y_n = [np.sum(is_y) for is_y in Y_is_y]
    a_n = [np.sum(is_a) for is_a in A_is_a]
    W = np.zeros(len(Y), dtype=float)
    for i in range(len(Ys)):
        for j in range(len(As)):
            # FIXME: laplace smoothing? Or throw meaningful error if denominator
            # is zero?
            is_both = Y_is_y[i] * A_is_a[j]
            w = (y_n[i] * a_n[j] / (len(Y) * np.sum(is_both)))
            W += w * is_both
    return W


# add_weights_methods = ["fit", "fit_transform"]
# estimator_methods = ["fit", "fit_transform", "predict_proba", "decision_function", "predict", "predict_log_proba"]

class Reweighing(BaseEstimator, MetaEstimatorMixin):
    # TODO meta estimator
    # look at exponentiated gradient
    # write to roman about discussion
    def __init__(self, estimator, *, passthrough_sensitive_features=False):
        self.estimator = estimator
        self.passthrough_sensitive_features = passthrough_sensitive_features

        # TODO: mock all other functions of base-estimator?
        # # These can be extended
        # all_methods = []
        # for method_name in dir(estimator):
        #     try:
        #         if callable(getattr(estimator, method_name)) and method_name in estimator_methods:
        #             all_methods.append(method_name)
        #     except AttributeError:
        #         pass

        # for method_name in all_methods:
        #     def method(*args, **kwargs):
        #         if len(args) > 1:
        #             y = args[1]
        #             sensitive_features = kwargs.get("sensitive_features")#_KW_SENSITIVE_FEATURES)
        #             if sensitive_features is None:
        #                 raise ValueError("oopsie")#_MESSAGE_SENSITIVE_FEATURES_NONE)
        #             if method_name in add_weights_methods:
        #                 W = calculate_weights(y, sensitive_features)
        #                 kwargs['sample_weight'] = W
        #         if passthrough_sensitive_features == False:
        #             del kwargs['sensitive_features']
        #         getattr(self.estimator, method_name)(*args, **kwargs)
        #     setattr(self, method_name, method)

    def fit(self, X, y, **kwargs):
        sensitive_features = kwargs.get("sensitive_features")#_KW_SENSITIVE_FEATURES)
        W = calculate_weights(y, sensitive_features)
        kwargs['sample_weight'] = W
        if self.passthrough_sensitive_features == False:
            del kwargs['sensitive_features']
        self.estimator.fit(X, y, **kwargs)

    def fit_transform(self, X, y, *, sensitive_features, **kwargs):
        sensitive_features = kwargs.get("sensitive_features")#_KW_SENSITIVE_FEATURES)
        W = calculate_weights(y, sensitive_features)
        kwargs['sample_weight'] = W
        if self.passthrough_sensitive_features == False:
            del kwargs['sensitive_features']
        self.estimator.fit_transform(X, y, **kwargs)

from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import fetch_openml
import numpy as np
data = fetch_openml(data_id=1590, as_frame=True)
X = data.data.drop(labels=['sex'], axis=1)
X = pd.get_dummies(X)
Y = (data.target == '>50K') * 1
A = (data.data['sex'] == 'Male') * 1
est = BernoulliNB()
metaest = Reweighing(est, passthrough_sensitive_features=False)
metaest.fit(X, Y, sensitive_features=A)
print(np.mean(est.predict(X) == Y))