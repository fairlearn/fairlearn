# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from fairlearn.postprocessing import Thresholder


def test_thresholder():
    X = pd.DataFrame([
        [0, 4], [6, 2], [1, 3], [10, 5], [1, 7], [-2, 1], [3, 10], [14, 5],
        [1, 3], [1, 5], [1, 7], [-5, 9], [3, 13], [7, 1], [-8, 4], [9, 1]])
    y = pd.Series([0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
    a1 = "A"
    a2 = "B"
    a3 = "C"
    a4 = "D"
    A = pd.DataFrame([[a1, a3, a1 + a3], [a1, a3, a1 + a3],
                      [a2, a3, a2 + a3], [a2, a3, a2 + a3], [a2, a3, a2 + a3], [a2, a3, a2 + a3],
                      [a2, a4, a2 + a4], [a2, a4, a2 + a4], [a2, a4, a2 + a4], [a2, a4, a2 + a4],
                      [a2, a4, a2 + a4], [a2, a4, a2 + a4], [a2, a4, a2 + a4], [a2, a4, a2 + a4],
                      [a1, a4, a1 + a4], [a1, a4, a1 + a4]],
                     columns=['SF1', 'SF2', 'SF1+2'])

    estimator = LogisticRegression()
    estimator.fit(X, y)

    # finish when there are answers about how to deal with multiple sensitive features
    # thresholder_multi = Thresholder(estimator=estimator, threshold_dict=//todo,
    #                                 prefit=True,
    #                                 predict_method='predict_proba')

    threshold_dict_combined = {'AC': .36, 'BC': .43, 'BD': .4, 'AD': .465}
    thresholder_combined = Thresholder(estimator=estimator,
                                       threshold_dict=threshold_dict_combined,
                                       prefit=True,
                                       predict_method='predict_proba')

    X_test = pd.concat([
        pd.DataFrame([[5, 4], [7, 2], [0, 3], [1, 2], [-2, 9], [1, 1], [0, 5], [-3, 3]]),
        X])
    A_test = pd.concat([
        pd.DataFrame([[a1, a3, a1 + a3], [a1, a3, a1 + a3],
                      [a2, a3, a2 + a3], [a2, a3, a2 + a3],
                      [a2, a4, a2 + a4], [a2, a4, a2 + a4],
                      [a1, a4, a1 + a4], [a1, a4, a1 + a4]],
                     columns=['SF1', 'SF2', 'SF1+2']),
        A])

    expected_y = pd.Series([1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0,
                           1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0])

    several_tests = [
        {'estimator': LinearRegression(),
         'predict_method': 'predict',
         'prefit': True
         },
        {'estimator': LogisticRegression(),
         'predict_method': 'predict_proba',
         'prefit': False
         },
        {'estimator': Pipeline(steps=[
            ("logistic_regression", LogisticRegression())]),
         'predict_method': 'predict_proba',
         'prefit': True
         }
    ]

    for test in several_tests:
        estimator = test['estimator']
        if test['prefit']:
            estimator.fit(X, y)
        thresholder_combined = Thresholder(estimator=estimator,
                                           threshold_dict=threshold_dict_combined,
                                           prefit=test['prefit'],
                                           predict_method=test['predict_method'])

        thresholder_combined.fit(X, y)
        outputted_y = thresholder_combined.predict(
            X_test, sensitive_features=A_test.loc[:, 'SF1+2'])

        assert (np.array_equal(outputted_y, expected_y))
