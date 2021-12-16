# Copyright (c) Fairlearn contributors.
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
    a1 = 'A'
    a2 = 'B'
    a3 = 'C'
    a4 = 'D'
    A = pd.DataFrame([[a1, a3, a1 + a3], [a1, a3, a1 + a3],
                      [a2, a3, a2 + a3], [a2, a3, a2 + a3], [a2, a3, a2 + a3], [a2, a3, a2 + a3],
                      [a2, a4, a2 + a4], [a2, a4, a2 + a4], [a2, a4, a2 + a4], [a2, a4, a2 + a4],
                      [a2, a4, a2 + a4], [a2, a4, a2 + a4], [a2, a4, a2 + a4], [a2, a4, a2 + a4],
                      [a1, a4, a1 + a4], [a1, a4, a1 + a4]],
                     columns=['SF1', 'SF2', 'SF1+2'])

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

    A_combined = A.loc[:, 'SF1+2']
    A_multiple = A.loc[:, ['SF1', 'SF2']]

    A_test_combined = A_test.loc[:, 'SF1+2']
    A_test_multiple = A_test.loc[:, ['SF1', 'SF2']]

    threshold_dict_combined = {'AC': .36, 'BC': .43, 'BD': ('>', .4)}
    threshold_dict_multiple = {('A', 'C'): .36, ('B', 'C'): .43, ('B', 'D'): .4}

    expected_y = pd.Series([1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0,
                           1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0])

    several_tests = [
        # test for predict_method = 'predict'
        {'estimator': LinearRegression(),
         'predict_method': 'predict',
         'prefit': True,
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': threshold_dict_combined
         },
        # test for predict_method = 'predict_proba', and prefit = False
        {'estimator': LogisticRegression(),
         'predict_method': 'predict_proba',
         'prefit': False,
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': threshold_dict_combined
         },
        # test for Pipeline
        {'estimator': Pipeline(steps=[
            ("logistic_regression", LogisticRegression())]),
         'predict_method': 'predict_proba',
         'prefit': True,
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': threshold_dict_combined
         },
        # test for multiple sensitive features
        {'estimator': LogisticRegression(),
         'predict_method': 'predict_proba',
         'prefit': False,
         'A': A_multiple,
         'A_test': A_test_multiple,
         'threshold_dict': threshold_dict_multiple
         }
    ]

    for test in several_tests:
        estimator = test['estimator']
        if test['prefit']:
            estimator.fit(X, y)
        thresholder = Thresholder(estimator=estimator,
                                  threshold_dict=test['threshold_dict'],
                                  prefit=test['prefit'],
                                  predict_method=test['predict_method'],
                                  default_threshold=.465)

        thresholder.fit(X, y, sensitive_features=test['A'])
        outputted_y = thresholder.predict(
            X_test, sensitive_features=test['A_test'])

        assert (np.array_equal(outputted_y, expected_y))
