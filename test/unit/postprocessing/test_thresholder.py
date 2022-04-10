# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import warnings

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from fairlearn.postprocessing import Thresholder


def test_thresholder():
    warnings.filterwarnings("error")

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

    # For testing if correct error pops up
    A_test_unseen_sf = pd.concat([
        pd.DataFrame([[a1, a3, a1 + a3], [a1, a3, a1 + 'F'],
                      [a2, a3, a2 + a3], [a2, a3, a2 + a3],
                      [a2, a4, a2 + a4], [a2, a4, a2 + a4],
                      [a1, a4, a1 + 'E'], [a1, a4, a1 + a4]],
                     columns=['SF1', 'SF2', 'SF1+2']),
        A])
    A_test_unseen_sf_combined = A_test_unseen_sf.loc[:, 'SF1+2']

    A_combined = A.loc[:, 'SF1+2']
    A_multiple = A.loc[:, ['SF1', 'SF2']]

    A_test_combined = A_test.loc[:, 'SF1+2']
    A_test_multiple = A_test.loc[:, ['SF1', 'SF2']]

    threshold_dict_combined = {'AC': .36, 'BC': .43, 'BD': ('>', .4)}
    threshold_dict_multiple = {('A', 'C'): .36, ('B', 'C'): .43, ('B', 'D'): .4}
    threshold_dict_dec_func = {'AC': -0.4, 'BC': -0.1, 'BD': ('<', 0.1)}

    expected_y = pd.Series([1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0,
                           1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0])
    expected_y_def_threshold_auto = pd.Series([1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0,
                                               1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0])
    expected_y_dec_func = pd.Series(
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0])

    several_tests = [
        ########################## Test correct output ##########################

        # Test for predict_method = 'predict'
        {'estimator': LinearRegression(),
         'predict_method': 'predict',
         'prefit': True,
         'default_threshold': 0.465,
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': threshold_dict_combined,
         'expected_y': expected_y
         },

        # predict_method = 'auto' should have same output
        {'estimator': LinearRegression(),
         'predict_method': 'auto',
         'prefit': True,
         'default_threshold': 0.465,
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': threshold_dict_combined,
         'expected_y': expected_y
         },

        # Test prefit = False
        {'estimator': LinearRegression(),
         'predict_method': 'auto',
         'prefit': False,
         'default_threshold': 0.465,
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': threshold_dict_combined,
         'expected_y': expected_y
         },

        # Test for predict_method = 'predict_proba'
        {'estimator': LogisticRegression(),
         'predict_method': 'predict_proba',
         'prefit': True,
         'default_threshold': 0.465,
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': threshold_dict_combined,
         'expected_y': expected_y
         },

        # For LogisticRegression, predict_method = 'auto' should
        # produce the same result as 'predict_proba'
        {'estimator': LogisticRegression(),
         'predict_method': 'auto',
         'prefit': True,
         'default_threshold': 0.465,
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': threshold_dict_combined,
         'expected_y': expected_y
         },

        # Test default_threshold = 'auto' with decision_method = 'predict'
        {'estimator': LinearRegression(),
         'predict_method': 'predict',
         'prefit': True,
         'default_threshold': 'auto',
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': threshold_dict_combined,
         'expected_y': expected_y_def_threshold_auto
         },

        # Test default_threshold = 'auto' for decision_method = 'predict_proba'
        {'estimator': LogisticRegression(),
         'predict_method': 'predict_proba',
         'prefit': False,
         'default_threshold': 'auto',
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': threshold_dict_combined,
         'expected_y': expected_y_def_threshold_auto
         },

        # Test for predict_method = 'decision_function', with default_threshold = 'auto'
        {'estimator': LogisticRegression(),
         'predict_method': 'decision_function',
         'prefit': True,
         'default_threshold': 'auto',
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': threshold_dict_dec_func,
         'expected_y': expected_y_dec_func
         },

        # Test for Pipeline
        {'estimator': Pipeline(steps=[
            ("logistic_regression", LogisticRegression())]),
         'predict_method': 'predict_proba',
         'prefit': True,
         'default_threshold': 0.465,
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': threshold_dict_combined,
         'expected_y': expected_y
         },

        # Test for multiple sensitive features
        {'estimator': LogisticRegression(),
         'predict_method': 'auto',
         'prefit': False,
         'default_threshold': 0.465,
         'A': A_multiple,
         'A_test': A_test_multiple,
         'threshold_dict': threshold_dict_multiple,
         'expected_y': expected_y
         },

        ########################## Test correct error messages ##########################

        # Threshold specified incorrectly in threshold_dict -> type 'int'
        {'estimator': LinearRegression(),
         'predict_method': 'predict',
         'prefit': True,
         'default_threshold': 0.465,
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': {'AC': .36, 'BC': .43, 'BD': ('>', .4), 'AD': 4},
         'expected_y': expected_y,
         'expected_message': "All specified thresholds should be of type 'float' or 'tuple', " +\
         "but 4 is of type 'int'"
         },

        # Threshold specified incorrectly in threshold_dict -> type (float, '<')
        # instead of other way around
        {'estimator': LinearRegression(),
         'predict_method': 'predict',
         'prefit': True,
         'default_threshold': 0.465,
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': {'AC': .36, 'BC': .43, 'BD': ('>', .4), 'AD': (0.3, '>')},
         'expected_y': expected_y,
         'expected_message': "The operator of a specified threshold operation should " +\
         "be either '>' or '<'. However, for (0.3, '>') it is 0.3." +\
         " The threshold should be of type 'float', " +\
         "however > is of type 'str'."
         },

        # Default threshold specified incorrectly
        # -> not a tuple or float
        {'estimator': LinearRegression(),
         'predict_method': 'predict',
         'prefit': True,
         'default_threshold': '0.465',
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': threshold_dict_combined,
         'expected_y': expected_y,
         'expected_message': "All specified thresholds should be of type 'float' or 'tuple', " +\
         "but 0.465 is of type 'str'"
         },

        # predict_method incorrect
        {'estimator': LinearRegression(),
         'predict_method': 'just_some_made_up_method',
         'prefit': True,
         'default_threshold': 0.465,
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': threshold_dict_combined,
         'expected_y': expected_y,
         'expected_message': "'predict_method' should be any of 'auto'" +\
         ", 'predict_proba', 'decision_function', 'predict', but is 'just_some_made_up_method'"
         },

        # Threshold dict keys not the same type
        {'estimator': LinearRegression(),
         'predict_method': 'predict',
         'prefit': True,
         'default_threshold': 0.465,
         'A': A_combined,
         'A_test': A_test_combined,
         'threshold_dict': {'AC': .36, 'BC': .43, 'BD': ('>', .4), 3: .5},
         'expected_y': expected_y,
         'expected_message': "Not all the keys of 'threshold_dict' are of the same type. " +\
         "AC is of type 'str', while 3 is of type 'int'. " +\
         "Please make sure that all keys are of the same type."
         },

        # SF seen at predict time but not at fit time
        {'estimator': LinearRegression(),
         'predict_method': 'predict',
         'prefit': True,
         'default_threshold': 0.465,
         'A': A_combined,
         'A_test': A_test_unseen_sf_combined,
         'threshold_dict': threshold_dict_combined,
         'expected_y': expected_y,
         'expected_message': "The following groups are provided at predict time, " +\
         "but were not observed at fit time: AF AE"
         }
    ]

    # Perform all tests
    for test in several_tests:

        estimator = test['estimator']
        if test['prefit']:
            estimator.fit(X, y)

        try:
            thresholder = Thresholder(estimator=estimator,
                                      threshold_dict=test['threshold_dict'],
                                      prefit=test['prefit'],
                                      predict_method=test['predict_method'],
                                      default_threshold=test['default_threshold'])

            thresholder.fit(X, y, sensitive_features=test['A'])
            outputted_y = thresholder.predict(
                X_test, sensitive_features=test['A_test'])

            assert (np.array_equal(outputted_y, test['expected_y']))

        except (Exception, Warning) as message:
            assert(str(message) == test['expected_message'])
