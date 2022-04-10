# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from fairlearn.postprocessing import RejectOptionClassifier


def test_thresholder():
    X_train = pd.DataFrame([
        [0, 4], [6, 2], [1, 3], [10, 5], [1, 7], [-2, 1], [3, 10], [14, 5],
        [1, 3], [1, 5], [1, 7], [-5, 9], [3, 13], [7, 1], [-8, 4], [9, 1]])
    y_train = pd.Series([0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0])

    a1 = 'A'
    a2 = 'B'

    A_train = pd.DataFrame([[a1], [a1], [a1], [a1],
                            [a2], [a2], [a1], [a2],
                            [a1], [a2], [a2], [a2],
                            [a1], [a2], [a2], [a1]],
                           columns=['SF'])

    X_test = pd.DataFrame([[5, 4], [7, 2], [0, 3], [1, 2], [-2, 9], [1, 1], [0, 5], [-3, 3]])
    A_test = pd.DataFrame([[a1], [a1], [a2], [a1], [a2], [a2], [a1], [a1]], columns=['SF'])

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    expected_y = pd.Series([0, 0, 0, 1, 0, 0, 1, 1])
    expected_y_switched_label_or_groups = pd.Series([0, 0, 1, 0, 1, 0, 0, 0])
    expected_y_cw_1 = pd.Series([1, 1, 0, 1, 0, 0, 1, 1])
    expexted_y_cw_0 = pd.Series([0, 0, 0, 0, 1, 0, 0, 0])

    tests = [
        #################### Test if output is correct ####################

        # Test same result for prefit True/False
        {'cw': 0.2, 'group_to_upselect': a1, 'group_to_downselect': a2,
            'selection_label': 1, 'prefit': True, 'expected_y': expected_y},
        {'cw': 0.2, 'group_to_upselect': a1, 'group_to_downselect': a2,
            'selection_label': 1, 'prefit': False, 'expected_y': expected_y},

        # Check result switching group to up/down select
        {'cw': 0.2, 'group_to_upselect': a2, 'group_to_downselect': a1, 'selection_label': 1,
            'prefit': True, 'expected_y': expected_y_switched_label_or_groups},

        # Check result switching selection label
        {'cw': 0.2, 'group_to_upselect': a1, 'group_to_downselect': a2, 'selection_label': 0,
            'prefit': True, 'expected_y': expected_y_switched_label_or_groups},

        # Check result if critical_width is at either max (1) or min (0)
        {'cw': 1.0, 'group_to_upselect': a1, 'group_to_downselect': a2, 'selection_label': 1,
            'prefit': True, 'expected_y': expected_y_cw_1},
        {'cw': 0.0, 'group_to_upselect': a1, 'group_to_downselect': a2, 'selection_label': 1,
            'prefit': True, 'expected_y': expexted_y_cw_0},

        #################### Test if correct error messages come up ####################

        # cw is not a float
        {'cw': 1, 'group_to_upselect': a1, 'group_to_downselect': a2,
            'selection_label': 1, 'prefit': True, 'expected_y': expected_y,
            'expected_error': "critical_width should be of type 'float', but is of type 'int'"},

        # cw not in correct range
        {'cw': 1.3, 'group_to_upselect': a1, 'group_to_downselect': a2,
            'selection_label': 1, 'prefit': True, 'expected_y': expected_y,
            'expected_error': "critical_width should be between 0 and 1, but is 1.3"},

        # Group to up/down select not in data
        {'cw': 0.2, 'group_to_upselect': 'C', 'group_to_downselect': a2,
            'selection_label': 1, 'prefit': True, 'expected_y': expected_y,
            'expected_error': "The observed sensitive feature value 'A' does " +\
            "not correspond to the specified values of the group_to_upselect and group_to_downselect: 'C' and 'B'."},

        # Selection label not 0 or 1
        {'cw': 0.2, 'group_to_upselect': a1, 'group_to_downselect': a2,
            'selection_label': 3, 'prefit': True, 'expected_y': expected_y,
            'expected_error': "selection_label should be 0 or 1 (of type 'int'), but is 3"}
    ]

    # Perform all tests
    for test in tests:
        clf = LogisticRegression()
        if test['prefit']:
            clf.fit(X_train, y_train)

        try:
            reject_option_clf = RejectOptionClassifier(estimator=clf,
                                                       critical_width=test['cw'],
                                                       group_to_upselect=test['group_to_upselect'],
                                                       group_to_downselect=test['group_to_downselect'],
                                                       selection_label=test['selection_label'],
                                                       prefit=test['prefit'])

            reject_option_clf.fit(X_train, y_train, sensitive_features=A_train)
            outputted_y = reject_option_clf.predict(X_test, sensitive_features=A_test)

            assert(np.array_equal(outputted_y, test['expected_y']))

        except Exception as error_message:
            assert(str(error_message) == test['expected_error'])
