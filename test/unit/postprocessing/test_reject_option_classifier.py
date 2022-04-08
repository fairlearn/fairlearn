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

    # Test output if as expected for prefit is True/False
    for prefit in [True, False]:
        clf = LogisticRegression()
        if prefit:
            clf.fit(X_train, y_train)
        reject_option_clf = RejectOptionClassifier(estimator=clf,
                                                   critical_width=0.2,
                                                   group_to_upselect=a1,
                                                   group_to_downselect=a2,
                                                   prefit=prefit)

        reject_option_clf.fit(X_train, y_train, sensitive_features=A_train)
        outputted_y = reject_option_clf.predict(X_test, sensitive_features=A_test)

        assert(np.array_equal(outputted_y, expected_y))
