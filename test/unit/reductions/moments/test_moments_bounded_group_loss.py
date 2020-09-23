# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from fairlearn.reductions import BoundedGroupLoss, SquareLoss
from test.unit.reductions.grid_search.utilities import _quick_data


class TestBoundedGroupLoss:

    def test_eps(self):
        X, Y, A = _quick_data()
        eps = 0.01
        self.estimator = LinearRegression().fit(X, Y)
        def predictor(x): return self.estimator.predict(x)
        self.disparity_criterion = BoundedGroupLoss(SquareLoss(-np.inf, np.inf), upper_bound=eps)
        self.disparity_criterion.load_data(X, Y, sensitive_features=A)
        bnd = self.disparity_criterion.bound()
        loss_eps = self.disparity_criterion.gamma(predictor) - bnd
        loss = self.disparity_criterion.gamma(predictor)
        assert(np.all(np.isclose(loss - eps, loss_eps)))

    def test_bgl_gpa_data(self):

        names = ['gender', 'physics', 'biology', 'history', 'second_language',
                 'geography', 'literature', 'portuguese', 'math', 'chemistry', 'gpa']
        data = pd.DataFrame(np.array([[0, 622.6, 491.56, 439.93, 707.64, 663.65, 557.09, 711.37,
                                       731.31, 509.8, 1.33333], [1, 538, 490.58, 406.59, 529.05,
                                                                 532.28, 447.23, 527.58, 379.14,
                                                                 488.64, 2.98333], [1, 455.18, 440,
                                                                                    570.86, 417.54,
                                                                                    453.53, 425.87,
                                                                                    475.63, 476.11,
                                                                                    407.15,
                                                                                    1.97333],
                                      [0, 756.91, 679.62, 531.28, 583.63, 534.42, 521.4, 592.41,
                                       783.76, 588.26, 2.53333], [1, 584.54, 649.84, 637.43,
                                                                  609.06, 670.46, 515.38,
                                                                  572.52, 581.25, 529.04, 1.58667],
                                      [1, 325.99, 466.74, 597.06, 554.43, 535.77, 717.03, 477.6,
                                       503.82, 422.92, 1.66667], [0, 622.6, 587.04, 598.85, 603.32,
                                                                  690.7, 652.86, 533.05, 755.3,
                                                                  628.73, 3.72333], [1, 527.65,
                                                                                     559.99,
                                                                                     758.37,
                                                                                     669.71,
                                                                                     645.62,
                                                                                     648.67,
                                                                                     539.23,
                                                                                     470.78,
                                                                                     486.13,
                                                                                     3.08333]]),
                            columns=names)
        X = data.iloc[:, :-1]
        A = X.gender
        del X['gender']

        X = X.subtract(X.min(axis=0)).divide(X.max(axis=0) - X.min(axis=0))
        assert (np.all(X.max(axis=0) == 1))
        assert (np.all(X.min(axis=0) == 0))

        X['gender'] = A
        Y = data.gpa
        y = data.gpa
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=42)
        A_train = X_train.gender
        A_test = X_test.gender
        del X_train['gender']
        del X_test['gender']

        X_train.reset_index(inplace=True, drop=True)
        X_test.reset_index(inplace=True, drop=True)
        A_train.reset_index(inplace=True, drop=True)
        A_test.reset_index(inplace=True, drop=True)
        y_train.reset_index(inplace=True, drop=True)
        y_test.reset_index(inplace=True, drop=True)

        estimator = LinearRegression().fit(X, Y)
        def predictor(x): return estimator.predict(x)
        eps = 0.05

        disparity_criterion = BoundedGroupLoss(SquareLoss(-np.inf, np.inf), upper_bound=eps)
        disparity_criterion.load_data(X, Y, sensitive_features=A)
        bnd = disparity_criterion.bound()
        loss_eps = disparity_criterion.gamma(predictor)
        loss = disparity_criterion.gamma(predictor)

        assert not np.any(loss - loss_eps)
        assert bnd.shape == loss.shape
