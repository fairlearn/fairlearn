# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd


class LeastSquaresBinaryClassifierLearner:
    def __init__(self):
        self.weights = None

    def fit(self, X, Y, sample_weight):
        sqrtW = np.sqrt(np.array(sample_weight))
        matX = np.array(X) * sqrtW[:, np.newaxis]
        vecY = Y * sqrtW
        self.lsqinfo = np.linalg.lstsq(matX, vecY, rcond=-1)
        self.weights = pd.Series(self.lsqinfo[0], index=list(X))

    def predict(self, X):
        pred = X.dot(np.asarray(self.weights))
        return 1 * (pred > 0.5)


class LeastSquaresRegressor:
    """Plain weighted least-squares regressor with optional ridge regularization.

    Parameters
    ----------
    alpha : float, default=0.0
        Ridge regularization strength. When ``alpha > 0`` the estimator
        solves ``min_w  sum_i s_i (y_i - x_i^T w)^2 + alpha ||w||^2``
        instead of unregularized weighted least squares. Even a tiny positive
        ``alpha`` makes the optimization problem strictly convex, which
        guarantees a unique minimizer regardless of the underlying
        BLAS/LAPACK implementation. This is used in tests where the
        unregularized problem can be rank-deficient and would otherwise
        produce non-deterministic predictions across platforms (see
        https://github.com/fairlearn/fairlearn/issues/265).
    """

    def __init__(self, alpha=0.0):
        self.alpha = alpha
        self.weights = None

    def fit(self, X, Y, sample_weight):
        sqrtW = np.sqrt(np.array(sample_weight))
        matX = np.array(X) * sqrtW[:, np.newaxis]
        vecY = np.asarray(Y) * sqrtW
        if self.alpha > 0:
            n_features = matX.shape[1]
            # Augmented system: stacking sqrt(alpha) * I onto the design and
            # zeros onto the target encodes the ridge penalty as an
            # ordinary least-squares problem solvable by np.linalg.lstsq.
            matX = np.vstack([matX, np.sqrt(self.alpha) * np.eye(n_features)])
            vecY = np.concatenate([vecY, np.zeros(n_features)])
        self.lsqinfo = np.linalg.lstsq(matX, vecY, rcond=-1)
        self.weights = pd.Series(self.lsqinfo[0], index=list(X))

    def predict(self, X):
        return X.dot(self.weights)


class MockEstimator:
    def fit(self, X, y, sample_weight):
        pass

    def predict(self, X):
        return np.ones(X.shape[0])
