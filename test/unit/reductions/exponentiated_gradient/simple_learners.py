# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd


class LeastSquaresBinaryClassifierLearner:
    def __init__(self):
        self.weights = None

    def fit(self, X, Y, sample_weight):
        sqrtW = np.sqrt(sample_weight)
        matX = np.array(X) * sqrtW[:, np.newaxis]
        vecY = Y * sqrtW
        self.lsqinfo = np.linalg.lstsq(matX, vecY, rcond=-1)
        self.weights = pd.Series(self.lsqinfo[0], index=list(X))

    def predict(self, X):
        pred = X.dot(np.asarray(self.weights))
        return 1 * (pred > 0.5)


class LeastSquaresRegressor:
    def __init__(self):
        self.weights = None

    def fit(self, X, Y, sample_weight):
        sqrtW = np.sqrt(sample_weight)
        matX = np.array(X) * sqrtW[:, np.newaxis]
        vecY = Y * sqrtW
        self.lsqinfo = np.linalg.lstsq(matX, vecY, rcond=-1)
        self.weights = pd.Series(self.lsqinfo[0], index=list(X))

    def predict(self, X):
        return X.dot(self.weights)
