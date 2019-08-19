# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fairlearn.grid_search as gs

import numpy as np
import pandas as pd

import pytest

class LeastSquaresLearner:
    def __init__(self):
        self.weights = None

    def fit(self, X, Y, sample_weight):
        sqrtW = np.sqrt(sample_weight)
        matX = np.array(X) * sqrtW[:, np.newaxis]
        vecY = Y * sqrtW
        self.lsqinfo = np.linalg.lstsq(matX, vecY, rcond=-1)
        self.weights = pd.Series(self.lsqinfo[0], index=list(X))

    def predict(self, X):
        pred = X.dot(self.weights)
        return 1*(pred > 0.5)

class TestGridSearch:
    def test_generate_p0_p1_smoke(self):
        Y = [int(x) for x in '0110101' '0111101' '001011']

        p0, p1 = gs._generate_p0_p1(Y)

        assert p0 == 8/20
        assert p1 == 12/20

    def test_generate_p0_p1_nonbinary(self):
        Y = [int(x) for x in "01200011111"]

        with pytest.raises(RuntimeError, match=r"Supplied Y not binary"):
            _, _ = gs._generate_p0_p1(Y)

    def test_generate_p0_p1_not01(self):
        Y = [int(x) for x in "020202220002"]

        with pytest.raises(RuntimeError, match=r"Supplied Y labels not 0 or 1"):
            _, _ = gs._generate_p0_p1(Y)

    def test_generate_weights(self):
        A = [int(x) for x in "0011"]
        Y = [int(x) for x in "0101"]
        L = 10
        p_ratio = 2

        W_expect = [ -21, -19, 9, 11]

        W = gs._generate_weights(Y, A, L, p_ratio)

        assert np.array_equal(W_expect, W), str(W)+str(W_expect)

    def test_grid_smoke(self):
        A = pd.Series([int(x) for x in '0000000' '1111111' '010101'])
        Y = pd.Series([int(x) for x in '0110100' '0010111' '001111'])
        feat1 = [int(x) for x in '0110101' '0111101' '001011']
        feat2 = [int(x) for x in '0000100' '0000011' '111111']
        feat3 = [int(x) for x in '1111111' '1111111' '111111']
        X = pd.DataFrame({"feat1": feat1, "feat2": feat2, "feat3": feat3})

        result = gs.classification_binary_protected_1d(LeastSquaresLearner(), X, Y, A, num_Ls=11)
        assert len(result) == 11
        assert result[5]["lambda"] == 0
        lambdas = [x["lambda"] for x in result]
        assert len(np.unique(lambdas)) == len(result)
        assert sorted(lambdas)

    def test_grid_bad_Ls(self):
        message = r"Must specify either Ls or num_Ls"
        A = pd.Series([int(x) for x in '0000000' '1111111' '010101'])
        Y = pd.Series([int(x) for x in '0110100' '0010111' '001111'])
        feat1 = [int(x) for x in '0110101' '0111101' '001011']
        feat2 = [int(x) for x in '0000100' '0000011' '111111']
        feat3 = [int(x) for x in '1111111' '1111111' '111111']
        X = pd.DataFrame({"feat1": feat1, "feat2": feat2, "feat3": feat3})

        with pytest.raises(RuntimeError, match=message):
            _ = gs.classification_binary_protected_1d(LeastSquaresLearner(), X, Y, A, None, None)
        with pytest.raises(RuntimeError, match=message):
            _ = gs.classification_binary_protected_1d(LeastSquaresLearner(), X, Y, A, [-1,0,1], 3)

    def test_grid_bad_Y_shape(self):
        message = r"Supplied y not 1d vector"

        A = pd.Series([int(x) for x in '0000000' '1111111' '010101'])
        feat1 = [int(x) for x in '0110101' '0111101' '001011']
        feat2 = [int(x) for x in '0000100' '0000011' '111111']
        feat3 = [int(x) for x in '1111111' '1111111' '111111']
        X = pd.DataFrame({"feat1": feat1, "feat2": feat2, "feat3": feat3})

        badY = np.ones((2,3))
        with pytest.raises(RuntimeError, match=message):
            _ = gs.classification_binary_protected_1d(LeastSquaresLearner(), X, badY, A)

    def test_grid_bad_A_shape(self):
        message = r"Supplied protected_attribute not 1d vector"

        Y = pd.Series([int(x) for x in '0110100' '0010111' '001111'])
        feat1 = [int(x) for x in '0110101' '0111101' '001011']
        feat2 = [int(x) for x in '0000100' '0000011' '111111']
        feat3 = [int(x) for x in '1111111' '1111111' '111111']
        X = pd.DataFrame({"feat1": feat1, "feat2": feat2, "feat3": feat3})

        badA = np.ones((2,5))
        with pytest.raises(RuntimeError, match=message):
            _ = gs.classification_binary_protected_1d(LeastSquaresLearner(), X, Y, badA)

    def test_grid_A_Y_mismatch(self):
        message = r"Supplied protected_attribute and y not same length"

        A = pd.Series([int(x) for x in '0000000' '1111111' '010101' '1'])
        Y = pd.Series([int(x) for x in '0110100' '0010111' '001111'])
        feat1 = [int(x) for x in '0110101' '0111101' '001011']
        feat2 = [int(x) for x in '0000100' '0000011' '111111']
        feat3 = [int(x) for x in '1111111' '1111111' '111111']
        X = pd.DataFrame({"feat1": feat1, "feat2": feat2, "feat3": feat3})

        with pytest.raises(RuntimeError, match=message):
            _ = gs.classification_binary_protected_1d(LeastSquaresLearner(), X, Y, A)

    def test_grid_X_Y_mismatch(self):
        message = r"Supplied x and y do not have same number of rows"

        A = pd.Series([int(x) for x in '0000000' '1111111' '010101' '1'])
        Y = pd.Series([int(x) for x in '0110100' '0010111' '001111' '0'])
        feat1 = [int(x) for x in '0110101' '0111101' '001011']
        feat2 = [int(x) for x in '0000100' '0000011' '111111']
        feat3 = [int(x) for x in '1111111' '1111111' '111111']
        X = pd.DataFrame({"feat1": feat1, "feat2": feat2, "feat3": feat3})

        with pytest.raises(RuntimeError, match=message):
            _ = gs.classification_binary_protected_1d(LeastSquaresLearner(), X, Y, A)

    def test_grid_bad_A_labels(self):
        message = r"Supplied protected_attribute labels not 0 or 1"

        A = pd.Series([int(x) for x in '0000000' '2222222' '020202'])
        Y = pd.Series([int(x) for x in '0110100' '0010111' '001111'])
        feat1 = [int(x) for x in '0110101' '0111101' '001011']
        feat2 = [int(x) for x in '0000100' '0000011' '111111']
        feat3 = [int(x) for x in '1111111' '1111111' '111111']
        X = pd.DataFrame({"feat1": feat1, "feat2": feat2, "feat3": feat3})

        with pytest.raises(RuntimeError, match=message):
            _ = gs.classification_binary_protected_1d(LeastSquaresLearner(), X, Y, A)