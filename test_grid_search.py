# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fairlearn.grid_search as gs

import numpy as np

import pytest

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