# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fairlearn.grid_search as gs

import numpy as np

import pytest

class TestGridSearch:
    def test_generate_p0_p1_smoke(self):
        Y = [int(x) for x in '0110101' '0111101' '001011']

        p0, p1 = gs._generate_p0_p1(Y)

        assert p0 == (7/21)
        assert p1 == (14/21)