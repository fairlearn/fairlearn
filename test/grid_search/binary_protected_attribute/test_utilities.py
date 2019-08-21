# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

import fairlearn.grid_search.binary_protected_attribute.utilities as utilities


class TestUtilities:
    def test_generate_protected_attribute_info_smoke(self):
        number_attributes = 4
        A = np.random.randint(2, size=number_attributes)

        p0, p1, a0_val = utilities.generate_protected_attribute_info(A)

        assert p0 == 1 - sum(A) / number_attributes
        assert p1 == sum(A) / number_attributes
        # Need to handle case were array is all 1s
        if p0 == 1:
            assert a0_val == A[0]
        else:
            assert a0_val == 0

    def test_generate_protected_attribute_info_unary(self):
        A = [12, 12, 12]

        p0, p1, a0_val = utilities.generate_protected_attribute_info(A)

        assert p0 == 1
        assert p1 == 0
        assert a0_val == 12

    def test_generate_protected_attribute_info_non_numeric(self):
        A = ["AB", "AB", "CD", "AB"]

        p0, p1, a0_val = utilities.generate_protected_attribute_info(A)
        assert p0 == 0.75
        assert p1 == 0.25
        assert a0_val == "AB"

    def test_generate_protected_attribute_info_notbinary(self):
        A = np.random.randint(20, size=400)

        message = "Protected Attribute is not binary"
        with pytest.raises(RuntimeError, match=message):
            _, _, _ = utilities.generate_protected_attribute_info(A)
