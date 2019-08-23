# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

import fairlearn.grid_search.binary_protected_attribute.utilities as utilities


class TestUtilities:
    def test_generate_protected_attribute_info_smoke(self):
        number_attributes = 100

        # Fill A with 0s and 1s
        A = np.random.randint(2, size=number_attributes)

        p0, p1, a0_val = utilities.generate_protected_attribute_info(A)

        # Need to handle case where A only contains a single value
        if p0 == 1:
            assert p1 == 0
            assert a0_val == A[0]
        else:
            assert p0 == 1 - sum(A) / number_attributes
            assert p1 == sum(A) / number_attributes
            assert a0_val == 0

    def test_generate_protected_attribute_info_non_numeric(self):
        A = ["AB", "AB", "CD", "AB"]

        p0, p1, a0_val = utilities.generate_protected_attribute_info(A)
        assert p0 == 0.75
        assert p1 == 0.25
        assert a0_val == "AB"

    def test_generate_protected_attribute_info_notbinary(self):
        message = str("Protected Attribute does not have "
                      "exactly two unique values")

        A = np.random.randint(20, size=400)
        with pytest.raises(RuntimeError, match=message):
            _, _, _ = utilities.generate_protected_attribute_info(A)

        A = [1, 1, 1, 1, 1]
        with pytest.raises(RuntimeError, match=message):
            _, _, _ = utilities.generate_protected_attribute_info(A)
