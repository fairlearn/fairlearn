# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

single_binary_sf = [["a"], ["b"], ["a"], ["a"], ["b"], ["a"], ["b"], ["b"], ["a"], ["b"]]
single_categorical_sf = [["d"], ["e"], ["d"], ["c"], ["e"], ["d"], ["e"], ["c"], ["c"], ["e"]]


@pytest.fixture
def data_X():
    return [[0.], [1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.]]


@pytest.fixture
def data_y():
    return [1, 1, 1, 1, 0, 0, 1, 0, 0, 0]


@pytest.fixture(params=[single_binary_sf, single_categorical_sf])
def data_single_sf(request):
    return request.param


@pytest.fixture
def data_multiple_sf():
    return np.hstack((np.array(single_binary_sf), np.array(single_categorical_sf)))
