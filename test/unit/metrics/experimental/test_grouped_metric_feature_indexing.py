# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

import fairlearn.metrics.experimental as metrics
from .utils import _get_raw_GroupedMetric


gf0 = metrics.GroupFeature('SF', ['a', 'a', 'b', 'b', 'c', 'c'], 0, None)
gf1 = metrics.GroupFeature('SF', ['x', 'x', 'x', 'y', 'y', 'y'], 0, None)


class TestSingleFeatureIndexing:
    def test_group_indices_from_index_0(self):
        target = _get_raw_GroupedMetric()

        result = target._group_indices_from_index(0, [gf0])
        assert np.array_equal(result, [0])
        result = target._group_indices_from_index(1, [gf0])
        assert np.array_equal(result, [1])
        result = target._group_indices_from_index(2, [gf0])
        assert np.array_equal(result, [2])

    def test_group_indices_from_index_1(self):
        target = _get_raw_GroupedMetric()

        result = target._group_indices_from_index(0, [gf1])
        assert np.array_equal(result, [0])
        result = target._group_indices_from_index(1, [gf1])
        assert np.array_equal(result, [1])
    
