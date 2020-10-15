# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np

import fairlearn.metrics as metrics
from .utils import _get_raw_MetricFrame


gf0 = metrics._group_feature.GroupFeature('SF', ['a', 'a', 'b', 'b', 'c', 'c'], 0, None)
gf1 = metrics._group_feature.GroupFeature('SF', ['x', 'y', 'x', 'y', 'x', 'y'], 0, None)


class TestSingleFeatureIndexing:
    def test_mask_from_tuple_0(self):
        target = _get_raw_MetricFrame()

        result = target._mask_from_tuple(('a',), [gf0])
        assert np.array_equal(result, [True, True, False, False, False, False])
        result = target._mask_from_tuple(('b',), [gf0])
        assert np.array_equal(result, [False, False, True, True, False, False])
        result = target._mask_from_tuple(('c',), [gf0])
        assert np.array_equal(result, [False, False, False, False, True, True])


class TestTwoFeatureIndexing:
    def test_mask_from_tuple(self):
        target = _get_raw_MetricFrame()

        result = target._mask_from_tuple(('a', 'x'), [gf0, gf1])
        assert np.array_equal(result, [True, False, False, False, False, False])
        result = target._mask_from_tuple(('a', 'y'), [gf0, gf1])
        assert np.array_equal(result, [False, True, False, False, False, False])
        result = target._mask_from_tuple(('b', 'x'), [gf0, gf1])
        assert np.array_equal(result, [False, False, True, False, False, False])
        result = target._mask_from_tuple(('b', 'y'), [gf0, gf1])
        assert np.array_equal(result, [False, False, False, True, False, False])
        result = target._mask_from_tuple(('c', 'x'), [gf0, gf1])
        assert np.array_equal(result, [False, False, False, False, True, False])
        result = target._mask_from_tuple(('c', 'y'), [gf0, gf1])
        assert np.array_equal(result, [False, False, False, False, False, True])
