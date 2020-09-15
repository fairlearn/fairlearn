# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

import fairlearn.metrics.experimental as metrics
from .utils import _get_raw_GroupedMetric


gf0 = metrics.GroupFeature('SF', ['a', 'a', 'b', 'b', 'c', 'c'], 0, None)
gf1 = metrics.GroupFeature('SF', ['x', 'y', 'x', 'y', 'x', 'y'], 0, None)


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

    def test_group_tuple_from_index_0(self):
        target = _get_raw_GroupedMetric()

        result = target._group_tuple_from_indices([0], [gf0])
        assert result == ('a',)
        result = target._group_tuple_from_indices([1], [gf0])
        assert result == ('b',)
        result = target._group_tuple_from_indices([2], [gf0])
        assert result == ('c',)

    def test_mask_from_indices_0(self):
        target = _get_raw_GroupedMetric()

        result = target._mask_from_indices([0], [gf0])
        assert np.array_equal(result, [True, True, False, False, False, False])
        result = target._mask_from_indices([1], [gf0])
        assert np.array_equal(result, [False, False, True, True, False, False])
        result = target._mask_from_indices([2], [gf0])
        assert np.array_equal(result, [False, False, False, False, True, True])


class TestTwoFeatureIndexing:
    def test_group_indices_from_index(self):
        target = _get_raw_GroupedMetric()

        result = target._group_indices_from_index(0, [gf0, gf1])
        assert np.array_equal(result, [0, 0])
        result = target._group_indices_from_index(1, [gf0, gf1])
        assert np.array_equal(result, [0, 1])
        result = target._group_indices_from_index(2, [gf0, gf1])
        assert np.array_equal(result, [1, 0])
        result = target._group_indices_from_index(3, [gf0, gf1])
        assert np.array_equal(result, [1, 1])
        result = target._group_indices_from_index(4, [gf0, gf1])
        assert np.array_equal(result, [2, 0])
        result = target._group_indices_from_index(5, [gf0, gf1])
        assert np.array_equal(result, [2, 1])

    def test_group_tuple_from_index(self):
        target = _get_raw_GroupedMetric()

        result = target._group_tuple_from_indices([0, 0], [gf0, gf1])
        assert result == ('a', 'x')
        result = target._group_tuple_from_indices([0, 1], [gf0, gf1])
        assert result == ('a', 'y')
        result = target._group_tuple_from_indices([1, 0], [gf0, gf1])
        assert result == ('b', 'x')
        result = target._group_tuple_from_indices([1, 1], [gf0, gf1])
        assert result == ('b', 'y')
        result = target._group_tuple_from_indices([2, 0], [gf0, gf1])
        assert result == ('c', 'x')
        result = target._group_tuple_from_indices([2, 1], [gf0, gf1])
        assert result == ('c', 'y')

    def test_mask_from_indices(self):
        target = _get_raw_GroupedMetric()

        result = target._mask_from_indices([0, 0], [gf0, gf1])
        assert np.array_equal(result, [True, False, False, False, False, False])
        result = target._mask_from_indices([0, 1], [gf0, gf1])
        assert np.array_equal(result, [False, True, False, False, False, False])
        result = target._mask_from_indices([1, 0], [gf0, gf1])
        assert np.array_equal(result, [False, False, True, False, False, False])
        result = target._mask_from_indices([1, 1], [gf0, gf1])
        assert np.array_equal(result, [False, False, False, True, False, False])
        result = target._mask_from_indices([2, 0], [gf0, gf1])
        assert np.array_equal(result, [False, False, False, False, True, False])
        result = target._mask_from_indices([2, 1], [gf0, gf1])
        assert np.array_equal(result, [False, False, False, False, False, True])
