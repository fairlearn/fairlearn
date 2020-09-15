# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

import fairlearn.metrics.experimental as metrics


raw_feature = ['a', 'b', 'c', 'a']
expected_classes = ['a', 'b', 'c']


def common_validations(sf):
    assert np.array_equal(expected_classes, sf.classes)

    mask = sf.get_mask_for_class('a')
    assert np.array_equal(mask,  [True, False, False, True])
    mask_b = sf.get_mask_for_class('b')
    assert np.array_equal(mask_b, [False, True, False, False])
    mask_c = sf.get_mask_for_class('c')
    assert np.array_equal(mask_c, [False, False, True, False])


def test_list():
    target = metrics.GroupFeature("Sens", raw_feature, 0, None)

    assert target.name == "Sens 0"
    common_validations(target)


def test_named_list():
    expected_name = "My Named List"
    target = metrics.GroupFeature("Ignored", raw_feature, 2, expected_name)

    assert target.name == expected_name
    common_validations(target)


def test_ndarray():
    rf = np.asarray(raw_feature)

    target = metrics.GroupFeature("SF", rf, 1, None)

    assert target.name == "SF 1"
    common_validations(target)


def test_unnamed_Series():
    rf = pd.Series(data=raw_feature)

    target = metrics.GroupFeature("SF", rf, 2, None)

    assert target.name == "SF 2"
    common_validations(target)


def test_named_Series():
    name = "My Feature"
    rf = pd.Series(data=raw_feature, name=name)

    target = metrics.GroupFeature("Ignored", rf, 2, None)

    assert target.name == name
    common_validations(target)


def test_named_Series_override_name():
    expected_name = "Some Feature"
    series_name = "Unused Name"

    rf = pd.Series(data=raw_feature, name=series_name)
    target = metrics.GroupFeature("Not seen", rf, 2, expected_name)

    assert target.name == expected_name
    common_validations(target)
