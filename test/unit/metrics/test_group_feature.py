# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

import fairlearn.metrics as metrics

raw_feature = ["a", "b", "c", "a"]


def test_list():
    target = metrics._group_feature.GroupFeature("Sens", raw_feature, 0, None)

    assert target.name_ == "Sens0"


def test_named_list():
    expected_name = "My Named List"
    target = metrics._group_feature.GroupFeature("Ignored", raw_feature, 2, expected_name)

    assert target.name_ == expected_name


def test_ndarray():
    rf = np.asarray(raw_feature)

    target = metrics._group_feature.GroupFeature("SF", rf, 1, None)

    assert target.name_ == "SF1"


def test_unnamed_Series():
    rf = pd.Series(data=raw_feature)

    target = metrics._group_feature.GroupFeature("SF", rf, 2, None)

    assert target.name_ == "SF2"


def test_named_Series():
    name = "My Feature"
    rf = pd.Series(data=raw_feature, name=name)

    target = metrics._group_feature.GroupFeature("Ignored", rf, 2, None)

    assert target.name_ == name


def test_named_Series_override_name():
    expected_name = "Some Feature"
    series_name = "Unused Name"

    rf = pd.Series(data=raw_feature, name=series_name)
    target = metrics._group_feature.GroupFeature("Not seen", rf, 2, expected_name)

    assert target.name_ == expected_name


def test_Series_int_name():
    rf = pd.Series(data=raw_feature, name=1)

    msg = "Series name must be a string. Value '1' was of type <class 'int'>"
    with pytest.raises(ValueError) as execInfo:
        _ = metrics._group_feature.GroupFeature("Not seen", rf, 2, None)
    assert execInfo.value.args[0] == msg
