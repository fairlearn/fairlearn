# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from fairlearn.metrics._group_feature import GroupFeature

pytestmark = pytest.mark.narwhals

raw_feature = ["a", "b", "c", "a"]


def test_list():
    target = GroupFeature("Sens", raw_feature, 0)

    assert target.name_ == "Sens0"


def test_ndarray():
    rf = np.asarray(raw_feature)

    target = GroupFeature("SF", rf, 1)

    assert target.name_ == "SF1"


def test_unnamed_Series():
    rf = pd.Series(data=raw_feature)

    target = GroupFeature("SF", rf, 2)

    assert target.name_ == "SF2"


def test_named_Series(request: pytest.FixtureRequest, constructor):
    if "pyarrow_table" in str(request):
        request.applymarker(pytest.mark.xfail(reason="PyArrow chunked_array's are nameless"))

    name = "My Feature"
    rf = constructor({name: raw_feature})[name]

    target = GroupFeature("Ignored", rf, 2)

    assert target.name_ == name


def test_Series_int_name():
    rf = pd.Series(data=raw_feature, name=1)

    msg = "Series name must be a string. Value '1' was of type <class 'int'>"
    with pytest.raises(ValueError, match=msg):
        GroupFeature("Not seen", rf, 2)
