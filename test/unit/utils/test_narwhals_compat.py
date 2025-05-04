# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from narwhals.stable.v1 import Implementation

from fairlearn.utils._narwhals_compat import (
    all_to_native,
    get_native_namespace_or_default,
)


@pytest.mark.parametrize(
    "objects, expected_backend",
    [
        ([np.array([1, 2, 3])], Implementation.POLARS),
        ([pd.DataFrame({"a": [1, 2], "b": [3, 4]})], Implementation.PANDAS),
        ([pd.DataFrame({"a": [1, 2], "b": [3, 4]}), np.array([1, 2])], Implementation.PANDAS),
    ],
)
def test_get_native_namespace_or_default(objects, expected_backend):
    namespace = get_native_namespace_or_default(*objects)
    print(f"Namespace: {namespace}")
    assert Implementation.from_native_namespace(namespace) == expected_backend


@pytest.mark.parametrize(
    "objects",
    [
        [np.array([1, 2, 3])],
        [pl.DataFrame({"a": [1, 2], "b": [3, 4]})],
    ],
)
def test_get_native_namespace_or_default_with_default(objects):
    namespace = get_native_namespace_or_default(objects, default_backend=Implementation.POLARS)
    assert Implementation.from_native_namespace(namespace) == Implementation.POLARS


def test_get_native_namespace_or_default_error():
    with pytest.raises(ValueError):
        get_native_namespace_or_default(
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            pl.DataFrame({"a": [1, 2], "b": [3, 4]}),
        )
    with pytest.raises(ValueError):
        get_native_namespace_or_default(
            pl.DataFrame({"a": [1, 2], "b": [3, 4]}),
            default_backend=Implementation.PANDAS,
        )


@pytest.mark.parametrize(
    "objects, expected_type",
    [
        ([np.array([1, 2, 3])], np.ndarray),
        ([nw.from_native(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))], pd.DataFrame),
        ([nw.from_native(pl.DataFrame({"a": [1, 2], "b": [3, 4]}))], pl.DataFrame),
        ([pd.DataFrame({"a": [1, 2], "b": [3, 4]})], pd.DataFrame),
    ],
)
def test_all_to_native(objects, expected_type):
    native_objects = all_to_native(*objects)
    for obj in native_objects:
        assert isinstance(obj, expected_type)
