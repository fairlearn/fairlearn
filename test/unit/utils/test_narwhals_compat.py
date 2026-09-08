# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from importlib.util import find_spec
from unittest.mock import patch

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import pytest
from narwhals.stable.v1 import Implementation

from fairlearn.utils._narwhals_compat import (
    all_to_native,
    get_default_dataframe_backend,
    get_native_namespace_or_default,
)

POLARS_AVAILABLE = find_spec("polars") is not None

if POLARS_AVAILABLE:
    import polars as pl

DEFAULT_BACKEND = Implementation.POLARS if POLARS_AVAILABLE else Implementation.PANDAS


@pytest.mark.parametrize(
    "polars_available, pandas_available, pyarrow_available, expected_backend",
    [
        # Polars is available (preferred backend)
        (True, True, True, Implementation.POLARS),
        # Pandas fallback when polars is not available
        (False, True, True, Implementation.PANDAS),
        # PyArrow fallback when polars and pandas are not available
        (False, False, True, Implementation.PYARROW),
    ],
)
def test_get_default_dataframe_backend(
    polars_available, pandas_available, pyarrow_available, expected_backend
):
    with (
        patch(
            "fairlearn.utils._narwhals_compat.get_polars", return_value=polars_available
        ) as mock_get_polars,
        patch(
            "fairlearn.utils._narwhals_compat.get_pandas", return_value=pandas_available
        ) as mock_get_pandas,
        patch(
            "fairlearn.utils._narwhals_compat.get_pyarrow", return_value=pyarrow_available
        ) as mock_get_pyarrow,
    ):
        backend = get_default_dataframe_backend()

    assert backend == expected_backend
    mock_get_polars.assert_called_once()

    if expected_backend in {Implementation.PANDAS, Implementation.PYARROW}:
        mock_get_pandas.assert_called_once()
    else:
        mock_get_pandas.assert_not_called()

    if expected_backend == Implementation.PYARROW:
        mock_get_pyarrow.assert_called_once()
    else:
        mock_get_pyarrow.assert_not_called()


def test_get_default_dataframe_backend_error():
    with (
        patch("fairlearn.utils._narwhals_compat.get_polars", return_value=False),
        patch("fairlearn.utils._narwhals_compat.get_pandas", return_value=False),
        patch("fairlearn.utils._narwhals_compat.get_pyarrow", return_value=False),
    ):
        with pytest.raises(ImportError) as excinfo:
            get_default_dataframe_backend()
        assert "No supported dataframe backend found." in str(excinfo.value)


@pytest.mark.parametrize(
    "objects, expected_backend",
    [
        ([np.array([1, 2, 3])], DEFAULT_BACKEND),
        ([pd.DataFrame({"a": [1, 2], "b": [3, 4]})], Implementation.PANDAS),
        ([pd.DataFrame({"a": [1, 2], "b": [3, 4]}), np.array([1, 2])], Implementation.PANDAS),
    ],
)
def test_get_native_namespace_or_default(objects, expected_backend):
    namespace = get_native_namespace_or_default(*objects)
    assert Implementation.from_native_namespace(namespace) == expected_backend


NAMESPACE_WITH_DEFAULT_CASES = [
    ([np.array([1, 2, 3])], Implementation.PANDAS, Implementation.PANDAS),
]

if POLARS_AVAILABLE:
    NAMESPACE_WITH_DEFAULT_CASES.append(
        ([pl.DataFrame({"a": [1, 2], "b": [3, 4]})], Implementation.POLARS, Implementation.POLARS)
    )


@pytest.mark.parametrize(
    "objects, default_backend, expected_backend",
    NAMESPACE_WITH_DEFAULT_CASES,
)
def test_get_native_namespace_or_default_with_default(objects, default_backend, expected_backend):
    namespace = get_native_namespace_or_default(*objects, default_backend=default_backend)
    assert Implementation.from_native_namespace(namespace) == expected_backend


def test_get_native_namespace_or_default_error():
    with pytest.raises(ValueError):
        get_native_namespace_or_default(
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            default_backend=Implementation.POLARS,
        )

    if not POLARS_AVAILABLE:
        return

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


ALL_TO_NATIVE_CASES = [
    ([np.array([1, 2, 3])], np.ndarray),
    ([nw.from_native(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))], pd.DataFrame),
    ([pd.DataFrame({"a": [1, 2], "b": [3, 4]})], pd.DataFrame),
]

if POLARS_AVAILABLE:
    ALL_TO_NATIVE_CASES.append(
        ([nw.from_native(pl.DataFrame({"a": [1, 2], "b": [3, 4]}))], pl.DataFrame)
    )


@pytest.mark.parametrize(
    "objects, expected_type",
    ALL_TO_NATIVE_CASES,
)
def test_all_to_native(objects, expected_type):
    native_objects = all_to_native(*objects)
    for obj in native_objects:
        assert isinstance(obj, expected_type)
