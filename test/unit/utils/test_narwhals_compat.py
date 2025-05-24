# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from unittest.mock import patch

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from narwhals.stable.v1 import Implementation

from fairlearn.utils._narwhals_compat import (
    all_to_native,
    get_default_dataframe_backend,
    get_native_namespace_or_default,
)


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
    with patch(
        "fairlearn.utils._narwhals_compat.get_polars", return_value=polars_available
    ) as mock_get_polars, patch(
        "fairlearn.utils._narwhals_compat.get_pandas", return_value=pandas_available
    ) as mock_get_pandas, patch(
        "fairlearn.utils._narwhals_compat.get_pyarrow", return_value=pyarrow_available
    ) as mock_get_pyarrow:
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
    with patch("fairlearn.utils._narwhals_compat.get_polars", return_value=False), patch(
        "fairlearn.utils._narwhals_compat.get_pandas", return_value=False
    ), patch("fairlearn.utils._narwhals_compat.get_pyarrow", return_value=False):
        with pytest.raises(ImportError) as excinfo:
            get_default_dataframe_backend()
        assert "No supported dataframe backend found." in str(excinfo.value)


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
