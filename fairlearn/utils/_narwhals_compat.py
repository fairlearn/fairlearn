# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Any

from narwhals.dependencies import (
    get_pandas,
    get_polars,
    get_pyarrow,
    is_into_dataframe,
    is_into_series,
)
from narwhals.stable.v1 import Implementation, nw


def get_default_dataframe_backend() -> Implementation:
    """Get the default dataframe backend.

    The backend priority is polars > pandas.
    """
    if get_polars():
        return Implementation.POLARS
    if get_pandas():
        return Implementation.PANDAS
    if get_pyarrow():
        return Implementation.PYARROW
    raise ImportError(
        "No supported dataframe backend found. Please install either pandas or polars."
    )


def get_native_namespace_or_default(*maybe_narwhals_objects: Any) -> Any:
    """Try to get the native namespace of the objects.

    If none of the object is a narwhals-supported DataFrame or Series,
    return the default dataframe backend.

    Raises
    ------
    ValueError
        If the DataFrame and Series objects are not from the same native namespace.
    """
    narwhals_objects = tuple(
        o for o in maybe_narwhals_objects if is_into_dataframe(o) or is_into_series(o)
    )
    if narwhals_objects:
        return nw.get_native_namespace(*narwhals_objects)
    return get_default_dataframe_backend().to_native_namespace()


def all_to_native(*args) -> tuple:
    """Convert all inputs to native objects if they are narwhals objects.

    Temporary function to use while narwhals is not fully integrated.
    """
    return tuple(nw.to_native(arg, pass_through=True) for arg in args)
