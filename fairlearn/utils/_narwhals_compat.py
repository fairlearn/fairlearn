# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

from typing import Any

import narwhals.stable.v1 as nw
from narwhals.dependencies import (
    get_pandas,
    get_polars,
    get_pyarrow,
    is_into_dataframe,
    is_into_series,
)
from narwhals.stable.v1 import Implementation


def get_default_dataframe_backend() -> Implementation:
    """Get the default dataframe backend.

    The backend priority is polars > pandas > pyarrow.
    """
    if get_polars():
        return Implementation.POLARS
    if get_pandas():
        return Implementation.PANDAS
    if get_pyarrow():
        return Implementation.PYARROW
    raise ImportError(
        "No supported dataframe backend found. Please install either polars, pandas or pyarrow."
    )


def get_native_namespace_or_default(
    *maybe_narwhals_objects: Any, default_backend: Implementation | None = None
) -> Any:
    """Try to get the native namespace of the objects.

    If none of the object is a narwhals-supported DataFrame or Series,
    return the default dataframe backend.

    Parameters
    ----------
    maybe_narwhals_objects : Any
        Objects to check for narwhals DataFrame or Series.
    default_implementation : Implementation, optional
        Default implementation to use if no narwhals objects are found.
        If not provided, the default implementation is determined by
        the available backends (polars > pandas > pyarrow).

    Raises
    ------
    ValueError
        If the DataFrame and Series objects are not from the same native namespace.
    """
    narwhals_objects = tuple(
        o for o in maybe_narwhals_objects if is_into_dataframe(o) or is_into_series(o)
    )
    if narwhals_objects:
        objects_namespace = nw.get_native_namespace(*narwhals_objects)
        objects_backend = Implementation.from_native_namespace(objects_namespace)
        if default_backend and objects_backend != default_backend:
            raise ValueError(
                f"The objects implementation ({objects_backend}) does not match the provided default implementation ({default_backend})."
            )
        return objects_namespace
    if default_backend:
        return default_backend.to_native_namespace()
    return get_default_dataframe_backend().to_native_namespace()


def all_to_native(*args) -> tuple:
    """Convert all inputs to native objects if they are narwhals objects.

    Temporary function to use while narwhals is not fully integrated.
    """
    return tuple(nw.to_native(arg, pass_through=True) for arg in args)
