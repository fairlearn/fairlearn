# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals.dependencies import (
    get_pandas,
    get_polars,
    is_into_dataframe,
    is_into_series,
)
from narwhals.stable.v1 import Implementation, nw

if TYPE_CHECKING:
    import numpy as np
    from narwhals.typing import IntoFrame, IntoSeries


def get_default_dataframe_backend() -> Implementation:
    """Get the default dataframe backend.

    The backend priority is polars > pandas.
    """
    if get_polars():
        return Implementation.POLARS
    if get_pandas():
        return Implementation.PANDAS
    raise ImportError(
        "No supported dataframe backend found. Please install either pandas or polars."
    )


def get_native_namespace_or_default(
    X: np.ndarray | IntoFrame, y: np.ndarray | IntoSeries | IntoFrame | list | None
) -> Any:
    narwhals_objects = tuple(o for o in (X, y) if is_into_dataframe(o) or is_into_series(o))
    if narwhals_objects:
        return nw.get_native_namespace(*narwhals_objects)
    return get_default_dataframe_backend().to_native_namespace()


def all_to_native(*args) -> tuple:
    """Convert all inputs to native objects if they are narwhals objects.

    Temporary function to use while narwhals is not fully integrated.
    """
    return tuple(nw.to_native(arg, pass_through=True) for arg in args)
