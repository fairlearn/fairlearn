# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals.dependencies import (
    get_pandas,
    get_polars,
    is_into_dataframe,
    is_into_series,
)
from narwhals.stable.v1 import Implementation, nw

if TYPE_CHECKING:
    import numpy as np
    from narwhals.typing import IntoFrameT, IntoSeriesT


def get_default_dataframe_backend() -> Implementation:
    """Get the default dataframe backend.

    The backend priority is polars > pandas > pyarrow.
    """
    if get_polars():
        return Implementation.POLARS
    if get_pandas():
        return Implementation.PANDAS
    raise ImportError(
        "No supported dataframe backend found. Please install either pandas or polars."
    )


def get_common_backend(
    X: np.ndarray | IntoFrameT, y: np.ndarray | IntoSeriesT | IntoFrameT | list | None
) -> Implementation:
    """Get the common backend for the input data.

    Use the backend of the first non-None input data or the default backend if all inputs are None.
    """
    implementation_X = (
        Implementation.from_backend(nw.get_native_namespace(X)) if is_into_dataframe(X) else None
    )
    implementation_y = (
        Implementation.from_backend(nw.get_native_namespace(y))
        if is_into_series(y) or is_into_dataframe(y)
        else None
    )
    reference_impl = implementation_X or implementation_y
    if reference_impl is None:
        return get_default_dataframe_backend()
    return reference_impl
