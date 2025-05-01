# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

import logging
from typing import Any, Sequence

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
from narwhals.dependencies import is_into_dataframe
from narwhals.typing import IntoFrameT, IntoSeriesT
from sklearn.utils.validation import check_consistent_length

from fairlearn.utils._fixes import check_array
from fairlearn.utils._narwhals_compat import (
    all_to_native,
    get_native_namespace_or_default,
)

logger = logging.getLogger(__file__)

_KW_CONTROL_FEATURES = "control_features"

_MESSAGE_X_NONE = "Must supply X"
_MESSAGE_Y_NONE = "Must supply nonempty y"
_MESSAGE_SENSITIVE_FEATURES_NONE = "Must specify `sensitive_features` (for now)"
_MESSAGE_X_Y_ROWS = "X and y must have same number of rows"
_MESSAGE_X_SENSITIVE_ROWS = "X and the sensitive features must have same number of rows"
_MESSAGE_RATIO_NOT_IN_RANGE = "ratio must lie between (0,1]"
_INPUT_DATA_FORMAT_ERROR_MESSAGE = (
    "The only allowed input data formats for {} are: {}. Your provided data was of type {}."
)
_EMPTY_INPUT_ERROR_MESSAGE = "At least one of sensitive_features, labels, or scores are empty."
_LABELS_NOT_0_1_ERROR_MESSAGE = "Supplied y labels are not 0 or 1"
_MORE_THAN_ONE_COLUMN_ERROR_MESSAGE = "{} is a {} with more than one column"
_NOT_ALLOWED_TYPE_ERROR_MESSAGE = "{} is not an ndarray, Series or DataFrame"
_NDARRAY_NOT_TWO_DIMENSIONAL_ERROR_MESSAGE = "{} is an ndarray which is not 2D"
_NOT_ALLOWED_MATRIX_TYPE_ERROR_MESSAGE = "{} is not an ndarray or DataFrame"
_INCONSISTENT_ARRAY_LENGTH = "Arrays {} have an inconsistent number of items"

_ALLOWED_INPUT_TYPES_X = [np.ndarray, pd.DataFrame]
_ALLOWED_INPUT_TYPES_SENSITIVE_FEATURES = [np.ndarray, pd.DataFrame, pd.Series, list]
_ALLOWED_INPUT_TYPES_Y = [np.ndarray, pd.DataFrame, pd.Series, list]

_MERGE_COLUMN_SEPARATOR = ","


def _standardize_features(
    X: np.ndarray | IntoFrameT,
    features: np.ndarray | IntoSeriesT | IntoFrameT | list | None,
    *,
    native_namespace: Any,
) -> nw.Series[Any]:
    check_consistent_length(X, features)
    features = check_array(features, ensure_2d=False, dtype=None)

    # compress multiple features into a single column
    if len(features.shape) > 1 and features.shape[1] > 1:
        features = _merge_columns(features)

    return nw.new_series(name="", values=features.squeeze(), native_namespace=native_namespace)


def _validate_and_reformat_input(
    X: np.ndarray | IntoFrameT,
    y: np.ndarray | IntoSeriesT | list | None = None,
    expect_y: bool = True,
    expect_sensitive_features: bool = True,
    enforce_binary_labels: bool = False,
    sensitive_features: np.ndarray | IntoSeriesT | IntoFrameT | list | None = None,
    control_features: np.ndarray | IntoSeriesT | IntoFrameT | list | None = None,
    default_backend: nw.Implementation | None = None,
) -> tuple[np.ndarray | IntoFrameT, IntoSeriesT, IntoSeriesT | None, IntoSeriesT | None]:
    """Validate input data and return the data in an appropriate format.

    Parameters
    ----------
    X : numpy.ndarray, or a DataFrame
        The feature matrix
    y : numpy.ndarray, a DataFrame, a Series, or list
        The label vector. Must be of shape (n,) or (n,1).
    expect_y : bool, default=True
        If True, y needs to be provided and non-empty (np.asarray(y).size > 0),
        otherwise ignores the argument
    expect_sensitive_features : bool
        If true, sensitive_features must be provided to the call to this function.
        This is the default setting.
    enforce_binary_labels : bool
        If True, raise an exception if `y` contains values other than 0 and 1.
        Default is False.
    sensitive_features : numpy.ndarray, a DataFrame, a Series, or list
        The sensitive feature matrix
    control_features : numpy.ndarray, a DataFrame , a Series, or list
        The control feature matrix
    default_backend : narwhals.Implementation, optional
        The default dataframe backend to use if any input data is a dataframe or series.
        If not provided, the default backend is determined by the available backends.

    Returns
    -------
    Tuple(numpy.ndarray or a DataFrame, a Series, a Series or None, a Series or None)
        The validated and reformatted X, y, sensitive_features and control_features; note
        that certain estimators rely on metadata encoded in X which may be stripped during
        the reformatting process, so mitigation methods should ideally use the input X instead
        of the returned X for training estimators and leave potential reformatting of X to the
        estimator.
    """
    if X is None:
        raise ValueError(_MESSAGE_X_NONE)

    plx = get_native_namespace_or_default(X, y, default_backend=default_backend)

    if expect_y:
        if y is None:
            raise ValueError(f"{_MESSAGE_Y_NONE}, got {y=}.")
        y = np.asarray(y)
        if y.size == 0:
            raise ValueError(f"{_MESSAGE_Y_NONE}, got {y=}.")
        if not (y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1)):
            raise ValueError(f"`y` must be of shape (n,) or (n,1), got y of shape=({y.shape}).")
        if enforce_binary_labels and not set(np.unique(y)).issubset(set([0, 1])):
            raise ValueError(_LABELS_NOT_0_1_ERROR_MESSAGE)
        y = check_array(y.reshape(-1), ensure_2d=False, dtype="numeric", ensure_all_finite=False)

    result_X = check_array(X, dtype=None, ensure_all_finite=False, allow_nd=True)
    if is_into_dataframe(X):
        result_X = nw.from_numpy(result_X, native_namespace=plx)

    if (y is not None) and y.shape[0] != result_X.shape[0]:
        raise ValueError(_MESSAGE_X_Y_ROWS)

    if sensitive_features is not None:
        sensitive_features = _standardize_features(X, sensitive_features, native_namespace=plx)
    elif expect_sensitive_features:
        raise ValueError(_MESSAGE_SENSITIVE_FEATURES_NONE)

    # Handle the control features
    if control_features is not None:
        control_features = _standardize_features(X, control_features, native_namespace=plx)

    # If we don't have a y, then need to fiddle with return type to
    # avoid a warning from pandas
    if y is not None:
        result_y = nw.new_series(name="", values=y, native_namespace=plx)
    else:
        result_y = nw.new_series(name="", values=[], dtype=nw.Float64, native_namespace=plx)

    return all_to_native(result_X, result_y, sensitive_features, control_features)


def _merge_columns(feature_columns: np.ndarray) -> np.ndarray:
    """Merge multiple columns into a single new column.

    Compresses multiple columns into a single column by concatenating the
    stringified values. For example, given two columns
    :code:`['A', 'A', 'B', 'B']` and :code:`[4, 5, 4, 5]` the resultant
    column will be
    :code:`['A,4', 'A,5', 'B,4', 'B,5']`.

    Parameters
    ----------
    feature_column : numpy.ndarray
        Multi-dimensional array of columns to be merged

    Returns
    -------
    numpy.ndarray
        One-dimensional array of merged columns
    """
    if not isinstance(feature_columns, np.ndarray):
        raise ValueError(
            f"Received argument of type {type(feature_columns).__name__} instead of expected numpy.ndarray"
        )

    def _join_names(names: Sequence[str]) -> str:
        return _MERGE_COLUMN_SEPARATOR.join(
            [
                name
                # escape backslash and separator
                .replace("\\", "\\\\").replace(
                    _MERGE_COLUMN_SEPARATOR, f"\\{_MERGE_COLUMN_SEPARATOR}"
                )
                for name in names
            ]
        )

    return np.array([_join_names(row) for row in feature_columns.astype(str)])
