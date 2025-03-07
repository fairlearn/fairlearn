# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

import logging
from typing import Sequence

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
from narwhals.typing import IntoFrameT, IntoSeriesT
from sklearn.utils.validation import check_consistent_length

from fairlearn.utils._fixes import check_array
from fairlearn.utils._narwhals_compat import get_common_backend

logger = logging.getLogger(__file__)

_KW_SENSITIVE_FEATURES = "sensitive_features"
_KW_CONTROL_FEATURES = "control_features"

_MESSAGE_X_NONE = "Must supply X"
_MESSAGE_Y_NONE = "Must supply y"
_MESSAGE_SENSITIVE_FEATURES_NONE = "Must specify {0} (for now)".format(_KW_SENSITIVE_FEATURES)
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


def _validate_and_reformat_input(
    X: np.ndarray | IntoFrameT,
    y: np.ndarray | IntoSeriesT | IntoFrameT | list | None = None,
    expect_y: bool = True,
    expect_sensitive_features: bool = True,
    enforce_binary_labels: bool = False,
    **kwargs,
) -> tuple[np.ndarray | IntoFrameT, IntoSeriesT, IntoSeriesT | None, IntoSeriesT | None]:
    """Validate input data and return the data in an appropriate format.

    The :code:`**kwargs` can contain :code:`sensitive_features=` and :code:`control_features=`
    parameters.

    Parameters
    ----------
    X : numpy.ndarray, IntoFrameT
        The feature matrix
    y : numpy.ndarray, IntoFrameT, IntoSeriesT, or list
        The label vector
    expect_y : bool
        If True y needs to be provided, otherwise ignores the argument; default True
    expect_sensitive_features : bool
        If true, sensitive_features must be provided to the call to this function.
        This is the default setting.
    enforce_binary_labels : bool
        If True raise exception if there are more than two distinct
        values in the `y` data; default False

    Returns
    -------
    Tuple(numpy.ndarray | IntoFrameT, IntoSeriesT, IntoSeriesT, IntoSeriesT)
        The validated and reformatted X, y, sensitive_features and control_features; note
        that certain estimators rely on metadata encoded in X which may be stripped during
        the reformatting process, so mitigation methods should ideally use the input X instead
        of the returned X for training estimators and leave potential reformatting of X to the
        estimator.
    """
    if X is None:
        raise ValueError(_MESSAGE_X_NONE)

    X = nw.from_native(X, pass_through=True, eager_only=True)
    # get the native namespace for the X data or use the default
    plx = get_common_backend(X, y).to_native_namespace()
    if y is not None:
        y = nw.from_native(y, pass_through=True, eager_only=True, allow_series=True)
        # calling check_X_y with a 2-dimensional y causes a warning, so ensure it is 1-dimensional
        if isinstance(y, np.ndarray) and len(y.shape) == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        elif isinstance(y, nw.DataFrame) and y.shape[1] == 1:
            y = y.to_numpy().reshape(-1)

        # Using an adapted version of check_array to avoid a warning in sklearn version < 1.6
        y = check_array(y, ensure_2d=False, dtype="numeric", ensure_all_finite=False)
        if enforce_binary_labels and not set(np.unique(y)).issubset(set([0, 1])):
            raise ValueError(_LABELS_NOT_0_1_ERROR_MESSAGE)
    elif expect_y:
        raise ValueError(_MESSAGE_Y_NONE)

    result_X = check_array(X, dtype=None, ensure_all_finite=False, allow_nd=True)
    if isinstance(X, nw.DataFrame):
        result_X = plx.DataFrame(result_X)

    sensitive_features = kwargs.get(_KW_SENSITIVE_FEATURES)
    if sensitive_features is not None:
        check_consistent_length(X, sensitive_features)
        sensitive_features = check_array(sensitive_features, ensure_2d=False, dtype=None)

        # compress multiple sensitive features into a single column
        if len(sensitive_features.shape) > 1 and sensitive_features.shape[1] > 1:
            sensitive_features = _merge_columns(sensitive_features)

        sensitive_features = plx.Series(sensitive_features.squeeze())
    elif expect_sensitive_features:
        raise ValueError(_MESSAGE_SENSITIVE_FEATURES_NONE)

    # Handle the control features
    control_features = kwargs.get(_KW_CONTROL_FEATURES)
    if control_features is not None:
        check_consistent_length(X, control_features)
        control_features = check_array(control_features, ensure_2d=False, dtype=None)

        # compress multiple control features into a single column
        if len(control_features.shape) > 1 and control_features.shape[1] > 1:
            control_features = _merge_columns(control_features)

        control_features = plx.Series(control_features.squeeze())

    # If we don't have a y, then need to fiddle with return type to
    # avoid a warning from pandas
    if y is not None:
        result_y: IntoSeriesT = plx.Series(y)
    else:
        result_y: IntoSeriesT = nw.to_native(
            nw.from_native(plx.Series(), series_only=True).cast(nw.Float64)
        )
    # TODO: this are currently native objects,
    # they can be narwhals objects once the rest of the library supports it
    return (result_X, result_y, sensitive_features, control_features)


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
