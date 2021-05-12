# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

from fairlearn._input_validation import _merge_columns


def ensure_list(X):
    assert X is not None
    if isinstance(X, list):
        return X
    elif isinstance(X, np.ndarray):
        return X.tolist()
    elif isinstance(X, pd.Series):
        return X.tolist()
    elif isinstance(X, pd.DataFrame):
        return X.tolist()
    raise ValueError("Failed to convert to list")


def ensure_list_1d(X):
    assert X is not None
    if isinstance(X, list):
        return X
    elif isinstance(X, np.ndarray):
        return X.squeeze().tolist()
    elif isinstance(X, pd.Series):
        return X.tolist()
    elif isinstance(X, pd.DataFrame):
        return X.tolist()
    raise ValueError("Failed to convert to list")


def ensure_ndarray(X):
    assert X is not None
    if isinstance(X, list):
        return np.asarray(X)
    elif isinstance(X, np.ndarray):
        return X
    elif isinstance(X, pd.Series):
        return np.asarray(X)
    elif isinstance(X, pd.DataFrame):
        return np.asarray(X)
    raise ValueError("Failed to convert to ndarray")


def ensure_ndarray_2d(X):
    assert X is not None
    tmp = ensure_ndarray(X)
    if len(tmp.shape) not in [1, 2]:
        raise ValueError("Requires 1d or 2d array")
    if len(tmp.shape) == 2:
        return tmp
    result = np.expand_dims(tmp, 1)
    assert len(result.shape) == 2
    return result


def ensure_series(X):
    assert X is not None
    if isinstance(X, list):
        return pd.Series(X)
    elif isinstance(X, np.ndarray):
        if len(X.shape) == 1:
            return pd.Series(X)
        if X.shape[1] == 1:
            return pd.Series(X.squeeze())
    elif isinstance(X, pd.Series):
        return X
    elif isinstance(X, pd.DataFrame):
        return pd.Series(X)
    raise ValueError("Failed to convert to Series")


def ensure_dataframe(X):
    assert X is not None
    if isinstance(X, list):
        return pd.DataFrame(X)
    elif isinstance(X, np.ndarray):
        return pd.DataFrame(X)
    elif isinstance(X, pd.Series):
        return pd.DataFrame(X)
    elif isinstance(X, pd.DataFrame):
        return X
    raise ValueError("Failed to convert to Series")


conversions_for_1d = [ensure_list,
                      ensure_ndarray,
                      ensure_ndarray_2d,
                      ensure_series,
                      ensure_dataframe]


def _map_into_single_column(matrix):
    if len(np.array(matrix).shape) == 1:
        return np.array(matrix)

    return _merge_columns(matrix)
