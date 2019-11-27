# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd


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
    if len(tmp.shape) != 1:
        raise ValueError("Requires 1d array")
    result = np.expand_dims(tmp, 1)
    assert len(result.shape) == 2
    return result


def ensure_series(X):
    assert X is not None
    if isinstance(X, list):
        return pd.Series(X)
    elif isinstance(X, np.ndarray):
        return pd.Series(X)
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
