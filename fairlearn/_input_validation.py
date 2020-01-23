# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd


_MESSAGE_X_NONE = "Must supply X"
_MESSAGE_Y_NONE = "Must supply y"
_MESSAGE_X_Y_ROWS = "X and y must have same number of rows"
_MESSAGE_X_SENSITIVE_ROWS = "X and the sensitive features must have same number of rows"

_KW_SENSITIVE_FEATURES = "sensitive_features"


def _validate_and_reformat_reductions_input(X, y, enforce_binary_sensitive_feature=False,
                                            **kwargs):
    if X is None:
        raise ValueError(_MESSAGE_X_NONE)

    if y is None:
        raise ValueError(_MESSAGE_Y_NONE)

    if _KW_SENSITIVE_FEATURES not in kwargs:
        msg = "Must specify {0} (for now)".format(_KW_SENSITIVE_FEATURES)
        raise RuntimeError(msg)

    # Extract the target attribute
    sensitive_features_vector = _make_vector(kwargs[_KW_SENSITIVE_FEATURES],
                                             _KW_SENSITIVE_FEATURES)

    '''
    if enforce_binary_sensitive_feature:
        unique_labels = np.unique(sensitive_features_vector)
        if len(unique_labels) > 2:
            raise RuntimeError("Sensitive features contain more than two unique values")
    '''
    # Extract the Y values
    y_vector = _make_vector(y, "y")

    X_rows, _ = _get_matrix_shape(X, "X")
    if X_rows != y_vector.shape[0]:
        raise RuntimeError(_MESSAGE_X_Y_ROWS)
    if X_rows != sensitive_features_vector.shape[0]:
        raise RuntimeError(_MESSAGE_X_SENSITIVE_ROWS)

    return pd.DataFrame(X), y_vector, sensitive_features_vector


def _make_vector(formless, formless_name):
    formed_vector = None
    if isinstance(formless, list):
        formed_vector = pd.Series(formless)
    elif isinstance(formless, pd.DataFrame):
        if len(formless.columns) == 1:
            formed_vector = formless.iloc[:, 0]
        else:
            msgfmt = "{0} is a DataFrame with more than one column"
            raise RuntimeError(msgfmt.format(formless_name))
    elif isinstance(formless, pd.Series):
        formed_vector = formless
    elif isinstance(formless, np.ndarray):
        if len(formless.shape) == 1:
            formed_vector = pd.Series(formless)
        elif len(formless.shape) == 2 and formless.shape[1] == 1:
            formed_vector = pd.Series(formless[:, 0])
        else:
            msgfmt = "{0} is an ndarray with more than one column"
            raise RuntimeError(msgfmt.format(formless_name))
    else:
        msgfmt = "{0} not an ndarray, Series or DataFrame"
        raise RuntimeError(msgfmt.format(formless_name))

    return formed_vector


def _get_matrix_shape(formless, formless_name):
    num_rows = -1
    num_cols = -1

    if isinstance(formless, pd.DataFrame):
        num_cols = len(formless.columns)
        num_rows = len(formless.index)
    elif isinstance(formless, np.ndarray):
        if len(formless.shape) == 2:
            num_rows = formless.shape[0]
            num_cols = formless.shape[1]
        else:
            msgfmt = "{0} is an ndarray which is not 2D"
            raise RuntimeError(msgfmt.format(formless_name))
    else:
        msgfmt = "{0} not an ndarray or DataFrame"
        raise RuntimeError(msgfmt.format(formless_name))
    return num_rows, num_cols
