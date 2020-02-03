# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_X_y, check_consistent_length, check_array


_KW_SENSITIVE_FEATURES = "sensitive_features"

_MESSAGE_X_NONE = "Must supply X"
_MESSAGE_Y_NONE = "Must supply y"
_MESSAGE_SENSITIVE_FEATURES_NONE = "Must specify {0} (for now)".format(_KW_SENSITIVE_FEATURES)
_MESSAGE_X_Y_ROWS = "X and y must have same number of rows"
_MESSAGE_X_SENSITIVE_ROWS = "X and the sensitive features must have same number of rows"
_INPUT_DATA_FORMAT_ERROR_MESSAGE = "The only allowed input data formats for {} are: {}. " \
                                     "Your provided data was of type {}."
_EMPTY_INPUT_ERROR_MESSAGE = "At least one of sensitive_features, labels, or scores are empty."
_SENSITIVE_FEATURES_NON_BINARY_ERROR_MESSAGE = "Sensitive features contain more than two unique" \
                                               " values"
_LABELS_NOT_0_1_ERROR_MESSAGE = "Supplied y labels are not 0 or 1"
_MORE_THAN_ONE_COLUMN_ERROR_MESSAGE = "{} is a {} with more than one column"
_NOT_ALLOWED_TYPE_ERROR_MESSAGE = "{} is not an ndarray, Series or DataFrame"
_NDARRAY_NOT_TWO_DIMENSIONAL_ERROR_MESSAGE = "{} is an ndarray which is not 2D"
_NOT_ALLOWED_MATRIX_TYPE_ERROR_MESSAGE = "{} is not an ndarray or DataFrame"

_ALLOWED_INPUT_TYPES_X = [np.ndarray, pd.DataFrame]
_ALLOWED_INPUT_TYPES_SENSITIVE_FEATURES = [np.ndarray, pd.DataFrame, pd.Series, list]
_ALLOWED_INPUT_TYPES_Y = [np.ndarray, pd.DataFrame, pd.Series, list]


def _validate_and_reformat_input(X, y, expect_y=True, enforce_binary_sensitive_feature=False,
                                 enforce_binary_labels=False, **kwargs):
    """Validate input data and return the data in an appropriate format.

    :param X: The feature matrix
    :type X: numpy.ndarray or pandas.DataFrame
    :param y: The label vector
    :type y: numpy.ndarray, pandas.DataFrame, pandas.Series, or list
    :param expect_y: if True y needs to be provided, otherwise ignores the argument; default True
    :type expect_y: bool
    :param enforce_binary_sensitive_feature: if True raise exception if there are more than two
        distinct values in the `sensitive_features` data from `kwargs`; default False
    :type enforce_binary_sensitive_feature: bool
    :param enforce_binary_labels: if True raise exception if there are more than two distinct
        values in the `y` data; default False
    :type enforce_binary_labels: bool
    """
    if X is None:
        raise ValueError(_MESSAGE_X_NONE)

    if y is None and expect_y:
        raise ValueError(_MESSAGE_Y_NONE)
    else:
        X = check_array(X)

    sensitive_features = kwargs.get(_KW_SENSITIVE_FEATURES)
    if sensitive_features is None:
        raise ValueError(_MESSAGE_SENSITIVE_FEATURES_NONE)

    sensitive_features = kwargs.get(_KW_SENSITIVE_FEATURES)
    if sensitive_features is None:
        raise ValueError(_MESSAGE_SENSITIVE_FEATURES_NONE)

    if type(X) not in _ALLOWED_INPUT_TYPES_X:
        raise TypeError(_INPUT_DATA_FORMAT_ERROR_MESSAGE
                        .format(X.__name__, _ALLOWED_INPUT_TYPES_X, type(X).__name__))

    if type(sensitive_features) not in _ALLOWED_INPUT_TYPES_SENSITIVE_FEATURES:
        raise TypeError(_INPUT_DATA_FORMAT_ERROR_MESSAGE
                        .format(_KW_SENSITIVE_FEATURES, _ALLOWED_INPUT_TYPES_SENSITIVE_FEATURES,
                                type(sensitive_features).__name__))

    if expect_y and type(y) not in _ALLOWED_INPUT_TYPES_Y:
        raise TypeError(_INPUT_DATA_FORMAT_ERROR_MESSAGE
                        .format(y.__name__, _ALLOWED_INPUT_TYPES_Y, type(y).__name__))

    if len(X) == 0 or len(sensitive_features) == 0 or (expect_y and len(y) == 0):
        raise ValueError(_EMPTY_INPUT_ERROR_MESSAGE)

    # Extract the sensitive features. If there are multiple sensitive features compress them into
    # a single column which will serve as the binning vector.
    sensitive_features_vector = _make_vector(sensitive_features,
                                             _KW_SENSITIVE_FEATURES,
                                             compress_multiple_columns=True)

    if enforce_binary_sensitive_feature:
        if len(np.unique(sensitive_features_vector)) > 2:
            raise ValueError(_SENSITIVE_FEATURES_NON_BINARY_ERROR_MESSAGE)

    if expect_y and enforce_binary_labels:
        if not set(np.unique(y)).issubset(set([0, 1])):
            raise ValueError(_LABELS_NOT_0_1_ERROR_MESSAGE)

    X_rows, _ = _get_matrix_shape(X, "X")

    if expect_y:
        y_vector = _make_vector(y, "y")
        if X_rows != y_vector.shape[0]:
            raise ValueError(_MESSAGE_X_Y_ROWS)
    else:
        y_vector = None

    if X_rows != sensitive_features_vector.shape[0]:
        raise ValueError(_MESSAGE_X_SENSITIVE_ROWS)

    return pd.DataFrame(X), y_vector, sensitive_features_vector


def _make_vector(formless, formless_name, compress_multiple_columns=False):
    formed_vector = None
    if isinstance(formless, list):
        formed_vector = pd.Series(formless)
    elif isinstance(formless, pd.DataFrame):
        if len(formless.columns) == 1:
            formed_vector = formless.iloc[:, 0]
        elif compress_multiple_columns:
            formed_vector = formless.apply(
                lambda row: "-".join(
                    [str(row[col]).replace("-", "_") for col in formless.columns]),
                axis=1)
        else:
            raise ValueError(_MORE_THAN_ONE_COLUMN_ERROR_MESSAGE
                             .format(formless_name, pd.DataFrame.__name__))
    elif isinstance(formless, pd.Series):
        formed_vector = formless
    elif isinstance(formless, np.ndarray):
        if len(formless.shape) == 1:
            formed_vector = pd.Series(formless)
        elif len(formless.shape) == 2 and formless.shape[1] == 1:
            formed_vector = pd.Series(formless[:, 0])
        elif compress_multiple_columns:
            formed_vector = np.apply_along_axis(
                lambda row: ",".join([str(row[i])
                                      .replace("\\", "\\\\")  # escape backslash
                                      .replace(",", "\\,")  # escape commas using backslash
                                      for i in range(len(row))]),
                axis=1,
                arr=formless)
        else:
            raise ValueError(_MORE_THAN_ONE_COLUMN_ERROR_MESSAGE
                             .format(formless_name, np.ndarray.__name__))
    else:
        raise ValueError(_NOT_ALLOWED_TYPE_ERROR_MESSAGE.format(formless_name))

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
            raise ValueError(_NDARRAY_NOT_TWO_DIMENSIONAL_ERROR_MESSAGE.format(formless_name))
    else:
        raise ValueError(_NOT_ALLOWED_MATRIX_TYPE_ERROR_MESSAGE.format(formless_name))
    return num_rows, num_cols
