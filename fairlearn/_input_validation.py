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
_MESSAGE_RATIO_NOT_IN_RANGE = "ratio must lie between (0,1]"
_UTILITY_NOT_SPECIFIED = "At least one of the utility is not correctly specified."
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

_SENSITIVE_FEATURE_COMPRESSION_SEPARATOR = ","


def _validate_and_reformat_input(X, y=None, expect_y=True, enforce_binary_sensitive_feature=False,
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
    :return: the validated and reformatted X, y, and sensitive_features; note that certain
        estimators rely on metadata encoded in X which may be stripped during the reformatting
        process, so mitigation methods should ideally use the input X instead of the returned X
        for training estimators and leave potential reformatting of X to the estimator.
    :rtype: (pandas.DataFrame, pandas.Series, pandas.Series)
    """
    if y is not None:
        # calling check_X_y with a 2-dimensional y causes a warning, so ensure it is 1-dimensional
        if isinstance(y, np.ndarray) and len(y.shape) == 2 and y.shape[1] == 1:
            y = y.squeeze()
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.to_numpy().squeeze()

        X, y = check_X_y(X, y)
        y = check_array(y, ensure_2d=False, dtype='numeric')
        if enforce_binary_labels and not set(np.unique(y)).issubset(set([0, 1])):
            raise ValueError(_LABELS_NOT_0_1_ERROR_MESSAGE)
    elif expect_y:
        raise ValueError(_MESSAGE_Y_NONE)
    else:
        X = check_array(X)

    sensitive_features = kwargs.get(_KW_SENSITIVE_FEATURES)
    if sensitive_features is None:
        raise ValueError(_MESSAGE_SENSITIVE_FEATURES_NONE)

    check_consistent_length(X, sensitive_features)
    sensitive_features = check_array(sensitive_features, ensure_2d=False, dtype=None)

    # compress multiple sensitive features into a single column
    if len(sensitive_features.shape) > 1 and sensitive_features.shape[1] > 1:
        sensitive_features = \
            _compress_multiple_sensitive_features_into_single_column(sensitive_features)

    if enforce_binary_sensitive_feature:
        if len(np.unique(sensitive_features)) > 2:
            raise ValueError(_SENSITIVE_FEATURES_NON_BINARY_ERROR_MESSAGE)

    return pd.DataFrame(X), pd.Series(y), pd.Series(sensitive_features.squeeze())


def _compress_multiple_sensitive_features_into_single_column(sensitive_features):
    """Compress multiple sensitive features into a single column.

    The resulting mapping converts multiple dimensions into the Cartesian product of the
    individual columns.

    :param sensitive_features: multi-dimensional array of sensitive features
    :type sensitive_features: `numpy.ndarray`
    :return: one-dimensional array of mapped sensitive features
    """
    if not isinstance(sensitive_features, np.ndarray):
        raise ValueError("Received argument of type {} instead of expected numpy.ndarray"
                         .format(type(sensitive_features).__name__))
    return np.apply_along_axis(
        lambda row: _SENSITIVE_FEATURE_COMPRESSION_SEPARATOR.join(
            [str(row[i])
                .replace("\\", "\\\\")  # escape backslash and separator
                .replace(_SENSITIVE_FEATURE_COMPRESSION_SEPARATOR,
                         "\\" + _SENSITIVE_FEATURE_COMPRESSION_SEPARATOR)
                for i in range(len(row))]),
        axis=1,
        arr=sensitive_features)
