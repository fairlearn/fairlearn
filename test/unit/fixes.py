# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import sklearn


def get_sklearn_expected_1d_message():
    # Handle change of message for sklearn
    if sklearn.__version__ < "0.23.0":
        expected_message = "bad input shape"
    else:
        expected_message = "y should be a 1d array"
    return expected_message
