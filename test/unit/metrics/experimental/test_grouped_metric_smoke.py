# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pytest

import fairlearn.metrics.experimental as metrics
from test.unit.input_convertors import conversions_for_1d


# ===========================================================


def mock_func(y_true, y_pred):
    return np.sum(y_true)


def mock_func_weight(y_true, y_pred, sample_weight):
    return np.sum(np.multiply(y_true, sample_weight))


def mock_func_extra_arg(y_true, y_pred, my_arg=1):
    return np.sum([my_arg * y for y in y_true])


def mock_func_matrix_return(y_true, y_pred):
    return np.ones([len(y_true), sum(y_pred)])
