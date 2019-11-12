# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

_ARRAY_NOT_REALLY_1D = "'{0}' has more than one dimension longer than 1"


def _ensure_1d_ndarray(input, input_name):
    """Ensures that the input is represented as a 1d numpy.ndarray.

    The goal of this routine is to ensure that input arrays of shape
    (1,n) and (n,1), not to mention (1,1,n,1,1) can all be treated as
    an array of shape (n). However, an array of shape (2,2) will be
    rejected.

    This routine relies on the behaviour of numpy.asarray, and is not
    comprehensive as a result. For example
    numpy.asarray([[1,2], [2]])
    will result in a 1D ndarray, with two elements, each of which is a
    list. This method is not built to detect that issue
    """
    result = np.asarray(input)

    old_shape = np.asarray(result.shape)

    if np.count_nonzero(old_shape > 1) > 1:
        raise ValueError(_ARRAY_NOT_REALLY_1D.format(input_name))

    return result.reshape(old_shape.max())
