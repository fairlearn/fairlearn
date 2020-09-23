# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np

_ARRAY_NOT_1D = "Supplied input array has more than one non-trivial dimension"


def _convert_to_ndarray_and_squeeze(target):
    """Convert input to a `numpy.ndarray` and calls squeeze (to dispose of unit length dimensions).

    There is a special case to stop single element arrays being converted to scalars.
    """
    result = np.asarray(target)

    if result.size > 1:
        result = np.squeeze(result)
    else:
        result = result.reshape(1)

    return result


def _convert_to_ndarray_1d(target):
    """Convert the input to a `numpy.ndarray` with a single dimension.

    If the input is multi-dimension, this will raise an exception
    """
    result = _convert_to_ndarray_and_squeeze(target)

    if len(result.shape) > 1:
        raise ValueError(_ARRAY_NOT_1D)

    return result
