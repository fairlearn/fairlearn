# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np


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
