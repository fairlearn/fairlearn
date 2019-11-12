# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np


def _convert_to_1d_array(input):
    result = np.asarray(input)

    if len(result) > 1:
        result = np.squeeze(result)
    else:
        result = result.reshape(1)

    return result
