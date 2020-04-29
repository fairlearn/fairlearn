# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

sensitive_features = [str(x) for x in 'AAAAAAA' 'BBBBBBB' 'CCCCCC']
labels =     [int(x) for x in '0110100' '0010111' '001111']  # noqa: E222
X1 =         [int(x) for x in '0110101' '0111101' '001011']  # noqa: E222
X2 =         [int(x) for x in '0000100' '0000011' '111111']  # noqa: E222
X3 =         [int(x) for x in '1111111' '1111111' '111111']  # noqa: E222


def _get_data(A_two_dim=False):
    X = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
    y = pd.Series(labels)

    if A_two_dim:
        # Stacking the same column a few times will result in the identical groups
        # compared to using a single column, therefore results should be the same.
        A = np.stack((sensitive_features, sensitive_features), -1)
    else:
        A = pd.Series(sensitive_features)
    return X, y, A
