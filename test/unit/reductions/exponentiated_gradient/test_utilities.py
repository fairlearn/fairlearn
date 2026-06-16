# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

sensitive_features = [str(x) for x in "AAAAAAABBBBBBBCCCCCC"]
labels = [int(x) for x in "01101000010111001111"]  # noqa: E222
X1 = [int(x) for x in "01101010111101001011"]  # noqa: E222
X2 = [int(x) for x in "00001000000011111111"]  # noqa: E222
X3 = [int(x) for x in "11111111111111111111"]  # noqa: E222
scores = [int(x) / 10 for x in "00011240223346134447"]  # noqa: E222


def _get_data(A_two_dim=False, y_as_scores=False, flip_y=False):
    X = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})

    if y_as_scores:
        y = pd.Series(scores)
    else:  # classification
        y = pd.Series(labels)
        if flip_y:
            y = pd.Series([int(not i) for i in labels])

    if A_two_dim:
        # Stacking the same column a few times will result in the identical groups
        # compared to using a single column, therefore results should be the same.
        A = np.stack((sensitive_features, sensitive_features), -1)
    else:
        A = pd.Series(sensitive_features)
    return X, y, A
