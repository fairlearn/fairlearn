# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from fairlearn._input_validation import _validate_and_reformat_input


def test_input_column_metadata_retained():
    # The idea here is to validate that the categorical data in X1 stays categorical.
    # The validation used to only run check_array from scikit-learn which strips that metadata
    # away, so this test acts as a forcing function to retain that metadata.
    X0 = [1.2, 3.1, 4.5, -1.2, 9.3, 92.4, 33.7]
    X1 = [0,   1,   2,   0,    1,   2,    0]
    X = pd.DataFrame({"X0": X0, "X1": X1})
    X.X1 = X.X1.astype('category')

    X_checked, _, _ = _validate_and_reformat_input(X, sensitive_features=X.X1, expect_y=False)

    assert X_checked.dtypes[0] == np.float64
    # The important part here is that it's 0, 1, 2, and not 0.0, 1.0, and 2.0
    assert X_checked.dtypes[1] == pd.CategoricalDtype(categories=[0, 1, 2])
