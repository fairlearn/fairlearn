# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
from fairlearn._input_validation import _validate_and_reformat_input


def test_input_column_metadata_retained():
    X0 = [1.2, 3.1, 4.5, -1.2, 9.3, 92.4, 33.7]
    X1 = [0,   1,   2,   0,    1,   2,    0]
    X = pd.DataFrame({"X0": X0, "X1": X1})
    X.X1 = X.X1.astype('category')